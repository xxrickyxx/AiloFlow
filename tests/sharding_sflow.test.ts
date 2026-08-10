import assert from 'assert';
import { test } from 'node:test';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { parseGgufHeader } from '../core/model/gguf_parser.js';
import { createShardedSFlowModel } from '../core/sharding/shard_manager.js';
import { discoverComputerProfile } from '../core/hardware/discovery.js';
import { SFlowContainer } from '../formats/sflow/sflow_format.js';
import { AiloStreamingPipeline } from '../inference/custom_stream/stream_runner.js';
import { writeSyntheticGguf } from './helpers/gguf_writer.js';

function scratchDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-sflow-'));
}

test('shards a real GGUF into a .sflow container with intact tensor bytes', async () => {
  const dir = scratchDir();
  try {
    const written = writeSyntheticGguf(path.join(dir, 'model.gguf'), { blockCount: 8, elementsPerTensor: 512 });
    const meta = parseGgufHeader(written.filePath);
    const profile = await discoverComputerProfile();
    const outDir = path.join(dir, 'sflow');

    const progressEvents: number[] = [];
    const container = await createShardedSFlowModel(meta, profile.storageDrives, outDir, (p) =>
      progressEvents.push(p.percent)
    );

    assert.ok(fs.existsSync(container.sflowPath), '.sflow manifest should be written');
    assert.ok(progressEvents.length > 0, 'sharding should report progress');
    assert.strictEqual(container.manifest.blockCount, 8);
    assert.strictEqual(container.manifest.tensorMap.length, meta.tensorCount);

    // Every shard file must actually hold the bytes the manifest declares.
    const validation = container.validateShards(outDir);
    assert.strictEqual(validation.valid, true, `container should validate: ${validation.problems.join('; ')}`);
    assert.strictEqual(validation.shardsPresent, validation.shardsExpected);

    // Reloading from disk must reproduce the same manifest.
    const reloaded = SFlowContainer.load(container.sflowPath);
    assert.strictEqual(reloaded.manifest.modelName, 'model.gguf');
    assert.ok(reloaded.manifest.shards.length > 0);

    // Spot-check that a tensor's bytes survived the copy unchanged: layer 5 was
    // written as byte value 6 by the generator.
    const tensor = reloaded.manifest.tensorMap.find((t) => t.layerIndex === 5)!;
    const shard = reloaded.manifest.shards.find((s) => s.shardId === tensor.shardId)!;
    const shardPath = path.join(outDir, shard.relFilePath);
    const fd = fs.openSync(shardPath, 'r');
    const buf = Buffer.alloc(tensor.sizeBytes);
    fs.readSync(fd, buf, 0, buf.length, tensor.offsetInShard);
    fs.closeSync(fd);
    assert.ok(buf.every((byte) => byte === 6), 'sharded tensor bytes must match the source GGUF');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('layer sweep streams every layer off disk and reports measured figures', async () => {
  const dir = scratchDir();
  try {
    const written = writeSyntheticGguf(path.join(dir, 'model.gguf'), { blockCount: 6, elementsPerTensor: 1024 });
    const meta = parseGgufHeader(written.filePath);
    const profile = await discoverComputerProfile();
    const outDir = path.join(dir, 'sflow');
    const container = await createShardedSFlowModel(meta, profile.storageDrives, outDir);

    const pipeline = new AiloStreamingPipeline(profile);
    await pipeline.load(container.sflowPath);
    const result = await pipeline.runLayerSweep();
    pipeline.dispose();

    assert.deepStrictEqual(result.errors, [], 'no read errors expected');
    assert.strictEqual(result.layersCompleted, 6);
    assert.strictEqual(result.totalBytesRequested, written.totalTensorBytes);
    // Cache is cold on the first sweep, so every byte comes off storage.
    assert.strictEqual(result.bytesReadFromStorage, written.totalTensorBytes);
    assert.ok(result.durationMs > 0, 'duration must be measured');
    assert.ok(result.effectiveBandwidthMBps > 0, 'bandwidth must be measured');
    assert.strictEqual(result.bytesPerTokenStreamed, written.totalTensorBytes);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('refuses to shard when the source GGUF is absent', async () => {
  const profile = await discoverComputerProfile();

  const fakeMeta = {
    filename: 'ghost.gguf',
    filePath: path.join(os.tmpdir(), 'ailoflow-ghost-model.gguf'),
    fileSizeBytes: 1024,
    version: 3,
    tensorCount: 1,
    metadataKvCount: 0,
    architecture: 'llama',
    modelName: 'ghost',
    contextLength: 2048,
    embeddingLength: 64,
    blockCount: 1,
    headCount: 4,
    headCountKv: 4,
    quantization: 'F32',
    parameterCountBillions: 0,
    tensorDataOffset: 0,
    totalTensorDataBytes: 1024,
    estimatedRamRequiredBytes: 1024,
    estimatedVramRequiredBytes: 256,
    estimatedStorageRequiredBytes: 1024,
    metadataMap: new Map<string, unknown>(),
    tensors: [],
  };

  await assert.rejects(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    () => createShardedSFlowModel(fakeMeta as any, profile.storageDrives, path.join(os.tmpdir(), 'ailoflow-ghost-out')),
    /source GGUF not found/i
  );
});

test('a stub .sflow container fails validation instead of loading', () => {
  const dir = scratchDir();
  try {
    // A manifest that advertises far more data than the shard file contains —
    // exactly the shape of a demo fixture.
    const manifest = {
      formatVersion: '1.0' as const,
      modelName: 'Fake-120B.gguf',
      architecture: 'llama',
      parameterCountBillions: 120,
      quantization: 'Q4_K_M',
      totalSizeBytes: 60_000_000_000,
      blockCount: 2,
      contextLength: 8192,
      createdAt: new Date().toISOString(),
      shards: [
        {
          shardId: 'shard_000',
          layerStart: 0,
          layerEnd: 1,
          tensorCount: 1,
          sizeBytes: 30_000_000_000,
          targetDriveId: 'drive-0',
          targetMountPoint: 'C:/',
          relFilePath: 'shards/shard_000.bin',
          checksum: 'deadbeef',
          priority: 'HIGH' as const,
          accessFrequency: 1,
        },
      ],
      tensorMap: [],
    };

    fs.mkdirSync(path.join(dir, 'shards'), { recursive: true });
    fs.writeFileSync(path.join(dir, 'shards', 'shard_000.bin'), 'stub');
    const sflowPath = path.join(dir, 'fake.sflow');
    fs.writeFileSync(sflowPath, JSON.stringify(manifest));

    const container = SFlowContainer.load(sflowPath);
    const validation = container.validateShards();

    assert.strictEqual(validation.valid, false);
    assert.ok(validation.problems[0].includes('shard_000'));
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
