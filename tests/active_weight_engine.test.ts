import assert from 'assert';
import { test } from 'node:test';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { SFlowContainer, SFlowManifest } from '../formats/sflow/sflow_format.js';
import { AiloStorageFabric } from '../core/storage/storage_fabric.js';
import { HierarchicalCache } from '../core/cache/hierarchical_cache.js';
import { PrefetchEngine } from '../core/prefetch/prefetch_engine.js';
import { AiloActiveWeightEngine } from '../core/engine/active_weight_engine.js';
import { AiloHierarchicalBackend } from '../inference/ailo_hierarchical/ailo_hierarchical_backend.js';

test('AiloActiveWeightEngine fetches layer weights dynamically and streams tokens', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-active-engine-'));
  try {
    const shardPath = path.join(dir, 'shard-0.bin');
    const shardData = Buffer.alloc(128 * 1024, 0x42);
    fs.writeFileSync(shardPath, shardData);

    const manifest: SFlowManifest = {
      formatVersion: '1.0',
      modelName: 'Test-Active-Model',
      architecture: 'llama',
      parameterCountBillions: 1.0,
      quantization: 'Q4_K_M',
      totalSizeBytes: shardData.length,
      blockCount: 4,
      contextLength: 2048,
      createdAt: new Date().toISOString(),
      shards: [
        {
          shardId: 'shard-0',
          layerStart: 0,
          layerEnd: 4,
          tensorCount: 4,
          sizeBytes: shardData.length,
          targetDriveId: 'drive-0',
          targetMountPoint: dir,
          relFilePath: 'shard-0.bin',
          checksum: 'abc',
          priority: 'HIGH',
          accessFrequency: 1.0,
        },
      ],
      tensorMap: [
        { name: 'blk.0.attn_q.weight', layerIndex: 0, shardId: 'shard-0', offsetInShard: 0, sizeBytes: 1024 },
        { name: 'blk.1.attn_q.weight', layerIndex: 1, shardId: 'shard-0', offsetInShard: 1024, sizeBytes: 1024 },
        { name: 'blk.2.attn_q.weight', layerIndex: 2, shardId: 'shard-0', offsetInShard: 2048, sizeBytes: 1024 },
        { name: 'blk.3.attn_q.weight', layerIndex: 3, shardId: 'shard-0', offsetInShard: 3072, sizeBytes: 1024 },
      ],
    };

    const containerPath = path.join(dir, 'model.sflow');
    fs.writeFileSync(containerPath, JSON.stringify(manifest, null, 2));

    const fabric = new AiloStorageFabric([], dir);
    const cache = new HierarchicalCache(10 * 1024 * 1024, 20 * 1024 * 1024);
    const prefetcher = new PrefetchEngine(cache, fabric, manifest, 2);
    const engine = new AiloActiveWeightEngine(manifest, fabric, cache, prefetcher);

    const stepResult = await engine.generateStep([1, 2, 3], 0, { temperature: 0.7 });

    assert.ok(stepResult.tokenId >= 0);
    assert.ok(stepResult.tokenText.length > 0);
    assert.ok(stepResult.weightMetrics.tensorsLoaded > 0);
    assert.strictEqual(stepResult.stepIndex, 0);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('AiloHierarchicalBackend streams text generation for .sflow containers', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-backend-test-'));
  try {
    const shardPath = path.join(dir, 'shard-0.bin');
    fs.writeFileSync(shardPath, Buffer.alloc(64 * 1024, 0x11));

    const manifest: SFlowManifest = {
      formatVersion: '1.0',
      modelName: 'Hierarchical-Stream-Model',
      architecture: 'qwen2',
      parameterCountBillions: 0.5,
      quantization: 'Q4_0',
      totalSizeBytes: 64 * 1024,
      blockCount: 2,
      contextLength: 1024,
      createdAt: new Date().toISOString(),
      shards: [
        {
          shardId: 'shard-0',
          layerStart: 0,
          layerEnd: 2,
          tensorCount: 2,
          sizeBytes: 64 * 1024,
          targetDriveId: 'drive-0',
          targetMountPoint: dir,
          relFilePath: 'shard-0.bin',
          checksum: 'def',
          priority: 'HIGH',
          accessFrequency: 1.0,
        },
      ],
      tensorMap: [
        { name: 'blk.0.attn_q.weight', layerIndex: 0, shardId: 'shard-0', offsetInShard: 0, sizeBytes: 512 },
        { name: 'blk.1.attn_q.weight', layerIndex: 1, shardId: 'shard-0', offsetInShard: 512, sizeBytes: 512 },
      ],
    };

    const containerPath = path.join(dir, 'model.sflow');
    fs.writeFileSync(containerPath, JSON.stringify(manifest, null, 2));

    const backend = new AiloHierarchicalBackend();
    const isAvailable = await backend.checkAvailability();
    assert.strictEqual(isAvailable.available, true);

    const initialized = await backend.initialize(containerPath);
    assert.strictEqual(initialized, true);

    const streamedTokens: string[] = [];
    const result = await backend.generateStream(
      {
        prompt: 'Explain quantum computing',
        maxTokens: 5,
      },
      (token) => {
        streamedTokens.push(token.token);
      }
    );

    assert.strictEqual(streamedTokens.length, 5);
    assert.ok(result.text.length > 0);
    assert.strictEqual(result.metrics.completionTokens, 5);
    assert.strictEqual(result.metrics.backendId, 'ailo-hierarchical');
    assert.ok((result.metrics.tokensPerSecond || 0) > 0);

    await backend.dispose();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
