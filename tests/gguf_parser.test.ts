import assert from 'assert';
import { test } from 'node:test';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { parseGgufHeader } from '../core/model/gguf_parser.js';
import { isGgufFile } from '../core/model/model_registry.js';
import { writeSyntheticGguf } from './helpers/gguf_writer.js';

function scratchDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-gguf-'));
}

test('parses a real GGUF binary header', () => {
  const dir = scratchDir();
  try {
    const written = writeSyntheticGguf(path.join(dir, 'model.gguf'), { blockCount: 6, elementsPerTensor: 128 });
    const meta = parseGgufHeader(written.filePath);

    assert.strictEqual(meta.architecture, 'llama');
    assert.strictEqual(meta.modelName, 'synthetic-test-model');
    assert.strictEqual(meta.blockCount, 6);
    assert.strictEqual(meta.contextLength, 2048);
    assert.strictEqual(meta.headCount, 4);
    assert.strictEqual(meta.headCountKv, 2);
    assert.strictEqual(meta.tensorCount, 6);
    assert.strictEqual(meta.quantization, 'F32');

    // Offsets must land exactly on the data the writer emitted.
    assert.strictEqual(meta.tensorDataOffset, written.tensorDataOffset);
    assert.strictEqual(meta.totalTensorDataBytes, written.totalTensorBytes);
    assert.strictEqual(meta.tensors[0].layerIndex, 0);
    assert.strictEqual(meta.tensors[5].layerIndex, 5);

    // Read one tensor back through its computed absolute offset and confirm
    // the payload matches what was written for that layer.
    const fd = fs.openSync(written.filePath, 'r');
    const buf = Buffer.alloc(meta.tensors[3].sizeBytes);
    fs.readSync(fd, buf, 0, buf.length, meta.tensors[3].absoluteOffset);
    fs.closeSync(fd);
    assert.ok(buf.every((byte) => byte === 4), 'layer 3 payload should be filled with byte 4');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('rejects a missing file instead of returning placeholder metadata', () => {
  assert.throws(
    () => parseGgufHeader(path.join(os.tmpdir(), 'does-not-exist-ailoflow.gguf')),
    /not found or unreadable/i
  );
});

test('a split model reports the whole model, not just its first file', async () => {
  const dir = scratchDir();
  try {
    // Two parts named the way Hugging Face publishes models over 50 GB.
    const partA = writeSyntheticGguf(path.join(dir, 'big-00001-of-00002.gguf'), {
      blockCount: 4,
      elementsPerTensor: 256,
    });
    const partB = writeSyntheticGguf(path.join(dir, 'big-00002-of-00002.gguf'), {
      blockCount: 6,
      elementsPerTensor: 256,
    });

    const { discoverFileModels, inspectModel } = await import('../core/model/model_registry.js');
    const { updateConfig, loadConfig } = await import('../core/config/config.js');

    const originalDirs = loadConfig().modelDirectories;
    updateConfig({ modelDirectories: [dir] });

    try {
      const models = discoverFileModels([dir]);

      // The two files are one model, not two.
      assert.strictEqual(models.length, 1, 'split parts must collapse into a single entry');
      const model = models[0];
      assert.strictEqual(model.splitParts?.length, 2);
      assert.strictEqual(model.complete, true);

      const singleFileSize = fs.statSync(partA.filePath).size;
      const bothFilesSize = singleFileSize + fs.statSync(partB.filePath).size;
      assert.strictEqual(model.fileSizeBytes, bothFilesSize, 'listed size covers every part');

      const inspection = await inspectModel(model.id);

      // The heart of the matter: reading only part one would report 4 tensors
      // and half the bytes, understating the model by the split factor.
      assert.strictEqual(inspection.tensorCount, partA.tensors.length + partB.tensors.length);
      assert.strictEqual(inspection.fileSizeBytes, bothFilesSize);
      assert.strictEqual(
        inspection.totalTensorDataBytes,
        partA.totalTensorBytes + partB.totalTensorBytes
      );
      assert.strictEqual(inspection.splitPartCount, 2);
      assert.ok(inspection.tensorCount > partA.tensors.length, 'must exceed a single part');
    } finally {
      updateConfig({ modelDirectories: originalDirs });
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('an incomplete split set is not runnable', async () => {
  const dir = scratchDir();
  try {
    // Part 2 of 3 is missing: the set advertises three files but holds two.
    writeSyntheticGguf(path.join(dir, 'partial-00001-of-00003.gguf'), { blockCount: 2 });
    writeSyntheticGguf(path.join(dir, 'partial-00003-of-00003.gguf'), { blockCount: 2 });

    const { discoverFileModels } = await import('../core/model/model_registry.js');
    const models = discoverFileModels([dir]);

    assert.strictEqual(models.length, 1);
    assert.strictEqual(models[0].complete, false, 'a set missing parts is incomplete');
    assert.deepStrictEqual(models[0].runnableWith, [], 'an incomplete set cannot be loaded');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('rejects a non-GGUF file', () => {
  const dir = scratchDir();
  try {
    const bogus = path.join(dir, 'notamodel.gguf');
    fs.writeFileSync(bogus, 'this is plain text, not a model');

    assert.strictEqual(isGgufFile(bogus), false);
    assert.throws(() => parseGgufHeader(bogus), /Invalid GGUF magic/i);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
