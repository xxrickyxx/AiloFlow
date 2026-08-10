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
