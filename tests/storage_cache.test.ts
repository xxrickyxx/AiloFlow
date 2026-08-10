import assert from 'assert';
import { test } from 'node:test';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { AiloStorageFabric } from '../core/storage/storage_fabric.js';
import { HierarchicalCache } from '../core/cache/hierarchical_cache.js';
import { discoverStorage } from '../core/hardware/discovery.js';

test('storage fabric performs real reads and counts real bytes', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-fabric-'));
  try {
    const drives = await discoverStorage();
    const fabric = new AiloStorageFabric(drives, dir);
    assert.strictEqual(fabric.getDriveCount(), drives.length);

    // A real file with a known pattern; the fabric must return exactly it.
    const payload = Buffer.alloc(64 * 1024, 0x37);
    fs.writeFileSync(path.join(dir, 'shard.bin'), payload);

    const driveId = drives[0]?.id || 'drive-0';
    const chunk = await fabric.readShardBlock(driveId, 'shard.bin', 1024, 4096);

    assert.strictEqual(chunk.length, 4096);
    assert.ok(chunk.every((b) => b === 0x37), 'returned bytes must match the file contents');

    const stats = fabric.getIoStats();
    assert.strictEqual(stats.totalBytesRead, 4096);
    assert.strictEqual(stats.totalReads, 1);
    // Device rate reflects the read itself; window throughput averages in the
    // idle remainder of the window, so only the former must be positive here.
    assert.ok(stats.averageRequestMBps > 0, 'per-request rate is measured over a real read');
    assert.ok(stats.currentBandwidthMBps >= 0);
    assert.ok(stats.iops > 0, 'a completed read counts toward IOPS');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a missing shard file is an error, not synthetic data', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-fabric-miss-'));
  try {
    const fabric = new AiloStorageFabric([], dir);
    await assert.rejects(() => fabric.readShardBlock('drive-0', 'nope.bin', 0, 1024));

    const stats = fabric.getIoStats();
    assert.strictEqual(stats.totalBytesRead, 0, 'a failed read must not be counted as throughput');
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a short read is reported rather than silently padded', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ailoflow-fabric-short-'));
  try {
    fs.writeFileSync(path.join(dir, 'tiny.bin'), Buffer.alloc(100, 1));
    const fabric = new AiloStorageFabric([], dir);

    await assert.rejects(() => fabric.readShardBlock('drive-0', 'tiny.bin', 0, 4096), /Short read/i);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('hierarchical cache promotes, evicts and tracks hit rate', () => {
  const oneMb = 1024 * 1024;
  // 2MB VRAM budget, 4MB RAM budget forces real eviction decisions.
  const cache = new HierarchicalCache(2 * oneMb, 4 * oneMb);

  for (let i = 0; i < 6; i++) {
    cache.put(`blk.${i}.attn_q.weight`, i, Buffer.alloc(oneMb, i), 'L1_RAM');
  }

  const hit = cache.get('blk.5.attn_q.weight');
  assert.ok(hit, 'the most recent tensor should still be resident');
  assert.strictEqual(hit.layerIndex, 5);
  // Reading from RAM promotes into the VRAM tier.
  assert.strictEqual(hit.tier, 'L0_VRAM');

  const evicted = cache.get('blk.0.attn_q.weight');
  assert.ok(evicted, 'evicted tensors demote to the SSD tier rather than vanish');

  const miss = cache.get('blk.99.attn_q.weight');
  assert.strictEqual(miss, undefined);

  const metrics = cache.getCacheMetrics();
  assert.strictEqual(metrics.misses, 1);
  assert.ok(metrics.hits >= 2);
  assert.ok(metrics.ramUsedBytes <= metrics.ramLimitBytes, 'RAM budget must be respected');
  assert.ok(metrics.vramUsedBytes <= metrics.vramLimitBytes, 'VRAM budget must be respected');
  assert.ok(metrics.hitRatePercent > 0 && metrics.hitRatePercent < 100);
});

test('an untouched cache reports no hit rate rather than a perfect one', () => {
  const cache = new HierarchicalCache(1024, 1024);
  const metrics = cache.getCacheMetrics();
  assert.strictEqual(metrics.hits, 0);
  assert.strictEqual(metrics.misses, 0);
  assert.strictEqual(metrics.hitRatePercent, 0);
});
