import assert from 'assert';
import { test } from 'node:test';
import { TelemetryMonitor } from '../core/telemetry/telemetry.js';
import { discoverComputerProfile } from '../core/hardware/discovery.js';

test('an idle snapshot reports unknowns as null, not as numbers', async () => {
  const profile = await discoverComputerProfile();
  const monitor = new TelemetryMonitor();
  await monitor.sample();

  const snapshot = monitor.generateSnapshot(profile);

  // Live host counters are always available.
  assert.ok(snapshot.cpuUsagePercent >= 0 && snapshot.cpuUsagePercent <= 100);
  assert.ok(snapshot.ramTotalBytes > 0);
  assert.ok(snapshot.ramUsagePercent > 0 && snapshot.ramUsagePercent <= 100);

  // Nothing has generated yet, so every inference figure must be null.
  assert.strictEqual(snapshot.generation.active, false);
  assert.strictEqual(snapshot.generation.tokensPerSecond, null);
  assert.strictEqual(snapshot.generation.firstTokenLatencyMs, null);
  assert.strictEqual(snapshot.generation.completionTokens, null);
  assert.strictEqual(snapshot.generation.modelId, null);

  // The storage pipeline has not run, so it reports no throughput at all.
  assert.strictEqual(snapshot.storage.bandwidthMBps, null);
  assert.strictEqual(snapshot.storage.iops, null);
  assert.strictEqual(snapshot.cache, null);
  assert.strictEqual(snapshot.prefetch, null);

  // And with nothing measured, there is no bottleneck verdict to give.
  assert.strictEqual(snapshot.bottleneck.type, 'IDLE');
});

test('CPU utilisation is derived from tick deltas across samples', async () => {
  const profile = await discoverComputerProfile();
  const monitor = new TelemetryMonitor();

  await monitor.sample();
  // Burn a little CPU so the second sample has non-idle ticks to observe.
  const until = Date.now() + 120;
  let acc = 0;
  while (Date.now() < until) acc += Math.sqrt(acc + 1);
  assert.ok(acc >= 0);

  await monitor.sample();
  const snapshot = monitor.generateSnapshot(profile);

  assert.ok(Number.isFinite(snapshot.cpuUsagePercent));
  assert.ok(snapshot.cpuUsagePercent >= 0 && snapshot.cpuUsagePercent <= 100);
});

test('a storage bottleneck is only claimed against measured bandwidth', async () => {
  const profile = await discoverComputerProfile();
  const monitor = new TelemetryMonitor();
  await monitor.sample();

  const snapshot = monitor.generateSnapshot(profile, {
    generationActive: true,
    lastGeneration: {
      promptTokens: 10,
      completionTokens: 100,
      firstTokenLatencyMs: 120,
      tokensPerSecond: 20,
      promptTokensPerSecond: 200,
      totalDurationMs: 5000,
      backendId: 'test',
      modelId: 'test:model',
    },
    lastGenerationAt: new Date().toISOString(),
    bytesPerTokenStreamed: 4 * 1024 * 1024 * 1024, // 4 GB per token: extreme demand
  });

  if (profile.measuredStorageReadBandwidthMBps === null) {
    // Without a benchmark there is no bandwidth to compare against, so STORAGE
    // must not be blamed on a guess.
    assert.notStrictEqual(snapshot.bottleneck.type, 'STORAGE');
  } else {
    assert.strictEqual(snapshot.bottleneck.type, 'STORAGE');
    assert.ok(snapshot.bottleneck.requestedBandwidthMBps! > snapshot.bottleneck.availableBandwidthMBps!);
  }
});
