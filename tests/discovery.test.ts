import assert from 'assert';
import { test } from 'node:test';
import {
  discoverCpu,
  discoverRam,
  discoverGpus,
  discoverStorage,
  discoverComputerProfile,
} from '../core/hardware/discovery.js';
import { calculateOptimalConfiguration } from '../core/hardware/optimizer.js';

test('CPU discovery reports verified capabilities only', async () => {
  const cpu = await discoverCpu();

  assert.ok(cpu.model.length > 0, 'CPU model should be reported');
  assert.ok(cpu.logicalThreads > 0, 'logical threads should be > 0');
  // physicalCores is null when the OS does not expose it — never a guess.
  assert.ok(cpu.physicalCores === null || cpu.physicalCores > 0);
  assert.ok(cpu.baseFrequencyGHz === null || cpu.baseFrequencyGHz > 0);

  // The recommended kernel must be backed by a detected instruction set.
  const map: Record<string, keyof typeof cpu.instructions | null> = {
    AVX512: 'avx512',
    AVX2: 'avx2',
    AVX: 'avx',
    NEON: 'neon',
    GENERIC: null,
  };
  const required = map[cpu.recommendedOptimization];
  if (required) {
    assert.strictEqual(cpu.instructions[required], true, `${cpu.recommendedOptimization} implies ${required}`);
  }
});

test('RAM discovery returns live figures', async () => {
  const ram = await discoverRam();

  assert.ok(ram.totalBytes > 0);
  assert.ok(ram.freeBytes >= 0 && ram.freeBytes <= ram.totalBytes);
  assert.strictEqual(ram.usedBytes, ram.totalBytes - ram.freeBytes);
  assert.ok(ram.systemLoadPercent >= 0 && ram.systemLoadPercent <= 100);
  // Module details are optional; when present they must be plausible.
  assert.ok(ram.speedMHz === null || ram.speedMHz > 0);
  assert.ok(ram.moduleCount === null || ram.moduleCount > 0);
});

test('GPU discovery never fabricates a device or its VRAM', async () => {
  const gpus = await discoverGpus();

  // Zero GPUs is a legitimate result: no placeholder device is invented.
  for (const gpu of gpus) {
    assert.ok(gpu.model.length > 0);
    assert.ok(gpu.supportedBackends.length > 0);
    assert.ok(gpu.supportedBackends.includes(gpu.recommendedBackend));

    if (gpu.vramTotalBytes === null) {
      assert.strictEqual(gpu.vramSource, 'unknown', 'unknown VRAM must be labelled as such');
    } else {
      assert.ok(gpu.vramTotalBytes > 0);
      assert.notStrictEqual(gpu.vramSource, 'unknown');
    }
  }
});

test('storage discovery labels estimates separately from measurements', async () => {
  const drives = await discoverStorage();

  for (const drive of drives) {
    assert.ok(drive.mountPoint.length > 0);
    assert.ok(drive.totalSizeBytes > 0);
    assert.ok(drive.freeSizeBytes <= drive.totalSizeBytes);

    const perf = drive.performanceProfile;
    assert.ok(perf, 'each drive carries a performance profile');
    assert.strictEqual(typeof perf!.measured, 'boolean');
    if (perf!.measured) {
      assert.ok(perf!.measuredAt, 'measured profiles record when they were taken');
    }
  }
});

test('optimizer derives configuration from the real profile', async () => {
  const profile = await discoverComputerProfile();
  const config = calculateOptimalConfiguration(profile);

  assert.ok(config.ramCacheBytes > 0, 'RAM cache should be budgeted');
  assert.ok(config.ramCacheBytes <= profile.ram.totalBytes, 'cache cannot exceed installed RAM');
  assert.ok(config.threadCount > 0 && config.threadCount <= profile.cpu.logicalThreads);
  assert.ok(config.prefetchDepthLayers >= 1);
  assert.strictEqual(typeof config.basedOnMeasuredStorage, 'boolean');

  // With no GPU (or no readable VRAM) there is no VRAM budget to claim.
  const primary = profile.gpus[0];
  if (!primary || primary.vramFreeBytes === null) {
    assert.strictEqual(config.vramCacheBytes, 0);
  }

  // Aggregate bandwidth is null until at least one drive is benchmarked.
  const anyMeasured = profile.storageDrives.some((d) => d.performanceProfile?.measured);
  if (!anyMeasured) {
    assert.strictEqual(profile.measuredStorageReadBandwidthMBps, null);
  }
});
