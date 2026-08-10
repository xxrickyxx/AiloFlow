import assert from 'assert';
import { test } from 'node:test';
import { MODEL_CATALOG, QUANT_BITS, estimateWeightBytes } from '../core/models/catalog.js';
import { estimatePerformance } from '../core/estimator/performance_model.js';
import { discoverComputerProfile } from '../core/hardware/discovery.js';

test('catalog entries are internally consistent', () => {
  assert.ok(MODEL_CATALOG.length >= 30, 'the catalogue should offer a real choice of models');

  const ids = new Set<string>();
  for (const entry of MODEL_CATALOG) {
    assert.ok(!ids.has(entry.id), `duplicate catalogue id: ${entry.id}`);
    ids.add(entry.id);

    assert.ok(entry.totalParamsB > 0, `${entry.id}: total parameters must be positive`);
    assert.ok(
      entry.activeParamsB > 0 && entry.activeParamsB <= entry.totalParamsB,
      `${entry.id}: active parameters must be between 0 and the total`
    );

    // The architecture label has to agree with the numbers, otherwise the
    // performance estimate silently models the wrong thing.
    if (entry.architecture === 'dense') {
      assert.strictEqual(
        entry.activeParamsB,
        entry.totalParamsB,
        `${entry.id}: a dense model activates all of its parameters`
      );
    } else {
      assert.ok(
        entry.activeParamsB < entry.totalParamsB,
        `${entry.id}: a MoE model must activate fewer parameters than it holds`
      );
    }

    assert.ok(entry.ggufRepos.length > 0, `${entry.id}: needs at least one GGUF repository`);
    for (const repo of entry.ggufRepos) {
      assert.match(repo, /^[\w.-]+\/[\w.-]+$/, `${entry.id}: "${repo}" is not a valid repo id`);
    }

    assert.ok(entry.contextLength >= 2048, `${entry.id}: implausible context length`);
    assert.ok(entry.license.length > 0, `${entry.id}: license must be stated`);
    assert.ok(entry.notes.length > 0, `${entry.id}: notes must explain the trade-off`);
    assert.ok(['gigante', 'grande', 'medio', 'piccolo'].includes(entry.sizeClass));
  }
});

test('size class matches the parameter count', () => {
  for (const entry of MODEL_CATALOG) {
    const expected =
      entry.totalParamsB >= 200 ? 'gigante'
      : entry.totalParamsB >= 70 ? 'grande'
      : entry.totalParamsB >= 14 ? 'medio'
      : 'piccolo';
    assert.strictEqual(entry.sizeClass, expected, `${entry.id} is labelled ${entry.sizeClass} at ${entry.totalParamsB}B`);
  }
});

test('weight size scales with the quantisation table', () => {
  const q4 = estimateWeightBytes(100, 'Q4_K_M');
  const q8 = estimateWeightBytes(100, 'Q8_0');
  const f16 = estimateWeightBytes(100, 'F16');

  assert.ok(q4 < q8 && q8 < f16, 'more bits per weight must mean more bytes');
  // 100B parameters at 16 bits is 200 GB; a rounding-tolerant check.
  assert.ok(Math.abs(f16 - 200e9) < 1e9);
  assert.ok(QUANT_BITS.Q4_K_M > 4 && QUANT_BITS.Q4_K_M < 5);
});

test('a MoE model is estimated faster than a dense model of the same size', async () => {
  const profile = await discoverComputerProfile();
  const inputs = {
    profile,
    ramBandwidthMBps: 30000,
    storageBandwidthMBps: 2000,
    storageBandwidthIsCacheInfluenced: false,
    contextUsed: 8192,
  };

  const base = { quantization: 'Q4_K_M', layers: 96, contextLength: 131072 } as const;
  const dense = estimatePerformance(
    { ...base, name: 'dense', totalParamsB: 300, activeParamsB: 300, architecture: 'dense' },
    inputs
  );
  const moe = estimatePerformance(
    { ...base, name: 'moe', totalParamsB: 300, activeParamsB: 32, architecture: 'moe' },
    inputs
  );

  assert.ok(dense.overlappedTokensPerSecond !== null && moe.overlappedTokensPerSecond !== null);
  assert.ok(
    moe.overlappedTokensPerSecond! > dense.overlappedTokensPerSecond!,
    'fewer active parameters must translate into a higher predicted rate'
  );
  // Both are far too large for this machine, so storage has to be the limit.
  assert.strictEqual(dense.bottleneckTier, 'STORAGE');
  assert.strictEqual(moe.bottleneckTier, 'STORAGE');
  assert.strictEqual(dense.fitsInMemory, false);
});

test('doubling storage bandwidth roughly doubles the predicted rate', async () => {
  const profile = await discoverComputerProfile();
  const spec = {
    name: 'moe',
    totalParamsB: 300,
    activeParamsB: 32,
    quantization: 'Q4_K_M',
    layers: 96,
    contextLength: 131072,
    architecture: 'moe' as const,
  };

  const slow = estimatePerformance(spec, {
    profile,
    ramBandwidthMBps: 30000,
    storageBandwidthMBps: 2000,
    storageBandwidthIsCacheInfluenced: false,
  });
  const fast = estimatePerformance(spec, {
    profile,
    ramBandwidthMBps: 30000,
    storageBandwidthMBps: 4000,
    storageBandwidthIsCacheInfluenced: false,
  });

  const ratio = fast.overlappedTokensPerSecond! / slow.overlappedTokensPerSecond!;
  assert.ok(ratio > 1.8 && ratio <= 2.05, `expected near-linear scaling, got ${ratio.toFixed(2)}x`);
});

test('a cache-influenced storage figure is reported as a warning', async () => {
  const profile = await discoverComputerProfile();
  const estimate = estimatePerformance(
    {
      name: 'moe',
      totalParamsB: 300,
      activeParamsB: 32,
      quantization: 'Q4_K_M',
      layers: 96,
      contextLength: 131072,
      architecture: 'moe',
    },
    {
      profile,
      ramBandwidthMBps: 30000,
      storageBandwidthMBps: 18000,
      storageBandwidthIsCacheInfluenced: true,
    }
  );

  assert.ok(
    estimate.warnings.some((w) => w.toLowerCase().includes('cache')),
    'an estimate built on page-cache numbers must say so'
  );
  assert.notStrictEqual(estimate.confidence, 'measured');
});
