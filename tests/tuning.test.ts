import assert from 'assert';
import { test } from 'node:test';
import {
  AUTO_ENGINE,
  buildTuningPlan,
  emptyOverrides,
  tuningToEngineArgs,
  type ModelShape,
  type TuningOverrides,
} from '../core/tuning/runtime_tuning.js';
import { discoverComputerProfile } from '../core/hardware/discovery.js';

/** An 80B-class MoE, shaped like the model these tests were written against. */
function moeModel(): ModelShape {
  return {
    architecture: 'qwen3next',
    layers: 48,
    contextLength: 262144,
    totalBytes: 48_958_621_568,
    isMoe: true,
    expertCount: 512,
    expertsUsedByDefault: 10,
    expertBytes: 46_808_432_640,
    denseBytes: 2_144_201_728,
    totalParamsB: 79.67,
  };
}

async function planWith(overrides: Partial<TuningOverrides>) {
  return buildTuningPlan({
    profile: await discoverComputerProfile(),
    model: moeModel(),
    overrides: { ...emptyOverrides(), ...overrides },
  });
}

test('an untouched plan leaves the engine its own defaults', async () => {
  const args = tuningToEngineArgs(await planWith({}));

  // Every one of these was imposed by an earlier version of the planner, and
  // each cost throughput: the engine decides them from live hardware, which no
  // arithmetic here can match. Their absence is the contract.
  for (const flag of ['--n-gpu-layers', '--n-cpu-moe', '--threads', '--batch-size', '--ubatch-size']) {
    assert.ok(!args.includes(flag), `${flag} must not be emitted when nothing was overridden`);
  }
});

test('an overridden value reaches the engine as a flag', async () => {
  const args = tuningToEngineArgs(await planWith({ expertsPerToken: 2, cpuMoeLayers: 48 }));

  assert.ok(args.includes('--override-kv'));
  assert.ok(args.includes('qwen3next.expert_used_count=int:2'));
  assert.deepStrictEqual(args.slice(args.indexOf('--n-cpu-moe'), args.indexOf('--n-cpu-moe') + 2), [
    '--n-cpu-moe',
    '48',
  ]);
});

test('AUTO_ENGINE stays out of the command line even when set explicitly', async () => {
  const args = tuningToEngineArgs(await planWith({ gpuLayers: AUTO_ENGINE, threads: AUTO_ENGINE }));

  assert.ok(!args.includes('--n-gpu-layers'));
  assert.ok(!args.includes('--threads'));
});

test('different overrides produce different arguments', async () => {
  // The backend decides whether to reuse a running engine by comparing the
  // arguments it was started with. If two distinct settings could produce the
  // same command line, a settings change would be accepted, reported as
  // applied, and silently ignored — which is exactly what happened once, and
  // made three rounds of benchmarking measure the configuration they had
  // replaced.
  const baseline = tuningToEngineArgs(await planWith({})).join(' ');

  for (const overrides of [
    { expertsPerToken: 2 },
    { cpuMoeLayers: 48 },
    { contextLength: 8192 },
    { kvCacheType: 'q8_0' as const },
    { expertsPerToken: 2, cpuMoeLayers: 48 },
  ]) {
    const changed = tuningToEngineArgs(await planWith(overrides)).join(' ');
    assert.notStrictEqual(changed, baseline, `${JSON.stringify(overrides)} left the command line unchanged`);
  }
});

test('a dense model is never told to place expert tensors', async () => {
  const dense: ModelShape = {
    ...moeModel(),
    architecture: 'qwen3',
    isMoe: false,
    expertCount: null,
    expertsUsedByDefault: null,
    expertBytes: 0,
    denseBytes: 48_958_621_568,
  };

  const plan = buildTuningPlan({
    profile: await discoverComputerProfile(),
    model: dense,
    overrides: emptyOverrides(),
  });

  assert.strictEqual(plan.expertsPerToken, null);
  assert.ok(!tuningToEngineArgs(plan).includes('--override-kv'));
});
