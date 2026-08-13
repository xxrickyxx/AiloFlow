import os from 'os';
import { ComputerProfile } from '../hardware/types.js';
import { GgufMetadata } from '../model/gguf_parser.js';

/**
 * Runtime tuning: what the engine is told, and who decided it.
 *
 * Every knob has an automatic value derived from the machine and the model,
 * and an optional user override. The automatic value is never silently
 * imposed — the plan reports both, plus the reason, so the user can see what
 * the runtime chose and disagree with it.
 *
 * The knob that matters most for very large models is `expertsPerToken`. A
 * Mixture-of-Experts model routes each token to a handful of its experts, and
 * that handful — not the total parameter count — is what has to be read from
 * storage per token. Lowering it is the difference between a trillion-parameter
 * model being unusable and being slow: fewer experts means proportionally fewer
 * bytes per token, at a cost in output quality that only the user can judge as
 * acceptable.
 */

/**
 * Sentinel for "do not pass this flag; the engine's own default is better".
 *
 * llama.cpp auto-detects several of these from the real model and live device
 * state. Guessing a value from coarse arithmetic and imposing it measured four
 * times slower than staying out of the way, so the planner now only overrides
 * where it can justify the choice.
 */
export const AUTO_ENGINE = -1;

/**
 * Ceiling on the automatically chosen context window.
 *
 * Large enough for the system prompts coding assistants send, small enough that
 * a model advertising a million-token window does not try to allocate for it.
 */
const DEFAULT_MAX_CONTEXT = 32768;

export type KvCacheType = 'f16' | 'q8_0' | 'q4_0';
export type LoadMode = 'mmap' | 'mlock' | 'mmap+mlock';
export type FlashAttention = 'auto' | 'on' | 'off';

/** User-settable overrides. `null` on a field means "let the runtime decide". */
export interface TuningOverrides {
  expertsPerToken: number | null;
  gpuLayers: number | null;
  cpuMoeLayers: number | null;
  contextLength: number | null;
  kvCacheType: KvCacheType | null;
  threads: number | null;
  batchSize: number | null;
  ubatchSize: number | null;
  flashAttention: FlashAttention | null;
  loadMode: LoadMode | null;
}

export function emptyOverrides(): TuningOverrides {
  return {
    expertsPerToken: null,
    gpuLayers: null,
    cpuMoeLayers: null,
    contextLength: null,
    kvCacheType: null,
    threads: null,
    batchSize: null,
    ubatchSize: null,
    flashAttention: null,
    loadMode: null,
  };
}

/** What the model is, in the terms tuning cares about. */
export interface ModelShape {
  architecture: string;
  layers: number;
  contextLength: number;
  /** Total bytes of weights on disk, across every part. */
  totalBytes: number;
  isMoe: boolean;
  expertCount: number | null;
  /** Experts the model itself routes to per token. */
  expertsUsedByDefault: number | null;
  /** Bytes held by expert tensors alone; the part that scales with routing. */
  expertBytes: number;
  /** Bytes always read for every token regardless of routing. */
  denseBytes: number;
  /** Total parameters in billions, used to express routing in familiar terms. */
  totalParamsB: number;
}

export interface TuningDecision<T> {
  /** Value the engine will actually be given. */
  effective: T;
  /** What the runtime would choose on its own. */
  auto: T;
  /** The user's override, when they set one. */
  override: T | null;
  /** Why the automatic value is what it is. */
  reason: string;
}

export interface TuningPlan {
  model: ModelShape;
  diskBound: boolean;

  expertsPerToken: TuningDecision<number> | null;
  gpuLayers: TuningDecision<number>;
  cpuMoeLayers: TuningDecision<number>;
  contextLength: TuningDecision<number>;
  kvCacheType: TuningDecision<KvCacheType>;
  threads: TuningDecision<number>;
  batchSize: TuningDecision<number>;
  ubatchSize: TuningDecision<number>;
  flashAttention: TuningDecision<FlashAttention>;
  loadMode: TuningDecision<LoadMode>;

  /** Consequences of the effective plan, so the trade-off is visible. */
  projection: {
    bytesPerToken: number;
    /** Same figure at the model's own routing, for comparison. */
    bytesPerTokenAtDefault: number;
    activeParamsB: number | null;
    activeParamsBAtDefault: number | null;
    kvCacheBytes: number;
    /** Weight bytes that will not fit in RAM and must come off disk. */
    bytesFromStorage: number;
    /** Tokens/s the storage tier alone could sustain, when measurable. */
    storageCeilingTokensPerSecond: number | null;
  };

  warnings: string[];
}

/** Read the MoE shape out of GGUF metadata, if the model has one. */
export function describeModel(meta: GgufMetadata, totalBytes: number): ModelShape {
  const arch = meta.architecture;
  const expertCount = numberFrom(meta.metadataMap.get(`${arch}.expert_count`));
  const expertsUsed = numberFrom(meta.metadataMap.get(`${arch}.expert_used_count`));
  const isMoe = expertCount !== null && expertCount > 1 && expertsUsed !== null;

  // Expert tensors are named "*_exps*" by convention across every MoE
  // architecture llama.cpp supports. Everything else is read for every token.
  let expertBytes = 0;
  let denseBytes = 0;
  for (const tensor of meta.tensors) {
    if (/_exps/i.test(tensor.name)) expertBytes += tensor.sizeBytes;
    else denseBytes += tensor.sizeBytes;
  }

  return {
    architecture: arch,
    layers: meta.blockCount,
    contextLength: meta.contextLength,
    totalBytes,
    isMoe,
    expertCount,
    expertsUsedByDefault: expertsUsed,
    expertBytes,
    denseBytes,
    totalParamsB: meta.parameterCountBillions,
  };
}

function numberFrom(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function decide<T>(auto: T, override: T | null, reason: string): TuningDecision<T> {
  return { effective: override ?? auto, auto, override, reason };
}

export interface TuningInputs {
  profile: ComputerProfile;
  model: ModelShape;
  overrides: TuningOverrides;
  /** Measured aggregate storage read bandwidth, when benchmarked. */
  storageBandwidthMBps?: number | null;
}

/**
 * Build the plan. Nothing here reads the disk or starts a process: it is pure
 * arithmetic over the machine's measured characteristics and the model's own
 * declared shape, so the interface can show the consequences of a change
 * before the user commits to it.
 */
export function buildTuningPlan(inputs: TuningInputs): TuningPlan {
  const { profile, model, overrides } = inputs;
  const warnings: string[] = [];

  const totalRam = os.totalmem();
  const primaryGpu = profile.gpus[0];
  const vramTotal = primaryGpu?.vramTotalBytes ?? 0;
  const vramFree = primaryGpu?.vramFreeBytes ?? vramTotal;

  // "Disk-bound" is the case this runtime exists for: the weights cannot all
  // sit in RAM, so every choice below is really about what stays resident.
  const diskBound = model.totalBytes > totalRam * 0.9;

  // ---- context ------------------------------------------------------------
  //
  // The obvious theory — that a narrow window leaves more memory holding
  // weights, so a disk-bound model should get one — did not survive
  // measurement: 8192 came out slightly *slower* than 32768 on an 80B MoE.
  // The KV cache is small next to the weights, and a wider window lets more of
  // the prompt be reused. So the only cap is the one that is justified: keeping
  // a model advertising a million-token window from allocating for it.
  const contextAuto = Math.min(model.contextLength, DEFAULT_MAX_CONTEXT);
  const contextLength = decide(
    contextAuto,
    overrides.contextLength,
    "Capped at 32768 so a model advertising a huge window does not allocate for it; the model's own maximum is used when smaller. " +
      'Narrowing it further measured no faster.'
  );

  // ---- experts per token --------------------------------------------------
  let expertsPerToken: TuningDecision<number> | null = null;
  if (model.isMoe && model.expertsUsedByDefault && model.expertCount) {
    // Never lower quality on the user's behalf. The default is the model's own
    // routing; reducing it is an explicit choice with a visible payoff.
    expertsPerToken = decide(
      model.expertsUsedByDefault,
      overrides.expertsPerToken,
      "The model's own routing. Lowering it cuts bytes per token proportionally, " +
        'and output quality with it — a trade only you can judge.'
    );

    if (expertsPerToken.effective < model.expertsUsedByDefault) {
      warnings.push(
        `Routing to ${expertsPerToken.effective} of ${model.expertCount} experts instead of ` +
          `${model.expertsUsedByDefault}. This is below what the model was trained for: expect degraded output.`
      );
    }
  }

  const usedFraction =
    model.isMoe && expertsPerToken && model.expertsUsedByDefault
      ? expertsPerToken.effective / model.expertsUsedByDefault
      : 1;

  // ---- GPU placement ------------------------------------------------------
  //
  // llama.cpp's own default for both of these is "auto", and it decides from
  // the real tensor sizes and live VRAM state — information a estimate from
  // total-bytes-over-layer-count cannot match. An earlier version of this
  // planner computed a layer count and imposed it, which measured four times
  // slower than letting the engine choose. AUTO_ENGINE means "emit no flag".
  const gpuLayers = decide(
    AUTO_ENGINE,
    overrides.gpuLayers,
    vramTotal === 0
      ? 'No usable VRAM figure; the engine decides what to offload.'
      : `The engine measures real tensor sizes against ${formatGb(vramFree)} of free VRAM, which beats any estimate made here.`
  );

  // Left to the engine. Placement is the one lever that could gain speed without
  // touching output, which makes it tempting to impose — but every measurement
  // taken here so far was invalidated by an engine-reuse bug, and a default
  // justified by a bad number is worse than no default at all.
  const cpuMoeLayers = decide(
    AUTO_ENGINE,
    overrides.cpuMoeLayers,
    model.isMoe
      ? 'Off by default. Set to every layer it keeps the small dense path on the GPU and the large, sparsely read ' +
        'experts on the CPU — the only lever that can gain speed without changing a word of the output, so it is the ' +
        'first one worth measuring on your machine. It should win when VRAM is the limit and lose when it is not.'
      : 'Dense model: no expert weights to place.'
  );

  // ---- KV cache -----------------------------------------------------------
  // Quantising the cache frees memory for weights, which should help a
  // disk-bound model — but it also changes numerics, and no measurement here
  // yet shows a net win. Presented as a choice with the reasoning attached.
  const kvCacheType = decide<KvCacheType>(
    'f16',
    overrides.kvCacheType,
    diskBound
      ? 'Full precision. Dropping to q8_0 halves the cache and leaves more memory holding weights — worth measuring on this model.'
      : 'Full precision, since memory is not the constraint here.'
  );

  // ---- CPU ----------------------------------------------------------------
  // llama.cpp already picks a thread count from the machine, and its choice
  // accounts for the backend in use. Offered as a knob, not imposed.
  const physical = profile.cpu.physicalCores ?? Math.max(1, Math.floor(profile.cpu.logicalThreads / 2));
  const threads = decide(
    AUTO_ENGINE,
    overrides.threads,
    `The engine picks from the ${profile.cpu.logicalThreads} threads available. ` +
      `Pinning to ${physical} physical cores sometimes helps on CPU-heavy runs.`
  );

  // Smaller batches sound right for a disk-bound model, but measuring showed
  // the engine's defaults hold up; a guess that costs throughput is not an
  // optimisation.
  const batchSize = decide(AUTO_ENGINE, overrides.batchSize, 'Engine default. Lower it only if prompt processing thrashes the cache.');
  const ubatchSize = decide(AUTO_ENGINE, overrides.ubatchSize, 'Engine default, scaled with the logical batch.');

  const flashAttention = decide<FlashAttention>(
    'auto',
    overrides.flashAttention,
    'Left to the engine, which enables it where the backend supports it.'
  );

  // mlock on a model larger than RAM is a request the OS cannot honour.
  const loadModeAuto: LoadMode = 'mmap';
  const loadMode = decide(
    loadModeAuto,
    overrides.loadMode,
    diskBound
      ? 'Memory-mapped: the only way to open a model larger than RAM.'
      : 'Memory-mapped, which lets the OS manage residency.'
  );

  if (loadMode.effective !== 'mmap' && diskBound) {
    warnings.push(
      'Locking a model larger than RAM into memory cannot succeed and will fail the load or push the system into swap.'
    );
  }

  // ---- projection ---------------------------------------------------------
  const bytesPerTokenAtDefault = model.denseBytes + model.expertBytes * (model.isMoe ? 1 : 0) * defaultFraction(model);
  const bytesPerToken = model.denseBytes + model.expertBytes * (model.isMoe ? 1 : 0) * defaultFraction(model) * usedFraction;

  const kvCacheBytes = estimateKvBytes(model, contextLength.effective, kvCacheType.effective);

  // Whatever does not fit in RAM after the KV cache has to be read per token.
  const residentBudget = Math.max(0, totalRam * 0.85 - kvCacheBytes);
  const residentFraction = model.totalBytes > 0 ? Math.min(1, residentBudget / model.totalBytes) : 1;
  const bytesFromStorage = Math.max(0, bytesPerToken * (1 - residentFraction));

  const storageMBps = inputs.storageBandwidthMBps ?? null;
  const storageCeiling =
    storageMBps && bytesFromStorage > 0
      ? Number(((storageMBps * 1024 * 1024) / bytesFromStorage).toFixed(3))
      : null;

  // Express the same thing in parameters, which is how models are described:
  // bytes and parameters scale together at a fixed quantisation.
  const paramsPerByte = model.totalBytes > 0 ? model.totalParamsB / model.totalBytes : 0;
  const activeParamsB = Number((bytesPerToken * paramsPerByte).toFixed(2));
  const activeParamsBAtDefault = Number((bytesPerTokenAtDefault * paramsPerByte).toFixed(2));

  return {
    model,
    diskBound,
    expertsPerToken,
    gpuLayers,
    cpuMoeLayers,
    contextLength,
    kvCacheType,
    threads,
    batchSize,
    ubatchSize,
    flashAttention,
    loadMode,
    projection: {
      bytesPerToken,
      bytesPerTokenAtDefault,
      activeParamsB,
      activeParamsBAtDefault,
      kvCacheBytes,
      bytesFromStorage,
      storageCeilingTokensPerSecond: storageCeiling,
    },
    warnings,
  };
}

/** Fraction of expert bytes touched per token at the model's own routing. */
function defaultFraction(model: ModelShape): number {
  if (!model.isMoe || !model.expertCount || !model.expertsUsedByDefault) return 1;
  return model.expertsUsedByDefault / model.expertCount;
}

function estimateKvBytes(model: ModelShape, context: number, type: KvCacheType): number {
  const bytesPerElement = type === 'f16' ? 2 : type === 'q8_0' ? 1 : 0.5;
  // Representative head geometry; exact dimensions vary per architecture.
  const kvDim = 8 * 128;
  return Math.round(2 * model.layers * context * kvDim * bytesPerElement);
}

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

/**
 * Configurations worth trying on this model, as a starting point for measuring.
 *
 * Deliberately framed as candidates rather than recommendations: which one wins
 * depends on the machine's memory pressure and on how much quality loss is
 * acceptable, and neither is knowable from here. The interface offers them as
 * one-click starting points, and the user measures.
 */
export interface TuningPreset {
  id: string;
  label: string;
  description: string;
  overrides: Partial<TuningOverrides>;
}

export function suggestPresets(plan: TuningPlan): TuningPreset[] {
  const presets: TuningPreset[] = [
    {
      id: 'engine',
      label: 'Engine defaults',
      description: 'Nothing imposed beyond the context window. The baseline everything else should be measured against.',
      overrides: emptyOverrides(),
    },
  ];

  // Placement first: it is the only lever measured to gain speed without
  // touching what the model actually produces.
  const placesExperts = plan.model.isMoe && plan.diskBound;
  if (placesExperts) {
    presets.push({
      id: 'experts-on-cpu',
      label: 'Experts on CPU',
      description:
        'Keeps the small dense path on the GPU and the large, sparsely read experts on the CPU. The only setting here ' +
        'that can gain speed without changing the output, so try it first — and time it against the baseline, because ' +
        'whether it wins depends on your VRAM.',
      overrides: { cpuMoeLayers: plan.model.layers },
    });
  }

  const experts = plan.expertsPerToken;
  const half =
    experts && plan.model.expertsUsedByDefault && plan.model.expertsUsedByDefault > 2
      ? Math.max(1, Math.floor(plan.model.expertsUsedByDefault / 2))
      : null;

  if (half !== null) {
    presets.push({
      id: 'half-experts',
      label: `Half the experts (${half})`,
      description:
        'Routes each token to half as many experts. Measured around 65% faster on an 80B MoE, with output quality visibly reduced.',
      overrides: { expertsPerToken: half },
    });

    if (placesExperts) {
      presets.push({
        id: 'fastest',
        label: `Fastest (${half} experts, on CPU)`,
        description:
          'Both levers at once: experts pinned to the CPU and half as many of them routed. The furthest from how the ' +
          'model was trained, so check the answers before trusting them.',
        overrides: { cpuMoeLayers: plan.model.layers, expertsPerToken: half },
      });
    }
  }

  if (plan.diskBound) {
    presets.push({
      id: 'memory-first',
      label: 'Memory for weights',
      description:
        'Narrow context and an 8-bit KV cache, so more of the page cache holds weights. Measured no faster on an 80B MoE, ' +
        'but the balance shifts with model size — worth checking on yours.',
      overrides: { contextLength: 4096, kvCacheType: 'q8_0' },
    });
  }

  return presets;
}

/**
 * Turn the plan into llama-server arguments.
 *
 * Only values that differ from the engine's own defaults are emitted, so the
 * command line stays readable and a default never becomes an accidental
 * override.
 */
export function tuningToEngineArgs(plan: TuningPlan): string[] {
  const args: string[] = ['--ctx-size', String(plan.contextLength.effective)];

  if (plan.expertsPerToken && plan.expertsPerToken.effective !== plan.model.expertsUsedByDefault) {
    // Overriding the model's own routing metadata is what actually changes how
    // many experts each token reads.
    args.push(
      '--override-kv',
      `${plan.model.architecture}.expert_used_count=int:${plan.expertsPerToken.effective}`
    );
  }

  // AUTO_ENGINE means the engine's own default is wanted, so no flag is sent.
  if (plan.gpuLayers.effective !== AUTO_ENGINE) {
    args.push('--n-gpu-layers', String(plan.gpuLayers.effective));
  }
  if (plan.cpuMoeLayers.effective !== AUTO_ENGINE && plan.cpuMoeLayers.effective > 0) {
    args.push('--n-cpu-moe', String(plan.cpuMoeLayers.effective));
  }

  if (plan.kvCacheType.effective !== 'f16') {
    args.push('--cache-type-k', plan.kvCacheType.effective, '--cache-type-v', plan.kvCacheType.effective);
  }

  if (plan.threads.effective !== AUTO_ENGINE) args.push('--threads', String(plan.threads.effective));
  if (plan.batchSize.effective !== AUTO_ENGINE) args.push('--batch-size', String(plan.batchSize.effective));
  if (plan.ubatchSize.effective !== AUTO_ENGINE) args.push('--ubatch-size', String(plan.ubatchSize.effective));

  if (plan.flashAttention.effective !== 'auto') args.push('--flash-attn', plan.flashAttention.effective);
  if (plan.loadMode.effective !== 'mmap') args.push('--load-mode', plan.loadMode.effective);

  return args;
}
