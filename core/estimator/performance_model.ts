import { ComputerProfile } from '../hardware/types.js';
import { estimateWeightBytes } from '../models/catalog.js';

/**
 * Predicts decode speed for a model that does not fit in memory.
 *
 * The physics is simple and unforgiving: generating one token requires reading
 * every active weight exactly once. Decode is therefore bandwidth-bound, and
 * the achievable rate is
 *
 *     tokens/s  =  1 / Σ (bytes served by tier / bandwidth of tier)
 *
 * with the tiers being VRAM, RAM and storage. Everything below is arithmetic on
 * measured bandwidths; where a bandwidth could not be measured the estimate is
 * flagged rather than quietly filled in.
 */

export interface ModelSpec {
  name: string;
  totalParamsB: number;
  /** Parameters touched per token. Equals total for dense models. */
  activeParamsB: number;
  quantization: string;
  layers: number;
  contextLength: number;
  architecture: 'dense' | 'moe';
}

export interface TierBandwidth {
  tier: 'VRAM' | 'RAM' | 'STORAGE';
  capacityBytes: number;
  bandwidthMBps: number | null;
  /** How the bandwidth was obtained; drives how much the estimate is trusted. */
  source: 'measured' | 'spec-sheet' | 'estimated' | 'unknown';
  note: string;
}

export interface TierPlacement {
  tier: 'VRAM' | 'RAM' | 'STORAGE';
  /** Total weight bytes resident in this tier. */
  residentBytes: number;
  /** Of the per-token active bytes, how many come from here. */
  bytesPerToken: number;
  bandwidthMBps: number | null;
  secondsPerToken: number | null;
}

export interface PerformanceEstimate {
  model: ModelSpec;
  totalWeightBytes: number;
  activeBytesPerToken: number;
  kvCacheBytes: number;
  fitsInMemory: boolean;

  tiers: TierBandwidth[];
  placement: TierPlacement[];

  /** Tiers transfer one after another: the pessimistic bound. */
  sequentialTokensPerSecond: number | null;
  /**
   * Tiers transfer concurrently thanks to prefetching: the optimistic bound,
   * and what a well-tuned pipeline approaches.
   */
  overlappedTokensPerSecond: number | null;

  bottleneckTier: 'VRAM' | 'RAM' | 'STORAGE' | null;
  /** Storage bandwidth that would be needed to hit a target rate. */
  bandwidthForTenTokensPerSecondMBps: number | null;

  confidence: 'measured' | 'partial' | 'speculative';
  assumptions: string[];
  warnings: string[];
}

/**
 * Peak memory bandwidth by GPU family, in MB/s.
 *
 * These are manufacturer figures, not measurements: AILOFlow has no portable
 * way to benchmark VRAM bandwidth from Node. Any estimate relying on them is
 * marked `spec-sheet` so it is never mistaken for a measured result.
 */
const GPU_BANDWIDTH_SPEC: Array<{ pattern: RegExp; mbps: number; label: string }> = [
  { pattern: /rtx\s*50[89]0/i, mbps: 1_792_000, label: 'RTX 5080/5090 (GDDR7)' },
  { pattern: /rtx\s*5060\s*ti/i, mbps: 448_000, label: 'RTX 5060 Ti (GDDR7 128-bit)' },
  { pattern: /rtx\s*40[89]0/i, mbps: 1_008_000, label: 'RTX 4080/4090' },
  { pattern: /rtx\s*30[89]0/i, mbps: 936_000, label: 'RTX 3080/3090' },
  { pattern: /rx\s*7900/i, mbps: 960_000, label: 'RX 7900 XT/XTX' },
  { pattern: /rx\s*6[789]\d0/i, mbps: 432_000, label: 'RX 6700-6950 (GDDR6 192/256-bit)' },
  { pattern: /rx\s*7[678]\d0/i, mbps: 624_000, label: 'RX 7600-7800' },
  { pattern: /arc\s*[ab]\d+/i, mbps: 456_000, label: 'Intel Arc' },
  { pattern: /apple|m[1-4]\s*(pro|max|ultra)?/i, mbps: 200_000, label: 'Apple unified memory' },
  { pattern: /uhd graphics|iris/i, mbps: 0, label: 'iGPU: condivide la banda della RAM di sistema' },
];

function lookupGpuBandwidth(model: string): { mbps: number; label: string } | null {
  for (const entry of GPU_BANDWIDTH_SPEC) {
    if (entry.pattern.test(model)) return { mbps: entry.mbps, label: entry.label };
  }
  return null;
}

export interface EstimatorInputs {
  profile: ComputerProfile;
  /** Measured RAM bandwidth in MB/s; null when never benchmarked. */
  ramBandwidthMBps: number | null;
  /**
   * Storage read bandwidth to assume, in MB/s. Callers should pass a
   * cache-free measurement; a page-cache figure produces a fantasy result.
   */
  storageBandwidthMBps: number | null;
  storageBandwidthIsCacheInfluenced?: boolean;
  /** Context length actually used, for KV cache sizing. */
  contextUsed?: number;
  /** Fraction of VRAM/RAM left to the OS and the KV cache. */
  memoryHeadroomFraction?: number;
}

export function estimatePerformance(model: ModelSpec, inputs: EstimatorInputs): PerformanceEstimate {
  const assumptions: string[] = [];
  const warnings: string[] = [];

  const headroom = inputs.memoryHeadroomFraction ?? 0.15;
  const totalWeightBytes = estimateWeightBytes(model.totalParamsB, model.quantization);
  const activeBytesPerToken = estimateWeightBytes(model.activeParamsB, model.quantization);

  // KV cache at f16, sized from layers and context. Head dimensions are not in
  // the catalogue, so a representative 128-dim head with 8 KV heads is used.
  const contextUsed = inputs.contextUsed ?? Math.min(8192, model.contextLength);
  const kvCacheBytes = 2 * model.layers * contextUsed * 8 * 128 * 2;
  assumptions.push(
    `KV cache stimata su ${contextUsed.toLocaleString()} token di contesto con 8 teste KV da 128 dimensioni (f16).`
  );

  // ---- Tier capacities and bandwidths -------------------------------------

  const primaryGpu = inputs.profile.gpus[0];
  const vramCapacity = primaryGpu?.vramTotalBytes
    ? Math.max(0, primaryGpu.vramTotalBytes * (1 - headroom) - kvCacheBytes)
    : 0;

  const gpuSpec = primaryGpu ? lookupGpuBandwidth(primaryGpu.model) : null;
  const vramTier: TierBandwidth = {
    tier: 'VRAM',
    capacityBytes: vramCapacity,
    bandwidthMBps: gpuSpec && gpuSpec.mbps > 0 ? gpuSpec.mbps : null,
    source: gpuSpec && gpuSpec.mbps > 0 ? 'spec-sheet' : 'unknown',
    note: primaryGpu
      ? gpuSpec
        ? `${primaryGpu.model} — banda di targa: ${gpuSpec.label}.`
        : `${primaryGpu.model}: banda VRAM non nota, la GPU è esclusa dal calcolo.`
      : 'Nessuna GPU rilevata.',
  };
  if (vramTier.source === 'spec-sheet') {
    assumptions.push('La banda VRAM è un dato di targa del produttore, non una misura.');
  }

  const ramCapacity = Math.max(0, inputs.profile.ram.totalBytes * (1 - headroom) - vramCapacity * 0);
  const ramTier: TierBandwidth = {
    tier: 'RAM',
    capacityBytes: ramCapacity,
    bandwidthMBps: inputs.ramBandwidthMBps,
    source: inputs.ramBandwidthMBps ? 'measured' : 'unknown',
    note: inputs.ramBandwidthMBps
      ? `Banda misurata con benchmark memcpy su questa macchina.`
      : 'Banda RAM mai misurata: esegui "ailoflow system membench".',
  };

  const storageCapacity = inputs.profile.storageDrives.reduce((sum, d) => sum + d.freeSizeBytes, 0);
  const storageTier: TierBandwidth = {
    tier: 'STORAGE',
    capacityBytes: storageCapacity,
    bandwidthMBps: inputs.storageBandwidthMBps,
    source: inputs.storageBandwidthMBps
      ? inputs.storageBandwidthIsCacheInfluenced
        ? 'estimated'
        : 'measured'
      : 'unknown',
    note: inputs.storageBandwidthMBps
      ? inputs.storageBandwidthIsCacheInfluenced
        ? 'ATTENZIONE: banda misurata con la page cache attiva, quindi ottimistica.'
        : 'Banda aggregata misurata sui dischi utilizzabili.'
      : 'Storage mai misurato: esegui "ailoflow storage benchmark".',
  };

  if (inputs.storageBandwidthIsCacheInfluenced) {
    warnings.push(
      'La banda dello storage deriva da letture servite dalla cache del sistema operativo: ' +
        'la stima che segue è un limite superiore, non una previsione.'
    );
  }

  const tiers = [vramTier, ramTier, storageTier];

  // ---- Placement: fill the fastest tier first ------------------------------

  let remaining = totalWeightBytes;
  const vramResident = vramTier.bandwidthMBps ? Math.min(remaining, vramTier.capacityBytes) : 0;
  remaining -= vramResident;
  const ramResident = Math.min(remaining, ramTier.capacityBytes);
  remaining -= ramResident;
  const storageResident = Math.max(0, remaining);

  const fitsInMemory = storageResident === 0;
  if (!fitsInMemory && storageResident > storageCapacity) {
    warnings.push(
      `Il modello richiede ${(totalWeightBytes / 1e9).toFixed(0)} GB ma restano solo ` +
        `${(storageCapacity / 1e9).toFixed(0)} GB liberi sui dischi: non ci sta.`
    );
  }

  // Active weights are assumed to be spread uniformly across the tiers in
  // proportion to what each holds. For a MoE this is the neutral assumption:
  // without expert-hit statistics we cannot claim the hot experts are cached.
  const residentShare = (bytes: number) => (totalWeightBytes > 0 ? bytes / totalWeightBytes : 0);
  assumptions.push(
    model.architecture === 'moe'
      ? 'Gli esperti attivi sono distribuiti fra i livelli in proporzione a quanto ciascuno contiene. ' +
          'Con un caching efficace degli esperti caldi il risultato reale può essere migliore.'
      : 'Modello denso: ogni token richiede tutti i pesi, senza alcuna scorciatoia.'
  );

  const buildPlacement = (tier: TierBandwidth, residentBytes: number): TierPlacement => {
    const bytesPerToken = activeBytesPerToken * residentShare(residentBytes);
    const secondsPerToken =
      tier.bandwidthMBps && tier.bandwidthMBps > 0
        ? bytesPerToken / (tier.bandwidthMBps * 1024 * 1024)
        : bytesPerToken > 0
          ? null
          : 0;
    return {
      tier: tier.tier,
      residentBytes,
      bytesPerToken,
      bandwidthMBps: tier.bandwidthMBps,
      secondsPerToken,
    };
  };

  const placement = [
    buildPlacement(vramTier, vramResident),
    buildPlacement(ramTier, ramResident),
    buildPlacement(storageTier, storageResident),
  ];

  // ---- Rates ---------------------------------------------------------------

  const times = placement.map((p) => p.secondsPerToken);
  const anyUnknown = times.some((t) => t === null);

  let sequentialTokensPerSecond: number | null = null;
  let overlappedTokensPerSecond: number | null = null;

  if (!anyUnknown) {
    const known = times.map((t) => t ?? 0);
    const total = known.reduce((sum, t) => sum + t, 0);
    const slowest = Math.max(...known);
    sequentialTokensPerSecond = total > 0 ? Number((1 / total).toFixed(3)) : null;
    overlappedTokensPerSecond = slowest > 0 ? Number((1 / slowest).toFixed(3)) : null;
  } else {
    warnings.push(
      'Almeno un livello di memoria non ha una banda nota, quindi non è possibile ' +
        'produrre una stima numerica completa.'
    );
  }

  const slowestPlacement = placement
    .filter((p) => p.secondsPerToken !== null && p.bytesPerToken > 0)
    .sort((a, b) => (b.secondsPerToken || 0) - (a.secondsPerToken || 0))[0];

  const storageBytesPerToken = placement[2].bytesPerToken;
  const bandwidthForTenTokensPerSecondMBps =
    storageBytesPerToken > 0 ? Number(((storageBytesPerToken * 10) / (1024 * 1024)).toFixed(0)) : null;

  const confidence: PerformanceEstimate['confidence'] = anyUnknown
    ? 'speculative'
    : tiers.some((t) => t.source === 'spec-sheet' || t.source === 'estimated')
      ? 'partial'
      : 'measured';

  return {
    model,
    totalWeightBytes,
    activeBytesPerToken,
    kvCacheBytes,
    fitsInMemory,
    tiers,
    placement,
    sequentialTokensPerSecond,
    overlappedTokensPerSecond,
    bottleneckTier: slowestPlacement ? slowestPlacement.tier : null,
    bandwidthForTenTokensPerSecondMBps,
    confidence,
    assumptions,
    warnings,
  };
}

/**
 * What the estimate would become on a better storage configuration: useful for
 * answering "how many NVMe drives do I need before this is usable?".
 */
export function projectWithStorageBandwidth(
  model: ModelSpec,
  inputs: EstimatorInputs,
  storageBandwidthMBps: number
): PerformanceEstimate {
  return estimatePerformance(model, {
    ...inputs,
    storageBandwidthMBps,
    storageBandwidthIsCacheInfluenced: false,
  });
}
