import os from 'os';
import { ComputerProfile } from '../hardware/types.js';
import { GpuMonitor, GpuLiveSample } from '../hardware/gpu_monitor.js';
import { StorageIoStats } from '../storage/storage_fabric.js';
import { PrefetchStats } from '../prefetch/prefetch_engine.js';
import { GenerationMetrics } from '../../inference/base.js';

export type BottleneckType =
  | 'STORAGE'
  | 'VRAM'
  | 'RAM'
  | 'CPU'
  | 'GPU_COMPUTE'
  | 'CACHE'
  | 'NONE'
  | 'IDLE'
  | 'UNKNOWN';

/**
 * A telemetry snapshot. Any field typed `| null` means "not measurable on this
 * machine right now" and must be rendered as such — never substituted with a
 * placeholder number.
 */
export interface TelemetrySnapshot {
  timestamp: string;

  // Live host metrics
  cpuUsagePercent: number;
  ramUsagePercent: number;
  ramUsedBytes: number;
  ramTotalBytes: number;
  gpuUsagePercent: number | null;
  vramUsagePercent: number | null;
  vramUsedBytes: number | null;
  vramTotalBytes: number | null;
  gpuTemperatureC: number | null;
  /** Per-adapter detail, so multi-GPU systems are not collapsed into one bar. */
  gpus: GpuLiveSample[];
  /** Populated when no GPU telemetry source exists on this machine. */
  gpuUnavailableReason: string | null;

  // Inference metrics from the last real generation
  generation: {
    active: boolean;
    modelId: string | null;
    backendId: string | null;
    tokensPerSecond: number | null;
    promptTokensPerSecond: number | null;
    firstTokenLatencyMs: number | null;
    completionTokens: number | null;
    promptTokens: number | null;
    measuredAt: string | null;
  };

  // Storage pipeline metrics (only non-null once the pipeline has run)
  storage: {
    bandwidthMBps: number | null;
    iops: number | null;
    totalBytesRead: number;
    queueDepth: number;
  };

  cache: {
    hitRatePercent: number | null;
    hits: number;
    misses: number;
    vramUsedBytes: number;
    ramUsedBytes: number;
  } | null;

  prefetch: {
    hitRatePercent: number | null;
    triggered: number;
    throttled: boolean;
  } | null;

  bottleneck: {
    type: BottleneckType;
    title: string;
    description: string;
    requestedBandwidthMBps?: number;
    availableBandwidthMBps?: number;
  };
}

// ---------------------------------------------------------------------------
// Real CPU utilisation via /proc-equivalent tick deltas from os.cpus()
// ---------------------------------------------------------------------------

interface CpuTicks {
  idle: number;
  total: number;
}

function readCpuTicks(): CpuTicks {
  let idle = 0;
  let total = 0;
  for (const cpu of os.cpus()) {
    idle += cpu.times.idle;
    total += cpu.times.user + cpu.times.nice + cpu.times.sys + cpu.times.idle + cpu.times.irq;
  }
  return { idle, total };
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

export interface TelemetrySources {
  ioStats?: StorageIoStats;
  prefetchStats?: PrefetchStats;
  cacheMetrics?: {
    hitRatePercent: number;
    hits: number;
    misses: number;
    vramUsedBytes: number;
    ramUsedBytes: number;
  };
  lastGeneration?: GenerationMetrics | null;
  lastGenerationAt?: string | null;
  generationActive?: boolean;
  /** Bytes of model weights that must be streamed per token, when known. */
  bytesPerTokenStreamed?: number | null;
}

export class TelemetryMonitor {
  private lastTicks: CpuTicks = readCpuTicks();
  private cpuUsagePercent = 0;
  private samplerHandle: NodeJS.Timeout | null = null;
  private gpuMonitor = new GpuMonitor();

  /** Teach the GPU monitor which adapters exist, so counters can be attributed. */
  public configureGpus(profile: ComputerProfile): void {
    this.gpuMonitor.setAdapters(
      profile.gpus.map((gpu) => ({ key: gpu.id, name: gpu.model, vramTotalBytes: gpu.vramTotalBytes }))
    );
  }

  /**
   * Fill in free VRAM from the live counters.
   *
   * Only nvidia-smi reports free memory directly; for every other vendor the
   * figure is total minus the measured usage. Doing it here means the optimizer
   * can budget VRAM on AMD and Intel too, instead of treating them as unknown.
   */
  public enrichProfileWithLiveGpu(profile: ComputerProfile): void {
    const state = this.gpuMonitor.getState();
    if (state.samples.length === 0) return;

    for (const gpu of profile.gpus) {
      if (gpu.vramFreeBytes !== null || gpu.vramTotalBytes === null) continue;

      const own = state.samples.find((s) => s.adapterKey === gpu.id);
      const shared = profile.gpus.length === 1 ? state.samples.find((s) => s.adapterKey === null) : undefined;
      const used = (own || shared)?.memoryUsedBytes;
      if (used === undefined || used === null) continue;

      gpu.vramFreeBytes = Math.max(0, gpu.vramTotalBytes - used);
    }
  }

  /** Begin periodic background sampling. Safe to call more than once. */
  public start(intervalMs = 1000): void {
    this.gpuMonitor.start();
    if (this.samplerHandle) return;
    this.samplerHandle = setInterval(() => { void this.sample(); }, intervalMs);
    this.samplerHandle.unref?.();
  }

  public stop(): void {
    if (this.samplerHandle) {
      clearInterval(this.samplerHandle);
      this.samplerHandle = null;
    }
    this.gpuMonitor.stop();
  }

  /** Take one measurement pass. Exposed so callers can sample on demand. */
  public async sample(): Promise<void> {
    const ticks = readCpuTicks();
    const idleDelta = ticks.idle - this.lastTicks.idle;
    const totalDelta = ticks.total - this.lastTicks.total;
    if (totalDelta > 0) {
      this.cpuUsagePercent = Number((100 * (1 - idleDelta / totalDelta)).toFixed(1));
    }
    this.lastTicks = ticks;

    await this.gpuMonitor.sample();
  }

  public generateSnapshot(profile: ComputerProfile, sources: TelemetrySources = {}): TelemetrySnapshot {
    const ramTotalBytes = os.totalmem();
    const ramFreeBytes = os.freemem();
    const ramUsedBytes = ramTotalBytes - ramFreeBytes;
    const ramUsagePercent = Number(((ramUsedBytes / ramTotalBytes) * 100).toFixed(1));

    const gpuState = this.gpuMonitor.getState();
    const gpu = gpuState.aggregate;
    const io = sources.ioStats;
    const storageBandwidthMBps = io ? io.currentBandwidthMBps : null;
    const cacheMetrics = sources.cacheMetrics;
    const prefetchStats = sources.prefetchStats;
    const gen = sources.lastGeneration || null;

    const effectiveVramUsedBytes =
      cacheMetrics && cacheMetrics.vramUsedBytes > 0
        ? cacheMetrics.vramUsedBytes
        : gpu.memoryUsedBytes;

    const effectiveVramTotalBytes =
      gpu.memoryTotalBytes || (profile.gpus?.[0]?.vramTotalBytes ?? null);

    const vramUsagePercent =
      effectiveVramUsedBytes !== null && effectiveVramTotalBytes
        ? Number(((effectiveVramUsedBytes / effectiveVramTotalBytes) * 100).toFixed(1))
        : null;

    const bottleneck = this.detectBottleneck({
      cpuUsagePercent: this.cpuUsagePercent,
      ramUsagePercent,
      vramUsagePercent,
      gpuUsagePercent: gpu.utilizationPercent,
      storageBandwidthMBps,
      cacheHitRatePercent: cacheMetrics ? cacheMetrics.hitRatePercent : null,
      generation: gen,
      generationActive: sources.generationActive === true,
      bytesPerTokenStreamed: sources.bytesPerTokenStreamed ?? null,
      availableStorageBandwidthMBps: measuredStorageBandwidth(profile),
    });

    return {
      timestamp: new Date().toISOString(),
      cpuUsagePercent: this.cpuUsagePercent,
      ramUsagePercent,
      ramUsedBytes,
      ramTotalBytes,
      gpuUsagePercent: gpu.utilizationPercent,
      vramUsagePercent,
      vramUsedBytes: effectiveVramUsedBytes,
      vramTotalBytes: effectiveVramTotalBytes,
      gpuTemperatureC: gpu.temperatureC,
      gpus: gpuState.samples,
      gpuUnavailableReason: gpuState.unavailableReason,

      generation: {
        active: sources.generationActive === true,
        modelId: gen ? gen.modelId : null,
        backendId: gen ? gen.backendId : null,
        tokensPerSecond: gen ? gen.tokensPerSecond : null,
        promptTokensPerSecond: gen ? gen.promptTokensPerSecond : null,
        firstTokenLatencyMs: gen ? gen.firstTokenLatencyMs : null,
        completionTokens: gen ? gen.completionTokens : null,
        promptTokens: gen ? gen.promptTokens : null,
        measuredAt: sources.lastGenerationAt || null,
      },

      storage: {
        bandwidthMBps: storageBandwidthMBps,
        iops: io ? io.iops : null,
        totalBytesRead: io ? io.totalBytesRead : 0,
        queueDepth: io ? io.activeQueueDepth : 0,
      },

      cache: cacheMetrics
        ? {
            hitRatePercent: cacheMetrics.hits + cacheMetrics.misses > 0 ? cacheMetrics.hitRatePercent : null,
            hits: cacheMetrics.hits,
            misses: cacheMetrics.misses,
            vramUsedBytes: cacheMetrics.vramUsedBytes,
            ramUsedBytes: cacheMetrics.ramUsedBytes,
          }
        : null,

      prefetch: prefetchStats
        ? {
            hitRatePercent: prefetchStats.prefetchHitRatePercent,
            triggered: prefetchStats.prefetchesTriggered,
            throttled: prefetchStats.throttledDueToMemoryPressure,
          }
        : null,

      bottleneck,
    };
  }

  /**
   * Bottleneck detection from measured values only. When nothing has run yet
   * there is no bottleneck to report, and we say exactly that.
   */
  private detectBottleneck(input: {
    cpuUsagePercent: number;
    ramUsagePercent: number;
    vramUsagePercent: number | null;
    gpuUsagePercent: number | null;
    storageBandwidthMBps: number | null;
    cacheHitRatePercent: number | null;
    generation: GenerationMetrics | null;
    generationActive: boolean;
    bytesPerTokenStreamed: number | null;
    availableStorageBandwidthMBps: number | null;
  }): TelemetrySnapshot['bottleneck'] {
    if (!input.generation && !input.generationActive) {
      return {
        type: 'IDLE',
        title: 'NESSUNA INFERENZA IN CORSO',
        description:
          'Il collo di bottiglia viene calcolato su dati misurati durante una generazione. ' +
          'Avvia una chat o un benchmark per ottenere una diagnosi.',
      };
    }

    // Storage is only the bottleneck when we can compare a real demand against
    // a real measured bandwidth.
    const tps = input.generation?.tokensPerSecond ?? null;
    if (input.bytesPerTokenStreamed && tps && input.availableStorageBandwidthMBps) {
      const requiredMBps = (input.bytesPerTokenStreamed * tps) / (1024 * 1024);
      if (requiredMBps > input.availableStorageBandwidthMBps * 0.9) {
        return {
          type: 'STORAGE',
          title: 'COLLO DI BOTTIGLIA: STORAGE',
          description:
            `A ${tps} token/s il pipeline richiede ${(requiredMBps / 1024).toFixed(2)} GB/s di lettura, ` +
            `contro ${(input.availableStorageBandwidthMBps / 1024).toFixed(2)} GB/s misurati sui dischi in uso.`,
          requestedBandwidthMBps: Number(requiredMBps.toFixed(1)),
          availableBandwidthMBps: Number(input.availableStorageBandwidthMBps.toFixed(1)),
        };
      }
    }

    if (input.vramUsagePercent !== null && input.vramUsagePercent > 95) {
      return {
        type: 'VRAM',
        title: 'COLLO DI BOTTIGLIA: VRAM',
        description: `VRAM al ${input.vramUsagePercent}%: ulteriori layer vengono spinti su RAM o storage.`,
      };
    }

    if (input.ramUsagePercent > 95) {
      return {
        type: 'RAM',
        title: 'COLLO DI BOTTIGLIA: RAM',
        description: `RAM di sistema al ${input.ramUsagePercent}%: lo swap su disco riduce il throughput.`,
      };
    }

    if (input.cacheHitRatePercent !== null && input.cacheHitRatePercent < 70) {
      return {
        type: 'CACHE',
        title: 'COLLO DI BOTTIGLIA: CACHE',
        description: `Hit rate della cache al ${input.cacheHitRatePercent}%: troppi miss, aumentare RAM cache o prefetch.`,
      };
    }

    if (input.gpuUsagePercent !== null && input.gpuUsagePercent > 90) {
      return {
        type: 'GPU_COMPUTE',
        title: 'COLLO DI BOTTIGLIA: GPU COMPUTE',
        description: `GPU al ${input.gpuUsagePercent}%: il limite è la potenza di calcolo, non l'I/O.`,
      };
    }

    if (input.cpuUsagePercent > 90) {
      return {
        type: 'CPU',
        title: 'COLLO DI BOTTIGLIA: CPU',
        description: `CPU al ${input.cpuUsagePercent}%: l'inferenza sta girando prevalentemente su CPU.`,
      };
    }

    if (input.gpuUsagePercent === null && input.storageBandwidthMBps === null) {
      return {
        type: 'UNKNOWN',
        title: 'DIAGNOSI NON DISPONIBILE',
        description:
          'Nessun contatore GPU né pipeline di storage attiva: non ci sono dati misurati sufficienti ' +
          'per attribuire un collo di bottiglia.',
      };
    }

    return {
      type: 'NONE',
      title: 'NESSUN COLLO DI BOTTIGLIA CRITICO',
      description: 'Tutte le risorse misurate sono sotto le soglie di saturazione.',
    };
  }
}

/** Sum of *measured* sequential read bandwidth; null when nothing is benchmarked. */
export function measuredStorageBandwidth(profile: ComputerProfile): number | null {
  const measured = profile.storageDrives.filter((d) => d.performanceProfile?.measured);
  if (measured.length === 0) return null;
  return measured.reduce((sum, d) => sum + (d.performanceProfile?.seqReadMBps || 0), 0);
}
