import os from 'os';
import path from 'path';
import { SFlowContainer } from '../../formats/sflow/sflow_format.js';
import { AiloStorageFabric } from '../../core/storage/storage_fabric.js';
import { HierarchicalCache } from '../../core/cache/hierarchical_cache.js';
import { PrefetchEngine } from '../../core/prefetch/prefetch_engine.js';
import { ComputerProfile } from '../../core/hardware/types.js';
import { calculateOptimalConfiguration } from '../../core/hardware/optimizer.js';

export interface LayerSweepProgress {
  layerIndex: number;
  totalLayers: number;
  bytesRead: number;
  elapsedMs: number;
  bandwidthMBps: number;
  cacheHitRatePercent: number;
}

export interface LayerSweepResult {
  /** Container that was streamed. */
  modelName: string;
  totalLayers: number;
  layersCompleted: number;
  /** Bytes actually pulled off disk (cache hits are excluded). */
  bytesReadFromStorage: number;
  totalBytesRequested: number;
  durationMs: number;
  /** Measured end-to-end throughput of the fabric during the sweep. */
  effectiveBandwidthMBps: number;
  /** Throughput seen by a single read request; below the aggregate when reads overlap. */
  averageRequestMBps: number;
  averageLayerLatencyMs: number;
  /**
   * true when the reads were likely served by the OS page cache rather than the
   * devices — always the case shortly after the shards were written. Comparing
   * drive counts under this condition measures RAM, not storage.
   */
  cacheInfluenced: boolean;
  /** Tokens per second the storage fabric alone could sustain at this rate. */
  sustainableTokensPerSecond: number;
  cacheHits: number;
  cacheMisses: number;
  cacheHitRatePercent: number;
  prefetchTriggered: number;
  prefetchThrottled: boolean;
  /** Lookahead actually used at the end of the sweep. */
  prefetchActiveDepth: number;
  drivesUsed: string[];
  prefetchDepth: number;
  /** Average weight bytes that must be streamed per decoded token. */
  bytesPerTokenStreamed: number;
  errors: string[];
}

/**
 * Run `worker` over every item with at most `limit` in flight at once.
 *
 * Concurrency is what turns a set of drives into a fabric: with one read
 * outstanding the aggregate throughput is whatever a single device manages,
 * regardless of how many are present.
 */
async function mapWithConcurrency<T>(
  items: T[],
  limit: number,
  worker: (item: T) => Promise<void>
): Promise<void> {
  if (items.length === 0) return;

  let cursor = 0;
  const runners = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
    while (cursor < items.length) {
      const item = items[cursor++];
      await worker(item);
    }
  });

  await Promise.all(runners);
}

/**
 * Executes the AILOFlow storage pipeline against a real `.sflow` container.
 *
 * This class does not generate text: producing tokens requires compute kernels,
 * which llama.cpp provides. What it does is real and measurable — it streams
 * every layer's tensors off the storage fabric through the hierarchical cache
 * with the predictive prefetcher running, and reports what that actually cost.
 * Those numbers are what the dashboard's storage figures are built from.
 */
export class AiloStreamingPipeline {
  public id = 'ailo-sflow-pipeline';
  public name = 'AILOFlow Storage Streaming Pipeline';

  private container: SFlowContainer | null = null;
  private fabric: AiloStorageFabric | null = null;
  private cache: HierarchicalCache | null = null;
  private prefetcher: PrefetchEngine | null = null;
  private profile: ComputerProfile;
  private prefetchDepth = 2;
  /** How many reads may be outstanding at once, from the optimizer's tuning. */
  private ioConcurrency = 16;

  constructor(profile: ComputerProfile) {
    this.profile = profile;
  }

  /** Build the full pipeline around a real .sflow container on disk. */
  public async load(sflowPath: string): Promise<void> {
    const container = SFlowContainer.load(sflowPath);
    const containerDir = path.dirname(sflowPath);

    const validation = container.validateShards(containerDir);
    if (!validation.valid) {
      throw new Error(
        `.sflow container is not backed by complete shard data:\n  - ${validation.problems.join('\n  - ')}`
      );
    }

    const config = calculateOptimalConfiguration(this.profile);
    this.prefetchDepth = config.prefetchDepthLayers;
    this.ioConcurrency = config.ioQueueDepth;

    this.container = container;
    this.fabric = new AiloStorageFabric(this.profile.storageDrives, containerDir);
    this.cache = new HierarchicalCache(config.vramCacheBytes, config.ramCacheBytes);
    this.prefetcher = new PrefetchEngine(this.cache, this.fabric, container.manifest, this.prefetchDepth);
  }

  public getFabric(): AiloStorageFabric | null { return this.fabric; }
  public getCache(): HierarchicalCache | null { return this.cache; }
  public getPrefetcher(): PrefetchEngine | null { return this.prefetcher; }
  public getContainer(): SFlowContainer | null { return this.container; }

  /**
   * Stream every layer in order, exactly as an inference pass would demand
   * them, and measure the result. Nothing here is simulated: each miss is a
   * real read from the storage fabric.
   */
  public async runLayerSweep(onProgress?: (p: LayerSweepProgress) => void): Promise<LayerSweepResult> {
    if (!this.container || !this.fabric || !this.cache || !this.prefetcher) {
      throw new Error('Pipeline not loaded. Call load(sflowPath) first.');
    }

    const manifest = this.container.manifest;
    const totalLayers = manifest.blockCount;
    const errors: string[] = [];

    this.fabric.resetStats();
    const startedAt = performance.now();
    let totalBytesRequested = 0;
    let layersCompleted = 0;
    const layerLatencies: number[] = [];

    for (let layer = 0; layer < totalLayers; layer++) {
      const layerStart = performance.now();

      // Ask the prefetcher to pull the upcoming layers while we consume this one.
      await this.prefetcher.onLayerStart(layer);

      const tensors = manifest.tensorMap.filter((t) => t.layerIndex === layer);
      for (const tensor of tensors) totalBytesRequested += tensor.sizeBytes;

      // Issue the layer's reads concurrently. Awaiting them one by one leaves
      // every drive but one idle, which caps the fabric at single-disk speed no
      // matter how many devices the shards are spread across — the opposite of
      // what distributing them was for.
      await mapWithConcurrency(tensors, this.ioConcurrency, async (tensor) => {
        if (this.cache!.get(tensor.name)) return;

        // The prefetcher may already be pulling this tensor. Waiting for that
        // read is the whole point of prefetching — issuing our own would
        // double the storage traffic.
        if (await this.prefetcher!.waitForInFlight(tensor.name)) {
          if (this.cache!.get(tensor.name)) return;
        }

        const shard = manifest.shards.find((s) => s.shardId === tensor.shardId);
        if (!shard) {
          errors.push(`Tensor ${tensor.name} references unknown shard ${tensor.shardId}`);
          return;
        }

        try {
          const data = await this.fabric!.readShardBlock(
            shard.targetDriveId,
            shard.relFilePath,
            tensor.offsetInShard,
            tensor.sizeBytes
          );
          this.cache!.put(tensor.name, tensor.layerIndex, data, 'L1_RAM');
        } catch (err) {
          errors.push(`${tensor.name}: ${(err as Error).message}`);
        }
      });

      const layerElapsed = performance.now() - layerStart;
      layerLatencies.push(layerElapsed);
      layersCompleted++;

      if (onProgress) {
        const io = this.fabric.getIoStats();
        const cacheMetrics = this.cache.getCacheMetrics();
        onProgress({
          layerIndex: layer,
          totalLayers,
          bytesRead: io.totalBytesRead,
          elapsedMs: Number((performance.now() - startedAt).toFixed(1)),
          bandwidthMBps: io.currentBandwidthMBps,
          cacheHitRatePercent: cacheMetrics.hitRatePercent,
        });
      }
    }

    const durationMs = performance.now() - startedAt;
    const io = this.fabric.getIoStats();
    const cacheMetrics = this.cache.getCacheMetrics();
    const prefetchStats = this.prefetcher.getStats();

    const drivesUsed = Object.entries(io.driveStats)
      .filter(([, s]) => s.bytesRead > 0)
      .map(([driveId]) => driveId);

    const effectiveBandwidthMBps = Number(
      (io.totalBytesRead / (1024 * 1024) / Math.max(0.001, durationMs / 1000)).toFixed(2)
    );

    return {
      modelName: manifest.modelName,
      totalLayers,
      layersCompleted,
      bytesReadFromStorage: io.totalBytesRead,
      totalBytesRequested,
      durationMs: Number(durationMs.toFixed(1)),
      effectiveBandwidthMBps,
      averageRequestMBps: io.averageRequestMBps,
      cacheInfluenced: totalBytesRequested < os.freemem(),
      // One token touches every layer, so this is the ceiling storage imposes.
      sustainableTokensPerSecond: Number(
        ((effectiveBandwidthMBps * 1024 * 1024) / Math.max(1, totalBytesRequested)).toFixed(3)
      ),
      averageLayerLatencyMs: Number(
        (layerLatencies.reduce((a, b) => a + b, 0) / Math.max(1, layerLatencies.length)).toFixed(2)
      ),
      cacheHits: cacheMetrics.hits,
      cacheMisses: cacheMetrics.misses,
      cacheHitRatePercent: cacheMetrics.hitRatePercent,
      prefetchTriggered: prefetchStats.prefetchesTriggered,
      prefetchThrottled: prefetchStats.throttledDueToMemoryPressure,
      prefetchActiveDepth: prefetchStats.activeDepth,
      drivesUsed,
      prefetchDepth: this.prefetchDepth,
      // One decoded token touches every layer once, so a full sweep is exactly
      // the weight traffic of a single token.
      bytesPerTokenStreamed: totalBytesRequested,
      errors,
    };
  }

  public dispose(): void {
    // Release the cached shard handles before dropping the fabric.
    void this.fabric?.dispose();
    this.container = null;
    this.fabric = null;
    this.cache = null;
    this.prefetcher = null;
  }
}
