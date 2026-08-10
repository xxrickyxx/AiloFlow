import { HierarchicalCache } from '../cache/hierarchical_cache.js';
import { AiloStorageFabric } from '../storage/storage_fabric.js';
import { SFlowManifest } from '../../formats/sflow/sflow_format.js';

export interface PrefetchStats {
  prefetchesTriggered: number;
  prefetchHits: number;
  prefetchMisses: number;
  /** null until at least one prefetch has resolved — never a placeholder. */
  prefetchHitRatePercent: number | null;
  /** Lookahead currently in use; drops under memory pressure but never to zero. */
  activeDepth: number;
  /** Lookahead the optimizer asked for, for comparison. */
  configuredDepth: number;
  /** true when pressure has reduced the lookahead below the configured depth. */
  throttledDueToMemoryPressure: boolean;
}

export class PrefetchEngine {
  private cache: HierarchicalCache;
  private fabric: AiloStorageFabric;
  private manifest: SFlowManifest;
  private prefetchDepth: number;

  private prefetchesTriggered = 0;
  private prefetchHits = 0;
  private prefetchMisses = 0;
  /** In-flight prefetches, so a consumer can await one instead of re-reading. */
  private activeJobs: Map<string, Promise<void>> = new Map();
  private isThrottled = false;
  /** Lookahead actually in use, after any reduction for memory pressure. */
  private currentDepth: number;

  constructor(
    cache: HierarchicalCache,
    fabric: AiloStorageFabric,
    manifest: SFlowManifest,
    prefetchDepth = 2
  ) {
    this.cache = cache;
    this.fabric = fabric;
    this.manifest = manifest;
    this.prefetchDepth = prefetchDepth;
    this.currentDepth = prefetchDepth;
  }

  /**
   * Depth to use right now, given how full the RAM cache is.
   *
   * Memory pressure is the normal state for a model larger than RAM, not an
   * emergency: the cache is *supposed* to be full, with the LRU recycling the
   * oldest layers. Suspending prefetch there would disable it for exactly the
   * workload it exists to serve, leaving every read to be paid for on the
   * critical path. So pressure narrows the lookahead instead of stopping it —
   * one layer ahead is still enough to overlap I/O with compute.
   */
  private depthForPressure(ramPressurePercent: number): number {
    if (ramPressurePercent >= 98) return 1;
    if (ramPressurePercent > 92) return Math.max(1, Math.floor(this.prefetchDepth / 3));
    if (ramPressurePercent > 85) return Math.max(1, Math.floor(this.prefetchDepth / 2));
    return this.prefetchDepth;
  }

  public async onLayerStart(currentLayerIndex: number): Promise<void> {
    const totalLayers = this.manifest.blockCount;

    // Check RAM pressure from cache metrics
    const metrics = this.cache.getCacheMetrics();
    const ramPressurePercent =
      metrics.ramLimitBytes > 0 ? (metrics.ramUsedBytes / metrics.ramLimitBytes) * 100 : 0;

    const effectiveDepth = this.depthForPressure(ramPressurePercent);
    this.isThrottled = effectiveDepth < this.prefetchDepth;
    this.currentDepth = effectiveDepth;

    // Prefetch Layer current + 1, current + 2 up to the effective depth
    for (let depth = 1; depth <= effectiveDepth; depth++) {
      const targetLayer = currentLayerIndex + depth;
      if (targetLayer >= totalLayers) break;

      const layerTensors = this.manifest.tensorMap.filter(t => t.layerIndex === targetLayer);

      for (const tensor of layerTensors) {
        if (this.activeJobs.has(tensor.name)) continue;
        if (this.cache.get(tensor.name)) {
          this.prefetchHits++;
          continue;
        }
        // Fire and forget: the point of prefetching is to overlap this I/O
        // with the current layer's compute.
        void this.prefetchTensorAsync(tensor);
      }
    }
  }

  /**
   * Resolve once any in-flight prefetch for `tensorName` has finished.
   *
   * Consumers must call this before falling back to their own read, otherwise
   * the same tensor gets pulled off disk twice — the prefetcher would then cost
   * bandwidth instead of saving it.
   */
  public async waitForInFlight(tensorName: string): Promise<boolean> {
    const job = this.activeJobs.get(tensorName);
    if (!job) return false;
    await job;
    return true;
  }

  private async prefetchTensorAsync(tensor: SFlowManifest['tensorMap'][0]): Promise<void> {
    this.prefetchesTriggered++;

    const job = this.runPrefetch(tensor);
    this.activeJobs.set(tensor.name, job);
    try {
      await job;
    } finally {
      this.activeJobs.delete(tensor.name);
    }
  }

  private async runPrefetch(tensor: SFlowManifest['tensorMap'][0]): Promise<void> {
    try {
      const shard = this.manifest.shards.find(s => s.shardId === tensor.shardId);
      const driveId = shard ? shard.targetDriveId : 'drive-0';
      const relPath = shard ? shard.relFilePath : `shards/${tensor.shardId}.bin`;

      // Read block from StorageFabric asynchronously
      const data = await this.fabric.readShardBlock(driveId, relPath, tensor.offsetInShard, tensor.sizeBytes);

      // Store in L1 RAM cache ready for GPU upload
      this.cache.put(tensor.name, tensor.layerIndex, data, 'L1_RAM');
    } catch {
      // The tensor stays uncached; the consumer will read it itself.
      this.prefetchMisses++;
    }
  }

  public getStats(): PrefetchStats {
    const total = this.prefetchHits + this.prefetchMisses;
    const hitRate = total > 0 ? Number(((this.prefetchHits / total) * 100).toFixed(1)) : null;

    return {
      prefetchesTriggered: this.prefetchesTriggered,
      prefetchHits: this.prefetchHits,
      prefetchMisses: this.prefetchMisses,
      prefetchHitRatePercent: hitRate,
      activeDepth: this.currentDepth,
      configuredDepth: this.prefetchDepth,
      throttledDueToMemoryPressure: this.isThrottled,
    };
  }
}
