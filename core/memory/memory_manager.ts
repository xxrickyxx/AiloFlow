import { HierarchicalCache, TensorTemperature } from '../cache/hierarchical_cache.js';

export interface MemoryPoolState {
  vramAllocatedMB: number;
  ramAllocatedMB: number;
  ssdCachedMB: number;
  hotTensorCount: number;
  warmTensorCount: number;
  coldTensorCount: number;
}

export class MemoryManager {
  private cache: HierarchicalCache;

  constructor(cache: HierarchicalCache) {
    this.cache = cache;
  }

  public classifyTensor(tensorName: string, recentAccesses: number): TensorTemperature {
    if (recentAccesses > 10) return 'HOT';
    if (recentAccesses > 2) return 'WARM';
    return 'COLD';
  }

  public getPoolState(): MemoryPoolState {
    const metrics = this.cache.getCacheMetrics();
    return {
      vramAllocatedMB: Number((metrics.vramUsedBytes / (1024 * 1024)).toFixed(1)),
      ramAllocatedMB: Number((metrics.ramUsedBytes / (1024 * 1024)).toFixed(1)),
      ssdCachedMB: Number(((metrics.l2Count * 4 * 1024 * 1024) / (1024 * 1024)).toFixed(1)),
      hotTensorCount: metrics.l0Count,
      warmTensorCount: metrics.l1Count,
      coldTensorCount: metrics.l2Count,
    };
  }
}
