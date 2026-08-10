export type CacheTier = 'L0_VRAM' | 'L1_RAM' | 'L2_SSD' | 'L3_STORAGE';
export type TensorTemperature = 'HOT' | 'WARM' | 'COLD';

export interface CacheEntry {
  tensorName: string;
  layerIndex: number;
  tier: CacheTier;
  temperature: TensorTemperature;
  data: Buffer;
  sizeBytes: number;
  lastAccessTimestamp: number;
  accessCount: number;
}

export class HierarchicalCache {
  private vramLimitBytes: number;
  private ramLimitBytes: number;
  
  private l0Vram: Map<string, CacheEntry> = new Map();
  private l1Ram: Map<string, CacheEntry> = new Map();
  private l2Ssd: Map<string, CacheEntry> = new Map();

  private vramUsedBytes = 0;
  private ramUsedBytes = 0;

  private hits = 0;
  private misses = 0;

  constructor(vramLimitBytes: number, ramLimitBytes: number) {
    this.vramLimitBytes = vramLimitBytes;
    this.ramLimitBytes = ramLimitBytes;
  }

  public get(tensorName: string): CacheEntry | undefined {
    const now = Date.now();

    // Check L0 (VRAM)
    if (this.l0Vram.has(tensorName)) {
      const entry = this.l0Vram.get(tensorName)!;
      entry.lastAccessTimestamp = now;
      entry.accessCount++;
      entry.temperature = 'HOT';
      this.hits++;
      return entry;
    }

    // Check L1 (RAM)
    if (this.l1Ram.has(tensorName)) {
      const entry = this.l1Ram.get(tensorName)!;
      entry.lastAccessTimestamp = now;
      entry.accessCount++;
      entry.temperature = 'HOT';
      this.hits++;

      // Promote to L0 VRAM if budget allows
      this.promoteToVram(entry);
      return entry;
    }

    // Check L2 (SSD Cache)
    if (this.l2Ssd.has(tensorName)) {
      const entry = this.l2Ssd.get(tensorName)!;
      entry.lastAccessTimestamp = now;
      entry.accessCount++;
      entry.temperature = 'WARM';
      this.hits++;

      // Promote to L1 RAM
      this.promoteToRam(entry);
      return entry;
    }

    this.misses++;
    return undefined;
  }

  public put(tensorName: string, layerIndex: number, data: Buffer, targetTier: CacheTier = 'L1_RAM'): void {
    const entry: CacheEntry = {
      tensorName,
      layerIndex,
      tier: targetTier,
      temperature: targetTier === 'L0_VRAM' ? 'HOT' : targetTier === 'L1_RAM' ? 'WARM' : 'COLD',
      data,
      sizeBytes: data.length,
      lastAccessTimestamp: Date.now(),
      accessCount: 1,
    };

    if (targetTier === 'L0_VRAM') {
      this.promoteToVram(entry);
    } else {
      this.promoteToRam(entry);
    }
  }

  private promoteToVram(entry: CacheEntry): void {
    // Evict cold tensors from VRAM if size limit exceeded
    while (this.vramUsedBytes + entry.sizeBytes > this.vramLimitBytes && this.l0Vram.size > 0) {
      this.evictFromVram();
    }

    entry.tier = 'L0_VRAM';
    entry.temperature = 'HOT';
    this.l0Vram.set(entry.tensorName, entry);
    this.vramUsedBytes += entry.sizeBytes;
  }

  private promoteToRam(entry: CacheEntry): void {
    // Evict LRU entries from RAM if budget exceeded
    while (this.ramUsedBytes + entry.sizeBytes > this.ramLimitBytes && this.l1Ram.size > 0) {
      this.evictFromRam();
    }

    entry.tier = 'L1_RAM';
    entry.temperature = 'WARM';
    this.l1Ram.set(entry.tensorName, entry);
    this.ramUsedBytes += entry.sizeBytes;
  }

  private evictFromVram(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.l0Vram.entries()) {
      if (entry.lastAccessTimestamp < oldestTime) {
        oldestTime = entry.lastAccessTimestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      const entry = this.l0Vram.get(oldestKey)!;
      this.l0Vram.delete(oldestKey);
      this.vramUsedBytes -= entry.sizeBytes;
      // Demote to RAM
      this.promoteToRam(entry);
    }
  }

  private evictFromRam(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, entry] of this.l1Ram.entries()) {
      if (entry.lastAccessTimestamp < oldestTime) {
        oldestTime = entry.lastAccessTimestamp;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      const entry = this.l1Ram.get(oldestKey)!;
      this.l1Ram.delete(oldestKey);
      this.ramUsedBytes -= entry.sizeBytes;
      entry.tier = 'L2_SSD';
      entry.temperature = 'COLD';
      this.l2Ssd.set(oldestKey, entry);
    }
  }

  public getCacheMetrics() {
    const totalRequests = this.hits + this.misses;
    // With no traffic there is no hit rate to report, not a perfect one.
    const hitRatePercent = totalRequests > 0 ? Number(((this.hits / totalRequests) * 100).toFixed(1)) : 0;

    return {
      vramUsedBytes: this.vramUsedBytes,
      vramLimitBytes: this.vramLimitBytes,
      ramUsedBytes: this.ramUsedBytes,
      ramLimitBytes: this.ramLimitBytes,
      l0Count: this.l0Vram.size,
      l1Count: this.l1Ram.size,
      l2Count: this.l2Ssd.size,
      hits: this.hits,
      misses: this.misses,
      hitRatePercent,
    };
  }
}
