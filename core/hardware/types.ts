export interface CpuInfo {
  vendor: string;
  model: string;
  arch: string;
  /** null when the OS does not report a physical core count. */
  physicalCores: number | null;
  logicalThreads: number;
  /** null when the reported clock is unavailable (common in VMs). */
  baseFrequencyGHz: number | null;
  instructions: {
    sse: boolean;
    sse2: boolean;
    avx: boolean;
    avx2: boolean;
    avx512: boolean;
    neon: boolean;
  };
  recommendedOptimization: 'AVX512' | 'AVX2' | 'AVX' | 'NEON' | 'GENERIC';
}

export interface RamInfo {
  totalBytes: number;
  freeBytes: number;
  usedBytes: number;
  systemLoadPercent: number;
  recommendedCacheBytes: number;
  /** Module details are null when the OS does not expose them. */
  speedMHz: number | null;
  moduleCount: number | null;
  memoryType: string | null;
}

export interface GpuInfo {
  id: string;
  vendor: 'NVIDIA' | 'AMD' | 'Intel' | 'Apple' | 'Unknown';
  model: string;
  /** null when the OS does not expose a trustworthy VRAM figure. */
  vramTotalBytes: number | null;
  vramFreeBytes: number | null;
  /** How the VRAM figures were obtained, so the UI can qualify them. */
  vramSource: 'nvidia-smi' | 'registry' | 'wmi' | 'sysfs' | 'unified-memory' | 'unknown';
  /** Windows PNP identifier, used to join live counters to this adapter. */
  pnpDeviceId?: string;
  driverVersion?: string;
  supportedBackends: ('CUDA' | 'Vulkan' | 'HIP' | 'Metal' | 'CPU')[];
  recommendedBackend: 'CUDA' | 'Vulkan' | 'HIP' | 'Metal' | 'CPU';
  temperatureC?: number | null;
  utilizationPercent?: number | null;
}

export interface StorageDriveInfo {
  id: string;
  devicePath: string;
  mountPoint: string;
  label: string;
  type: 'NVMe' | 'SATA' | 'HDD' | 'USB' | 'UNKNOWN';
  totalSizeBytes: number;
  freeSizeBytes: number;
  filesystem: string;
  performanceProfile?: StoragePerformanceProfile;
}

export interface StoragePerformanceProfile {
  seqReadMBps: number;
  seqWriteMBps: number;
  randReadMBps: number;
  randWriteMBps: number;
  latencyUs: number;
  iops: number;
  healthStatus: 'HEALTHY' | 'WARNING' | 'CRITICAL' | 'UNKNOWN';
  /**
   * true  → produced by an actual benchmark run on this machine.
   * false → a class-based estimate derived from the bus type only.
   * The UI must never present an estimate as a measurement.
   */
  measured: boolean;
  measuredAt?: string;
  /**
   * true when the read figures reflect the OS page cache rather than the
   * device. Such numbers must not be presented as device bandwidth.
   */
  cacheInfluenced?: boolean;
}

export interface ComputerProfile {
  timestamp: string;
  os: {
    platform: NodeJS.Platform;
    release: string;
    arch: string;
  };
  cpu: CpuInfo;
  ram: RamInfo;
  gpus: GpuInfo[];
  storageDrives: StorageDriveInfo[];
  selectedBackend: 'CUDA' | 'Vulkan' | 'HIP' | 'Metal' | 'CPU';
  totalStorageBytes: number;
  /** Sum of *benchmarked* read bandwidth; null while nothing is measured yet. */
  measuredStorageReadBandwidthMBps: number | null;
}

export interface OptimizerConfig {
  ramCacheBytes: number;
  vramCacheBytes: number;
  threadCount: number;
  backend: 'CUDA' | 'Vulkan' | 'HIP' | 'Metal' | 'CPU';
  usableDrives: StorageDriveInfo[];
  prefetchDepthLayers: number;
  ioQueueDepth: number;
  shardingStrategy: 'Layer' | 'TensorBlock' | 'RowColumn';
  ioChunkSizeBytes: number;
  /** false when the tuning is derived from bus-type estimates, not a benchmark. */
  basedOnMeasuredStorage: boolean;
}
