import { ComputerProfile, OptimizerConfig, StorageDriveInfo } from './types.js';

export function calculateOptimalConfiguration(profile: ComputerProfile): OptimizerConfig {
  const ramFree = profile.ram.freeBytes;
  const ramTotal = profile.ram.totalBytes;

  // Allocate 60% of free RAM (max 75% of total RAM) to cache to prevent OS swapping
  const ramCacheBytes = Math.min(Math.floor(ramFree * 0.65), Math.floor(ramTotal * 0.75));

  const primaryGpu = profile.gpus[0];
  // Unknown VRAM means no VRAM budget can be claimed — we do not guess one.
  const vramFree = primaryGpu?.vramFreeBytes ?? 0;
  // Reserve 1.5GB VRAM for OS/display context, allocate rest for L0 model layer staging
  const vramCacheBytes = Math.max(0, vramFree - 1.5 * 1024 * 1024 * 1024);

  // Backend selection priority
  const backend = profile.selectedBackend || (primaryGpu ? primaryGpu.recommendedBackend : 'CPU');

  // Filter usable drives (sorted by sequential read performance)
  const usableDrives = [...profile.storageDrives].sort((a, b) => {
    const speedA = a.performanceProfile?.seqReadMBps || 0;
    const speedB = b.performanceProfile?.seqReadMBps || 0;
    return speedB - speedA;
  });

  const topDrive = usableDrives[0]?.performanceProfile;
  const topDriveSpeed = topDrive?.seqReadMBps || 0;
  // Tuning derived from an unmeasured estimate is itself an estimate.
  const basedOnMeasuredStorage = topDrive?.measured === true;

  // Compute adaptive prefetch depth based on disk bandwidth and VRAM budget
  let prefetchDepthLayers = 2;
  if (topDriveSpeed > 6000) {
    prefetchDepthLayers = 4;
  } else if (topDriveSpeed > 3000) {
    prefetchDepthLayers = 3;
  } else if (topDriveSpeed < 1000) {
    prefetchDepthLayers = 1;
  }

  // CPU threads allocation (leave 1-2 cores free for I/O loop and OS)
  const threadCount = Math.max(1, profile.cpu.logicalThreads - 2);

  // I/O queue depth scale with drive performance
  const ioQueueDepth = topDriveSpeed > 5000 ? 64 : topDriveSpeed > 2000 ? 32 : 16;

  // Sharding strategy recommendation
  let shardingStrategy: OptimizerConfig['shardingStrategy'] = 'Layer';
  if (usableDrives.length > 1) {
    shardingStrategy = 'TensorBlock';
  }

  return {
    ramCacheBytes,
    vramCacheBytes,
    threadCount,
    backend,
    usableDrives,
    prefetchDepthLayers,
    ioQueueDepth,
    shardingStrategy,
    ioChunkSizeBytes: 4 * 1024 * 1024, // 4MB default I/O chunk
    basedOnMeasuredStorage,
  };
}
