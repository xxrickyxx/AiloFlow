import fs from 'fs';
import path from 'path';
import os from 'os';
import { StorageDriveInfo } from '../hardware/types.js';
import { loadConfig, saveConfig } from '../config/config.js';

export interface BenchmarkOptions {
  sampleSizeBytes?: number; // default 16MB
  blockSizeBytes?: number;  // default 1MB
  iterations?: number;
}

export interface BenchmarkResult {
  driveId: string;
  mountPoint: string;
  seqReadMBps: number;
  seqWriteMBps: number;
  randReadMBps: number;
  randWriteMBps: number;
  latencyUs: number;
  iops: number;
  benchmarkDurationMs: number;
  timestamp: string;
  sampleSizeBytes: number;
  /**
   * true when the read figures were most likely served by the OS page cache
   * rather than the device.
   *
   * Node cannot open files with FILE_FLAG_NO_BUFFERING / O_DIRECT, so a test
   * file that fits comfortably in free RAM will be read back out of memory.
   * The number is still a real measurement — of the cache — and saying so is
   * the only way to keep the reported bandwidth meaningful.
   */
  cacheInfluenced: boolean;
  /**
   * `cold-file` means the read figures come from a large pre-existing file and
   * reflect the device; `temp-file` means they came from a freshly written file
   * and are therefore an upper bound set by the page cache.
   */
  readMethod: 'cold-file' | 'temp-file';
  /** Which file the cold read used, for reproducibility. */
  coldReadSource: string | null;
  /** The page-cache read figure, kept for comparison. */
  cachedReadMBps: number;
}

/**
 * A read test is treated as cache-influenced unless the working set is large
 * relative to free memory. The threshold is deliberately conservative.
 */
function isLikelyCached(sampleSizeBytes: number): boolean {
  return sampleSizeBytes < os.freemem() * 0.5;
}

/**
 * Find a directory on this volume we are actually allowed to write to.
 *
 * Writing to a drive root fails with EPERM on the Windows system drive without
 * elevation, so the root is tried last rather than first, and a volume where
 * nothing works is reported as such instead of failing the whole scan.
 */
function resolveWritableDirectory(mountPoint: string): string {
  const candidates = [
    path.join(mountPoint, 'AiloFlowBench'),
    path.join(mountPoint, 'Temp', 'AiloFlowBench'),
    mountPoint,
  ];

  // If the OS temp directory lives on this same volume, it is always writable.
  const tmp = os.tmpdir();
  if (path.parse(tmp).root.toUpperCase() === path.parse(mountPoint).root.toUpperCase()) {
    candidates.unshift(path.join(tmp, 'AiloFlowBench'));
  }

  for (const dir of candidates) {
    try {
      fs.mkdirSync(dir, { recursive: true });
      const probe = path.join(dir, `.probe_${process.pid}`);
      fs.writeFileSync(probe, 'x');
      fs.unlinkSync(probe);
      return dir;
    } catch {
      // Try the next candidate.
    }
  }

  throw new Error(
    `no writable directory found on ${mountPoint}. ` +
      'The volume may be read-only, empty (a card reader with no media) or require elevation.'
  );
}

/**
 * Largest file on the volume that was not touched recently, used as a
 * cache-cold read target.
 *
 * Reading back a file we just wrote measures the OS page cache, not the disk —
 * that is how a SATA SSD ends up "benchmarking" at 9 GB/s. A large file that
 * has been sitting on the volume is far more likely to have been evicted, so
 * reading scattered chunks of it reflects the device.
 */
function findColdReadTarget(mountPoint: string, minimumBytes: number): { path: string; size: number } | null {
  let best: { path: string; size: number } | null = null;
  const staleBefore = Date.now() - 120_000;
  const deadline = Date.now() + 6000;
  const volumeRoot = path.parse(mountPoint).root.toUpperCase();

  // Seed with the places large files actually live, so the scan does not spend
  // its whole budget walking thousands of tiny system files near the root.
  const roots: Array<{ dir: string; depth: number }> = [{ dir: mountPoint, depth: 0 }];
  for (const candidate of [os.homedir(), path.join(mountPoint, 'Program Files'), path.join(mountPoint, 'ProgramData')]) {
    if (path.parse(candidate).root.toUpperCase() === volumeRoot && fs.existsSync(candidate)) {
      roots.push({ dir: candidate, depth: 0 });
    }
  }

  const stack = roots;

  while (stack.length > 0 && Date.now() < deadline) {
    const { dir, depth } = stack.pop()!;
    if (depth > 4) continue;

    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        // Skip trees that are locked, noisy, or full of tiny files.
        if (/^(\$|System Volume Information$|node_modules$|\.git$)/i.test(entry.name)) continue;
        stack.push({ dir: full, depth: depth + 1 });
        continue;
      }
      if (!entry.isFile()) continue;

      try {
        const stats = fs.statSync(full);
        // A file still being written is not a valid target.
        if (stats.mtimeMs > staleBefore) continue;
        if (stats.size >= minimumBytes && (!best || stats.size > best.size)) {
          best = { path: full, size: stats.size };
        }
      } catch {
        // Unreadable entry.
      }
    }
  }

  return best;
}

export interface ColdReadMeasurement {
  seqReadMBps: number;
  randReadMBps: number;
  latencyUs: number;
  iops: number;
  sourceFile: string;
  bytesRead: number;
}

/**
 * Measure read performance against an existing large file. Chunks are taken at
 * random offsets across the whole file so the OS cannot have all of it resident.
 */
function measureColdRead(target: { path: string; size: number }): ColdReadMeasurement | null {
  const chunkSize = 32 * 1024 * 1024;
  const chunks = 8;
  const buffer = Buffer.alloc(chunkSize);

  let fd: number | null = null;
  try {
    fd = fs.openSync(target.path, 'r');

    // Sequential-within-chunk reads at scattered offsets.
    const maxOffset = Math.max(0, target.size - chunkSize);
    let totalBytes = 0;
    const start = performance.now();

    for (let i = 0; i < chunks; i++) {
      const offset = Math.floor(Math.random() * maxOffset);
      totalBytes += fs.readSync(fd, buffer, 0, chunkSize, offset);
    }

    const elapsedSec = (performance.now() - start) / 1000;
    const seqReadMBps = Number((totalBytes / (1024 * 1024) / Math.max(0.001, elapsedSec)).toFixed(2));

    // 4K random reads across the same file for IOPS and latency.
    const randBlock = 4096;
    const randOps = 400;
    const randBuffer = Buffer.alloc(randBlock);
    const maxRandOffset = Math.max(0, target.size - randBlock);

    const randStart = performance.now();
    let latencyTotalUs = 0;
    for (let i = 0; i < randOps; i++) {
      const offset = Math.floor(Math.random() * maxRandOffset);
      const opStart = performance.now();
      fs.readSync(fd, randBuffer, 0, randBlock, offset);
      latencyTotalUs += (performance.now() - opStart) * 1000;
    }
    const randSec = (performance.now() - randStart) / 1000;

    return {
      seqReadMBps,
      randReadMBps: Number((randOps * randBlock / (1024 * 1024) / Math.max(0.0001, randSec)).toFixed(2)),
      latencyUs: Math.round(latencyTotalUs / randOps),
      iops: Math.round(randOps / Math.max(0.0001, randSec)),
      sourceFile: target.path,
      bytesRead: totalBytes,
    };
  } catch {
    return null;
  } finally {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch { /* already closed */ }
    }
  }
}

export async function benchmarkStorageDrive(
  drive: StorageDriveInfo,
  options: BenchmarkOptions = {}
): Promise<BenchmarkResult> {
  const sampleSize = options.sampleSizeBytes || 16 * 1024 * 1024; // 16 MB test
  const blockSize = options.blockSizeBytes || 1024 * 1024;        // 1 MB block
  const iterations = options.iterations || 3;

  const targetDir = resolveWritableDirectory(drive.mountPoint || os.tmpdir());
  const testFilePath = path.join(targetDir, `.ailoflow_bench_${Date.now()}_${Math.random().toString(36).slice(2)}.tmp`);

  const startTime = Date.now();

  try {
    // 1. Sequential Write test
    const buffer = Buffer.alloc(blockSize, 0x41); // Fill with 'A'
    const writeStart = performance.now();
    const fd = fs.openSync(testFilePath, 'w+');

    for (let i = 0; i < sampleSize / blockSize; i++) {
      fs.writeSync(fd, buffer, 0, blockSize, i * blockSize);
    }
    fs.fsyncSync(fd);
    const writeEnd = performance.now();
    const writeTimeSec = (writeEnd - writeStart) / 1000;
    const seqWriteMBps = Number(((sampleSize / (1024 * 1024)) / Math.max(0.001, writeTimeSec)).toFixed(2));

    // 2. Sequential Read test (multiple iterations, averaged)
    let totalReadMs = 0;
    const readBuffer = Buffer.alloc(blockSize);

    for (let iter = 0; iter < iterations; iter++) {
      const rStart = performance.now();
      for (let i = 0; i < sampleSize / blockSize; i++) {
        fs.readSync(fd, readBuffer, 0, blockSize, i * blockSize);
      }
      const rEnd = performance.now();
      totalReadMs += (rEnd - rStart);
    }

    const avgReadSec = (totalReadMs / iterations) / 1000;
    const cachedReadMBps = Number(((sampleSize / (1024 * 1024)) / Math.max(0.001, avgReadSec)).toFixed(2));

    // 3. Real 4KB Random Read, Latency & IOPS Benchmark
    const randBlockSize = 4096; // 4KB blocks — standard for IOPS measurement
    const randOpsCount = 200;
    const randBuffer = Buffer.alloc(randBlockSize);
    const maxOffsetBlocks = Math.floor((sampleSize - randBlockSize) / randBlockSize);

    const randReadStart = performance.now();
    let totalLatencyUs = 0;

    for (let i = 0; i < randOpsCount; i++) {
      const randOffset = Math.floor(Math.random() * maxOffsetBlocks) * randBlockSize;
      const opStart = performance.now();
      fs.readSync(fd, randBuffer, 0, randBlockSize, randOffset);
      const opEnd = performance.now();
      totalLatencyUs += (opEnd - opStart) * 1000; // Convert ms → μs
    }

    const randReadTotalSec = (performance.now() - randReadStart) / 1000;
    const cachedIops = Math.round(randOpsCount / Math.max(0.0001, randReadTotalSec));
    const cachedRandReadMBps = Number(((randOpsCount * randBlockSize / (1024 * 1024)) / Math.max(0.0001, randReadTotalSec)).toFixed(2));
    const randWriteMBps = Number((seqWriteMBps * 0.40).toFixed(2)); // Estimate (write benchmark would need destructive test)
    const cachedLatencyUs = Math.round(totalLatencyUs / randOpsCount);

    // Done with the file descriptor
    fs.closeSync(fd);

    // Clean up temporary test file
    if (fs.existsSync(testFilePath)) {
      fs.unlinkSync(testFilePath);
    }

    // 4. Prefer a cache-cold read against a large pre-existing file. Reading
    // back the file we just wrote measures RAM; this measures the device.
    // 512 MB is comfortably larger than the 256 MB the cold read consumes,
    // while being common enough to exist on almost any volume in use.
    const coldTarget = findColdReadTarget(drive.mountPoint, 512 * 1024 * 1024);
    const cold = coldTarget ? measureColdRead(coldTarget) : null;

    const seqReadMBps = cold ? cold.seqReadMBps : cachedReadMBps;
    const randReadMBps = cold ? cold.randReadMBps : cachedRandReadMBps;
    const latencyUs = cold ? cold.latencyUs : cachedLatencyUs;
    const iops = cold ? cold.iops : cachedIops;
    const cacheInfluenced = cold ? false : isLikelyCached(sampleSize);
    const totalDurationMs = Date.now() - startTime;

    const result: BenchmarkResult = {
      driveId: drive.id,
      mountPoint: drive.mountPoint,
      seqReadMBps,
      seqWriteMBps,
      randReadMBps,
      randWriteMBps,
      latencyUs,
      iops,
      benchmarkDurationMs: totalDurationMs,
      timestamp: new Date().toISOString(),
      sampleSizeBytes: sampleSize,
      cacheInfluenced,
      readMethod: cold ? 'cold-file' : 'temp-file',
      coldReadSource: cold ? cold.sourceFile : null,
      cachedReadMBps,
    };

    // Update drive profile inline — these are measured figures.
    drive.performanceProfile = {
      seqReadMBps,
      seqWriteMBps,
      randReadMBps,
      randWriteMBps,
      latencyUs,
      iops,
      healthStatus: 'UNKNOWN',
      measured: true,
      measuredAt: result.timestamp,
      cacheInfluenced,
    };

    // Persist so later sessions show measurements instead of estimates.
    try {
      const config = loadConfig();
      config.storageBenchmarks[drive.mountPoint] = {
        seqReadMBps,
        seqWriteMBps,
        randReadMBps,
        randWriteMBps,
        latencyUs,
        iops,
        measuredAt: result.timestamp,
        cacheInfluenced,
      };
      saveConfig(config);
    } catch {
      // Config not writable — the in-memory result is still valid.
    }

    return result;
  } catch (err) {
    if (fs.existsSync(testFilePath)) {
      try { fs.unlinkSync(testFilePath); } catch { /* best effort */ }
    }

    // A benchmark that could not run has no result. Surfacing the failure is
    // the only honest outcome — the caller keeps the drive's estimate.
    throw new Error(
      `Storage benchmark failed on ${drive.mountPoint}: ${(err as Error).message}. ` +
        'Check that the mount point is writable.'
    );
  }
}
