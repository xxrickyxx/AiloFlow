import os from 'os';
import { loadConfig, updateConfig } from '../config/config.js';

export interface MemoryBenchmarkResult {
  /** Bytes moved per second, counting both the read and the write side. */
  bandwidthMBps: number;
  /** Bytes copied per second (half the traffic above). */
  copyMBps: number;
  bufferSizeBytes: number;
  iterations: number;
  durationMs: number;
  timestamp: string;
}

/**
 * Measure real system memory bandwidth with a large memcpy loop.
 *
 * `Buffer.copy` lowers to memcpy, so the loop moves `2 × size` bytes per
 * iteration — one read plus one write. Weight streaming is read-dominated, and
 * a read-only stream reaches roughly the same figure, which is why
 * `bandwidthMBps` counts total traffic rather than bytes copied.
 *
 * Buffers are sized well past any CPU cache so the result reflects DRAM rather
 * than L3.
 */
export function benchmarkMemoryBandwidth(options: { bufferSizeBytes?: number; iterations?: number } = {}): MemoryBenchmarkResult {
  const freeBytes = os.freemem();
  // 256 MB comfortably exceeds L3 on any desktop CPU; shrink only if RAM is tight.
  const requested = options.bufferSizeBytes ?? 256 * 1024 * 1024;
  const bufferSize = Math.max(16 * 1024 * 1024, Math.min(requested, Math.floor(freeBytes * 0.1)));
  const iterations = options.iterations ?? 8;

  const source = Buffer.alloc(bufferSize);
  // Fill with varying bytes so nothing can be optimised into a zero-page trick.
  for (let i = 0; i < bufferSize; i += 4096) source[i] = i & 0xff;
  const destination = Buffer.alloc(bufferSize);

  // One untimed pass to fault in both buffers and warm the TLB.
  source.copy(destination);

  const startedAt = performance.now();
  for (let i = 0; i < iterations; i++) {
    source.copy(destination);
  }
  const durationMs = performance.now() - startedAt;

  const copiedBytes = bufferSize * iterations;
  const seconds = durationMs / 1000;
  const copyMBps = copiedBytes / (1024 * 1024) / seconds;

  const result: MemoryBenchmarkResult = {
    bandwidthMBps: Number((copyMBps * 2).toFixed(1)),
    copyMBps: Number(copyMBps.toFixed(1)),
    bufferSizeBytes: bufferSize,
    iterations,
    durationMs: Number(durationMs.toFixed(1)),
    timestamp: new Date().toISOString(),
  };

  try {
    updateConfig({ ramBandwidthMBps: result.bandwidthMBps });
  } catch {
    // Config not writable; the caller still gets the measurement.
  }

  return result;
}

/** Previously measured RAM bandwidth, or null if never benchmarked. */
export function getStoredRamBandwidth(): number | null {
  return loadConfig().ramBandwidthMBps;
}
