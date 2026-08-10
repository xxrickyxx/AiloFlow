import os from 'os';

/**
 * Widen libuv's threadpool before any file I/O happens.
 *
 * Node performs `fs` operations on a threadpool that defaults to **4** threads.
 * Every asynchronous read, no matter how many are queued, waits for one of
 * those four slots — so a storage fabric spread over eight devices still reads
 * with four in flight and tops out at roughly four times a single request.
 * Measured on a five-SSD fabric that ceiling was ~450 MB/s while the drives
 * together can do several times that.
 *
 * The variable is read by libuv when the pool is first used, so this module
 * must be imported before anything that touches the filesystem. Both entry
 * points import it first for that reason.
 */
export function configureThreadPool(): number {
  // Enough slots to keep every device busy with several outstanding requests,
  // without spawning threads the machine cannot schedule.
  const desired = Math.min(128, Math.max(16, os.cpus().length * 2));

  if (!process.env.UV_THREADPOOL_SIZE) {
    process.env.UV_THREADPOOL_SIZE = String(desired);
    return desired;
  }

  return Number(process.env.UV_THREADPOOL_SIZE) || desired;
}

export const THREAD_POOL_SIZE = configureThreadPool();
