import fs from 'fs';
import path from 'path';
import { StorageDriveInfo } from '../hardware/types.js';

export interface StorageIoStats {
  totalBytesRead: number;
  totalReads: number;
  activeQueueDepth: number;
  /**
   * Wall-clock throughput over the recent window, idle time included. This is
   * the pipeline's real delivery rate and correctly falls to 0 when nothing
   * is being read.
   */
  currentBandwidthMBps: number;
  /**
   * Bytes divided by the summed duration of every read in the window.
   *
   * This is the throughput an *individual* request sees, not the aggregate:
   * when reads run concurrently their durations overlap, so this figure stays
   * below `currentBandwidthMBps` by roughly the degree of parallelism. Useful
   * for spotting per-request latency problems, misleading as a device rating.
   */
  averageRequestMBps: number;
  /** Read operations per second over the same window. */
  iops: number;
  driveStats: Record<string, { bytesRead: number; reads: number; readMBps: number }>;
}

interface ReadEvent {
  at: number;
  bytes: number;
  driveId: string;
  durationMs: number;
}

const WINDOW_MS = 2000;

/**
 * Logical fabric over every storage device holding model shards.
 *
 * All reads are real filesystem reads. A missing shard file is an error the
 * caller must see — there is no synthetic-data path that would make a broken
 * container look like a working one.
 */
export class AiloStorageFabric {
  private drives: Map<string, StorageDriveInfo> = new Map();
  private driveStats: Map<string, { bytesRead: number; reads: number }> = new Map();
  private events: ReadEvent[] = [];
  private totalBytesRead = 0;
  private totalReads = 0;
  private activeQueueDepth = 0;
  private baseDirectory: string;
  /** Open shard files, keyed by absolute path. */
  private handles: Map<string, Promise<fs.promises.FileHandle>> = new Map();

  constructor(drives: StorageDriveInfo[], baseDirectory = process.cwd()) {
    this.baseDirectory = baseDirectory;
    for (const drive of drives) {
      this.drives.set(drive.id, drive);
      this.driveStats.set(drive.id, { bytesRead: 0, reads: 0 });
    }
  }

  public getDriveCount(): number {
    return this.drives.size;
  }

  public getDrives(): StorageDriveInfo[] {
    return Array.from(this.drives.values());
  }

  /**
   * Read `length` bytes at `offset` from a shard file. `filePath` may be
   * relative to the container directory.
   */
  public async readShardBlock(
    driveId: string,
    filePath: string,
    offset: number,
    length: number
  ): Promise<Buffer> {
    const absolute = path.isAbsolute(filePath) ? filePath : path.join(this.baseDirectory, filePath);

    this.activeQueueDepth++;
    const startTime = performance.now();

    try {
      const handle = await this.openShard(absolute);
      const buf = Buffer.alloc(length);
      const { bytesRead } = await handle.read(buf, 0, length, offset);

      if (bytesRead < length) {
        throw new Error(
          `Short read on ${absolute}: wanted ${length} bytes at offset ${offset}, got ${bytesRead}. ` +
            'The shard file is truncated or the manifest is out of date.'
        );
      }

      const now = performance.now();
      this.totalBytesRead += bytesRead;
      this.totalReads++;
      this.events.push({ at: now, bytes: bytesRead, driveId, durationMs: now - startTime });
      this.trimWindow(now);

      const ds = this.driveStats.get(driveId) || { bytesRead: 0, reads: 0 };
      ds.bytesRead += bytesRead;
      ds.reads++;
      this.driveStats.set(driveId, ds);

      return buf;
    } finally {
      this.activeQueueDepth = Math.max(0, this.activeQueueDepth - 1);
    }
  }

  /**
   * Open a shard once and keep the handle.
   *
   * A layer sweep issues one read per tensor — hundreds of thousands for a
   * large model — and opening plus closing the same handful of shard files
   * every time costs more than the reads themselves. Handles are released by
   * `dispose()`.
   */
  private async openShard(absolutePath: string): Promise<fs.promises.FileHandle> {
    const cached = this.handles.get(absolutePath);
    if (cached) return cached;

    // Store the promise, not the handle: concurrent readers of the same shard
    // must share one open() rather than racing to create duplicates.
    const pending = fs.promises.open(absolutePath, 'r');
    this.handles.set(absolutePath, pending);

    try {
      return await pending;
    } catch (err) {
      this.handles.delete(absolutePath);
      throw err;
    }
  }

  /** Close every cached shard handle. */
  public async dispose(): Promise<void> {
    const handles = Array.from(this.handles.values());
    this.handles.clear();

    await Promise.all(
      handles.map(async (pending) => {
        try {
          await (await pending).close();
        } catch {
          // Already closed or never opened successfully.
        }
      })
    );
  }

  private trimWindow(now: number): void {
    const cutoff = now - WINDOW_MS;
    while (this.events.length > 0 && this.events[0].at < cutoff) {
      this.events.shift();
    }
  }

  public getIoStats(): StorageIoStats {
    const now = performance.now();
    this.trimWindow(now);

    const windowBytes = this.events.reduce((sum, e) => sum + e.bytes, 0);
    const windowReads = this.events.length;
    const windowSeconds = WINDOW_MS / 1000;
    const busySeconds = this.events.reduce((sum, e) => sum + e.durationMs, 0) / 1000;

    const perDrive: StorageIoStats['driveStats'] = {};
    for (const [driveId, stats] of this.driveStats.entries()) {
      const driveWindowBytes = this.events
        .filter((e) => e.driveId === driveId)
        .reduce((sum, e) => sum + e.bytes, 0);
      perDrive[driveId] = {
        bytesRead: stats.bytesRead,
        reads: stats.reads,
        readMBps: Number((driveWindowBytes / (1024 * 1024) / windowSeconds).toFixed(2)),
      };
    }

    return {
      totalBytesRead: this.totalBytesRead,
      totalReads: this.totalReads,
      activeQueueDepth: this.activeQueueDepth,
      currentBandwidthMBps: Number((windowBytes / (1024 * 1024) / windowSeconds).toFixed(3)),
      averageRequestMBps:
        busySeconds > 0 ? Number((windowBytes / (1024 * 1024) / busySeconds).toFixed(2)) : 0,
      iops: Math.round(windowReads / windowSeconds),
      driveStats: perDrive,
    };
  }

  public resetStats(): void {
    this.events = [];
    this.totalBytesRead = 0;
    this.totalReads = 0;
    for (const key of this.driveStats.keys()) {
      this.driveStats.set(key, { bytesRead: 0, reads: 0 });
    }
  }
}
