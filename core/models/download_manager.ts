import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';
import { getConfigDirectory } from '../config/config.js';
import { DownloadItemProgress, HfFile, downloadGgufModel } from './downloader.js';

/**
 * Tracks downloads for the whole process, independently of any HTTP request.
 *
 * A multi-hundred-gigabyte transfer must not depend on a browser tab staying
 * open: the job is started here and keeps running whether or not anyone is
 * watching, and clients read its progress whenever they reconnect.
 */

export type DownloadStatus = 'running' | 'retrying' | 'completed' | 'failed' | 'cancelled';

export interface DownloadJob {
  id: string;
  repoId: string;
  fileName: string;
  status: DownloadStatus;
  startedAt: string;
  finishedAt: string | null;
  /** Bytes present on disk, including anything a previous attempt left. */
  receivedBytes: number;
  totalBytes: number;
  percent: number;
  /** Transfer rate for the current session only, excluding resumed bytes. */
  bytesPerSecond: number;
  etaSeconds: number | null;
  partIndex: number;
  totalParts: number;
  destination: string | null;
  error: string | null;
  /** How many times the transfer has been retried after a network failure. */
  attempts: number;
  /** When a retry is scheduled, the moment it fires. */
  retryAt: string | null;
}

interface JobRecord {
  job: DownloadJob;
  controller: AbortController;
  retryTimer?: NodeJS.Timeout;
}

/**
 * A transfer of a few hundred gigabytes spans hours, so a dropped connection is
 * a normal event rather than an exceptional one. Every part resumes from a byte
 * offset, which makes a retry nearly free: the only real cost of giving up is
 * the user having to notice and restart it by hand.
 */
const MAX_RETRY_ATTEMPTS = 200;
const RETRY_BASE_MS = 4000;
const RETRY_MAX_MS = 60_000;

function retryDelay(attempt: number): number {
  return Math.min(RETRY_BASE_MS * 2 ** Math.min(attempt, 5), RETRY_MAX_MS);
}

/** Where interrupted downloads are recorded so they can be picked up again. */
function statePath(): string {
  return path.join(getConfigDirectory(), 'downloads.json');
}

interface PersistedJob {
  repoId: string;
  file: HfFile;
}

class DownloadManager {
  private jobs = new Map<string, JobRecord>();
  /** Requests still owed a completed file, keyed by `repoId::path`. */
  private pending = new Map<string, PersistedJob>();

  /**
   * Re-queue anything that was still running when the process last exited.
   *
   * A download is a long-lived job that happens to have been requested over
   * HTTP; restarting the server must not silently abandon 150 GB of transfer.
   * Every part already on disk is skipped, so this resumes rather than restarts.
   */
  public resumeInterrupted(): number {
    let restored: PersistedJob[] = [];
    try {
      restored = JSON.parse(fs.readFileSync(statePath(), 'utf8'));
    } catch {
      return 0;
    }

    let count = 0;
    for (const entry of restored) {
      if (!entry?.repoId || !entry.file?.path) continue;
      this.start(entry.repoId, entry.file);
      count++;
    }
    return count;
  }

  private persist(): void {
    try {
      fs.mkdirSync(getConfigDirectory(), { recursive: true });
      fs.writeFileSync(statePath(), JSON.stringify(Array.from(this.pending.values()), null, 2), 'utf8');
    } catch {
      // Losing the journal only costs automatic resume, not the download.
    }
  }

  /** Start a download and return immediately with its job id. */
  public start(repoId: string, file: HfFile): DownloadJob {
    const existing = this.findActive(repoId, file.path);
    if (existing) return existing;

    this.pending.set(`${repoId}::${file.path}`, { repoId, file });
    this.persist();

    const id = randomUUID();
    const controller = new AbortController();

    const job: DownloadJob = {
      id,
      repoId,
      fileName: file.path,
      status: 'running',
      startedAt: new Date().toISOString(),
      finishedAt: null,
      receivedBytes: 0,
      totalBytes: file.totalSizeBytes,
      percent: 0,
      bytesPerSecond: 0,
      etaSeconds: null,
      partIndex: 1,
      totalParts: file.splitParts?.length || 1,
      destination: null,
      error: null,
      attempts: 0,
      retryAt: null,
    };

    const record: JobRecord = { job, controller };
    this.jobs.set(id, record);
    this.run(record, repoId, file);

    return job;
  }

  /** Run one attempt, scheduling another if the network drops mid-transfer. */
  private run(record: JobRecord, repoId: string, file: HfFile): void {
    const { job } = record;
    job.status = 'running';
    job.error = null;
    job.retryAt = null;

    const onProgress = (p: DownloadItemProgress) => {
      job.receivedBytes = p.overallReceivedBytes;
      job.totalBytes = p.overallTotalBytes || job.totalBytes;
      job.percent = p.percent;
      job.bytesPerSecond = p.bytesPerSecond;
      job.etaSeconds = p.etaSeconds;
      job.partIndex = p.partIndex;
      job.totalParts = p.totalParts;
      job.destination = p.destination;
    };

    // Deliberately not awaited: the caller gets its id back straight away.
    void downloadGgufModel(repoId, file, onProgress, record.controller.signal)
      .then((result) => {
        job.status = 'completed';
        job.percent = 100;
        job.receivedBytes = result.totalBytes;
        job.destination = result.primaryPath;
        job.finishedAt = new Date().toISOString();
        // Finished: no longer owed, so a restart will not re-queue it.
        this.pending.delete(`${repoId}::${file.path}`);
        this.persist();
      })
      .catch((err: Error) => {
        if (record.controller.signal.aborted || err.name === 'AbortError') {
          job.status = 'cancelled';
          job.error = 'Annullato';
          job.finishedAt = new Date().toISOString();
          this.pending.delete(`${repoId}::${file.path}`);
          this.persist();
          return;
        }

        job.attempts++;
        job.bytesPerSecond = 0;
        job.etaSeconds = null;

        if (job.attempts > MAX_RETRY_ATTEMPTS) {
          job.status = 'failed';
          job.error = `${err.message} (giving up after ${job.attempts} attempts)`;
          job.finishedAt = new Date().toISOString();
          return;
        }

        // Every part resumes from its byte offset, so retrying costs nothing
        // already transferred.
        const delay = retryDelay(job.attempts);
        job.status = 'retrying';
        job.error = err.message;
        job.retryAt = new Date(Date.now() + delay).toISOString();

        record.retryTimer = setTimeout(() => {
          if (record.controller.signal.aborted) return;
          this.run(record, repoId, file);
        }, delay);
        record.retryTimer.unref?.();
      });
  }

  private findActive(repoId: string, filePath: string): DownloadJob | null {
    for (const { job } of this.jobs.values()) {
      const active = job.status === 'running' || job.status === 'retrying';
      if (active && job.repoId === repoId && job.fileName === filePath) return job;
    }
    return null;
  }

  public list(): DownloadJob[] {
    return Array.from(this.jobs.values())
      .map((r) => r.job)
      .sort((a, b) => b.startedAt.localeCompare(a.startedAt));
  }

  public get(id: string): DownloadJob | null {
    return this.jobs.get(id)?.job ?? null;
  }

  public cancel(id: string): boolean {
    const record = this.jobs.get(id);
    if (!record) return false;
    if (record.job.status !== 'running' && record.job.status !== 'retrying') return false;

    // Stop a pending retry as well, otherwise the job resurrects itself.
    if (record.retryTimer) clearTimeout(record.retryTimer);
    record.job.status = 'cancelled';
    record.job.error = 'Annullato';
    record.job.retryAt = null;

    this.pending.delete(`${record.job.repoId}::${record.job.fileName}`);
    this.persist();
    record.controller.abort();
    return true;
  }

  /** Forget finished jobs so the list does not grow without bound. */
  public clearFinished(): number {
    let removed = 0;
    for (const [id, record] of this.jobs.entries()) {
      if (record.job.status === 'running' || record.job.status === 'retrying') continue;
      this.jobs.delete(id);
      removed++;
    }
    return removed;
  }
}

export const downloadManager = new DownloadManager();
