import { useEffect, useState } from 'react';
import { API_BASE } from './api';

/**
 * Client view of the server-side download registry.
 *
 * Progress is polled rather than streamed so that it survives tab switches and
 * full page reloads: the transfer belongs to the runtime, and the interface is
 * just an observer that can come and go.
 */

export type DownloadStatus = 'running' | 'retrying' | 'completed' | 'failed' | 'cancelled';

export interface DownloadJob {
  id: string;
  repoId: string;
  fileName: string;
  status: DownloadStatus;
  startedAt: string;
  finishedAt: string | null;
  receivedBytes: number;
  totalBytes: number;
  percent: number;
  bytesPerSecond: number;
  etaSeconds: number | null;
  partIndex: number;
  totalParts: number;
  destination: string | null;
  error: string | null;
  attempts: number;
  retryAt: string | null;
}

export async function startDownload(repoId: string, file: unknown): Promise<DownloadJob> {
  const res = await fetch(`${API_BASE}/downloads`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repoId, file }),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body?.error || `HTTP ${res.status}`);
  return body.job as DownloadJob;
}

export async function cancelDownload(id: string): Promise<void> {
  await fetch(`${API_BASE}/downloads/${id}/cancel`, { method: 'POST' });
}

export async function clearFinishedDownloads(): Promise<void> {
  await fetch(`${API_BASE}/downloads/clear`, { method: 'POST' });
}

export interface DownloadsState {
  jobs: DownloadJob[];
  active: DownloadJob[];
  directory: string | null;
  refresh: () => void;
}

export function useDownloads(pollMs = 1000): DownloadsState {
  const [jobs, setJobs] = useState<DownloadJob[]>([]);
  const [directory, setDirectory] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/downloads`);
        if (!res.ok) return;
        const body = await res.json();
        if (cancelled) return;
        setJobs(body.jobs || []);
        setDirectory(body.directory || null);
      } catch {
        // Server not reachable; the connection badge already reports that.
      }
    };

    void poll();
    const handle = window.setInterval(poll, pollMs);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [pollMs, nonce]);

  return {
    jobs,
    // A job waiting to retry is still in flight: it must not look finished, and
    // starting a duplicate for the same file would corrupt it.
    active: jobs.filter((j) => j.status === 'running' || j.status === 'retrying'),
    directory,
    refresh: () => setNonce((n) => n + 1),
  };
}
