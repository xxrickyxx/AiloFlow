import fs from 'fs';
import os from 'os';
import path from 'path';
import { loadConfig, updateConfig } from '../config/config.js';

/**
 * Hugging Face model discovery and download.
 *
 * Large GGUF models are published as split archives (`...-00001-of-00009.gguf`)
 * because of the 50 GB per-file limit, so a "download" here usually means
 * fetching a whole ordered set. Downloads resume with a Range request, which
 * matters when a single model is several hundred gigabytes.
 */

const HF_API = 'https://huggingface.co/api';

export interface HfModelSummary {
  repoId: string;
  downloads: number;
  likes: number;
  updatedAt: string | null;
  tags: string[];
}

export interface HfFile {
  path: string;
  sizeBytes: number;
  /** Files belonging to a split model, in order, when this is part 1. */
  splitParts?: string[];
  totalSizeBytes: number;
  quantization: string | null;
}

export interface DownloadItemProgress {
  repoId: string;
  fileName: string;
  partIndex: number;
  totalParts: number;
  receivedBytes: number;
  totalBytes: number;
  overallReceivedBytes: number;
  overallTotalBytes: number;
  percent: number;
  bytesPerSecond: number;
  etaSeconds: number | null;
  destination: string;
}

/**
 * Where downloads land.
 *
 * Without an explicit setting we pick the writable volume with the most free
 * space rather than the home directory: a single large model can be hundreds of
 * gigabytes, and the system drive is rarely where it should go.
 */
export function getDownloadDirectory(): string {
  const configured = loadConfig().downloadDirectory;
  if (configured) return configured;

  const roomiest = pickRoomiestVolume();
  return roomiest ? path.join(roomiest, 'AiloFlowModels') : path.join(os.homedir(), 'AiloFlowModels');
}

function pickRoomiestVolume(): string | null {
  const candidates: string[] = [];

  if (os.platform() === 'win32') {
    for (let code = 'C'.charCodeAt(0); code <= 'Z'.charCodeAt(0); code++) {
      candidates.push(`${String.fromCharCode(code)}:\\`);
    }
  } else {
    candidates.push(os.homedir(), '/');
  }

  let best: { root: string; free: number } | null = null;
  for (const root of candidates) {
    try {
      if (!fs.existsSync(root)) continue;
      const stats = fs.statfsSync(root);
      const free = stats.bavail * stats.bsize;
      if (free > 0 && (!best || free > best.free)) best = { root, free };
    } catch {
      // Not a mounted volume, or not readable.
    }
  }

  return best ? best.root : null;
}

export function setDownloadDirectory(directory: string): void {
  fs.mkdirSync(directory, { recursive: true });
  const config = loadConfig();
  const dirs = new Set(config.modelDirectories);
  dirs.add(directory);
  updateConfig({ downloadDirectory: directory, modelDirectories: Array.from(dirs) });
}

/** Search Hugging Face for repositories that publish GGUF files. */
export async function searchGgufRepos(query: string, limit = 20): Promise<HfModelSummary[]> {
  const url = `${HF_API}/models?search=${encodeURIComponent(query)}&filter=gguf&sort=downloads&direction=-1&limit=${limit}`;
  const res = await fetch(url, { headers: { 'User-Agent': 'AILOFlow' } });
  if (!res.ok) throw new Error(`Ricerca su Hugging Face fallita (HTTP ${res.status}).`);

  const body = (await res.json()) as Array<{
    id: string;
    downloads?: number;
    likes?: number;
    lastModified?: string;
    tags?: string[];
  }>;

  return body.map((m) => ({
    repoId: m.id,
    downloads: m.downloads ?? 0,
    likes: m.likes ?? 0,
    updatedAt: m.lastModified ?? null,
    tags: m.tags ?? [],
  }));
}

const SPLIT_PATTERN = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;

function detectQuantization(fileName: string): string | null {
  const match = fileName.match(/(IQ\d[A-Z_]*|Q\d(?:_[A-Z0-9]+)*|BF16|F16|F32)/i);
  return match ? match[1].toUpperCase() : null;
}

/**
 * List the GGUF files in a repository, collapsing split sets into a single
 * logical entry whose size is the sum of its parts.
 */
export async function listRepoGgufFiles(repoId: string): Promise<HfFile[]> {
  const collected: Array<{ path: string; size: number }> = [];

  // The tree endpoint is not recursive by default; quantisations often live in
  // per-quant subdirectories, so directories are walked explicitly.
  const walk = async (subPath: string, depth: number): Promise<void> => {
    if (depth > 2) return;
    const url = `${HF_API}/models/${repoId}/tree/main${subPath ? `/${subPath}` : ''}`;
    const res = await fetch(url, { headers: { 'User-Agent': 'AILOFlow' } });
    if (!res.ok) {
      if (!subPath) throw new Error(`Repository "${repoId}" non leggibile (HTTP ${res.status}).`);
      return;
    }

    const entries = (await res.json()) as Array<{ type: string; path: string; size: number }>;
    for (const entry of entries) {
      if (entry.type === 'directory') {
        await walk(entry.path, depth + 1);
      } else if (entry.path.toLowerCase().endsWith('.gguf')) {
        collected.push({ path: entry.path, size: entry.size });
      }
    }
  };

  await walk('', 0);

  const bySplitBase = new Map<string, Array<{ path: string; size: number; index: number; total: number }>>();
  const singles: HfFile[] = [];

  for (const file of collected) {
    const base = path.posix.basename(file.path);
    const match = base.match(SPLIT_PATTERN);
    if (match) {
      const key = path.posix.join(path.posix.dirname(file.path), match[1]);
      const list = bySplitBase.get(key) || [];
      list.push({ ...file, index: Number(match[2]), total: Number(match[3]) });
      bySplitBase.set(key, list);
    } else {
      singles.push({
        path: file.path,
        sizeBytes: file.size,
        totalSizeBytes: file.size,
        quantization: detectQuantization(base),
      });
    }
  }

  const splits: HfFile[] = [];
  for (const [key, parts] of bySplitBase.entries()) {
    parts.sort((a, b) => a.index - b.index);
    const totalSizeBytes = parts.reduce((sum, p) => sum + p.size, 0);
    splits.push({
      path: parts[0].path,
      sizeBytes: parts[0].size,
      splitParts: parts.map((p) => p.path),
      totalSizeBytes,
      quantization: detectQuantization(path.posix.basename(key)),
    });
  }

  return [...splits, ...singles].sort((a, b) => b.totalSizeBytes - a.totalSizeBytes);
}

const LOCK_STALE_MS = 30_000;

/**
 * Guard a destination against concurrent writers.
 *
 * The same model can be requested from the GUI, the CLI and an auto-resume at
 * once. Two processes appending to one file produce a corrupt model that only
 * fails much later at load time, so a heartbeat lock refuses the second writer
 * outright. A lock older than the heartbeat interval belongs to a dead process
 * and is ignored.
 */
class DownloadLock {
  private readonly lockPath: string;
  private handle: NodeJS.Timeout | null = null;

  constructor(destination: string) {
    this.lockPath = `${destination}.ailolock`;
  }

  acquire(): boolean {
    try {
      const stats = fs.statSync(this.lockPath);
      if (Date.now() - stats.mtimeMs < LOCK_STALE_MS) return false;
    } catch {
      // No lock present.
    }

    try {
      fs.writeFileSync(this.lockPath, String(process.pid));
    } catch {
      return false;
    }

    // Refresh while the transfer runs so the lock never looks stale.
    this.handle = setInterval(() => {
      try {
        fs.utimesSync(this.lockPath, new Date(), new Date());
      } catch {
        // Lock vanished; nothing useful to do from here.
      }
    }, LOCK_STALE_MS / 3);
    this.handle.unref?.();
    return true;
  }

  release(): void {
    if (this.handle) clearInterval(this.handle);
    this.handle = null;
    try {
      fs.unlinkSync(this.lockPath);
    } catch {
      // Already gone.
    }
  }
}

async function downloadPart(
  repoId: string,
  remotePath: string,
  destination: string,
  context: {
    partIndex: number;
    totalParts: number;
    overallReceivedBefore: number;
    overallTotalBytes: number;
    startedAt: number;
    /** Bytes already on disk when this run began; excluded from rate maths. */
    resumedBytes: () => number;
  },
  onProgress?: (p: DownloadItemProgress) => void,
  signal?: AbortSignal
): Promise<number> {
  const url = `https://huggingface.co/${repoId}/resolve/main/${remotePath}`;

  // Resume support: a partial file means we ask for the remaining range only.
  let existingBytes = 0;
  try {
    existingBytes = fs.statSync(destination).size;
  } catch {
    existingBytes = 0;
  }

  const headers: Record<string, string> = { 'User-Agent': 'AILOFlow' };
  if (existingBytes > 0) headers.Range = `bytes=${existingBytes}-`;

  const res = await fetch(url, { headers, redirect: 'follow', signal });

  if (res.status === 416) return existingBytes; // Already complete.
  if (!res.ok || !res.body) {
    throw new Error(`Download di ${remotePath} fallito (HTTP ${res.status}).`);
  }

  // A server that ignores Range restarts the file; truncate to stay consistent.
  const resuming = res.status === 206;
  if (existingBytes > 0 && !resuming) existingBytes = 0;

  const contentLength = Number(res.headers.get('content-length') || 0);
  const totalBytes = existingBytes + contentLength;

  fs.mkdirSync(path.dirname(destination), { recursive: true });
  const handle = await fs.promises.open(destination, resuming && existingBytes > 0 ? 'a' : 'w');

  let received = existingBytes;
  let lastReport = 0;
  const reader = res.body.getReader();

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await handle.write(value);
      received += value.length;

      const now = performance.now();
      if (onProgress && now - lastReport > 300) {
        lastReport = now;
        const overallReceived = context.overallReceivedBefore + received;
        const elapsedSec = (now - context.startedAt) / 1000;
        // Bytes a previous attempt already wrote were never transferred in this
        // session; counting them would report an absurd rate on every resume.
        const transferredThisSession = Math.max(0, overallReceived - context.resumedBytes());
        const rate = elapsedSec > 0 ? transferredThisSession / elapsedSec : 0;
        const remaining = context.overallTotalBytes - overallReceived;

        onProgress({
          repoId,
          fileName: path.posix.basename(remotePath),
          partIndex: context.partIndex,
          totalParts: context.totalParts,
          receivedBytes: received,
          totalBytes,
          overallReceivedBytes: overallReceived,
          overallTotalBytes: context.overallTotalBytes,
          percent: context.overallTotalBytes
            ? Number(((overallReceived / context.overallTotalBytes) * 100).toFixed(2))
            : 0,
          bytesPerSecond: Math.round(rate),
          etaSeconds: rate > 0 ? Math.round(remaining / rate) : null,
          destination,
        });
      }
    }
  } finally {
    await handle.close();
  }

  return received;
}

export interface DownloadResult {
  repoId: string;
  files: string[];
  /** Path to open with the runtime: part 1 of a split set, or the single file. */
  primaryPath: string;
  totalBytes: number;
  durationMs: number;
}

/**
 * Download a GGUF (including every part of a split set) into the model folder.
 * Existing complete parts are skipped, so an interrupted transfer resumes.
 */
export async function downloadGgufModel(
  repoId: string,
  file: HfFile,
  onProgress?: (p: DownloadItemProgress) => void,
  signal?: AbortSignal
): Promise<DownloadResult> {
  const parts = file.splitParts && file.splitParts.length > 0 ? file.splitParts : [file.path];
  const baseDir = path.join(getDownloadDirectory(), repoId.replace('/', '__'));
  fs.mkdirSync(baseDir, { recursive: true });

  const startedAt = performance.now();
  const written: string[] = [];
  let overallReceived = 0;
  // Everything a previous attempt left behind, so throughput reflects only the
  // bytes this run actually pulls over the network.
  let resumedBytes = 0;

  for (let i = 0; i < parts.length; i++) {
    const remotePath = parts[i];
    const destination = path.join(baseDir, path.posix.basename(remotePath));

    const lock = new DownloadLock(destination);
    if (!lock.acquire()) {
      throw new Error(
        `${path.posix.basename(remotePath)} is already being downloaded by another process. ` +
          'Wait for it to finish rather than starting a second transfer into the same file.'
      );
    }

    try {
      resumedBytes += fs.statSync(destination).size;
    } catch {
      // Nothing on disk for this part yet.
    }

    const alreadyResumed = resumedBytes;
    let received: number;
    try {
      received = await downloadPart(
      repoId,
      remotePath,
      destination,
      {
        partIndex: i + 1,
        totalParts: parts.length,
        overallReceivedBefore: overallReceived,
        overallTotalBytes: file.totalSizeBytes,
        startedAt,
        resumedBytes: () => alreadyResumed,
      },
        onProgress,
        signal
      );
    } finally {
      lock.release();
    }

    overallReceived += received;
    written.push(destination);
  }

  // Make the folder visible to model discovery from now on.
  const config = loadConfig();
  if (!config.modelDirectories.includes(getDownloadDirectory())) {
    updateConfig({ modelDirectories: [...config.modelDirectories, getDownloadDirectory()] });
  }

  return {
    repoId,
    files: written,
    primaryPath: written[0],
    totalBytes: overallReceived,
    durationMs: Number((performance.now() - startedAt).toFixed(0)),
  };
}

/**
 * Free space on the volume holding the download directory.
 *
 * The directory itself may not exist yet, so the closest existing ancestor is
 * measured instead — statfs on a missing path would just report nothing.
 */
export function getDownloadDirectoryFreeBytes(): number | null {
  let probe = getDownloadDirectory();

  for (let depth = 0; depth < 8; depth++) {
    try {
      if (fs.existsSync(probe)) {
        const stats = fs.statfsSync(probe);
        return stats.bavail * stats.bsize;
      }
    } catch {
      // Fall through and try the parent.
    }
    const parent = path.dirname(probe);
    if (parent === probe) break;
    probe = parent;
  }

  return null;
}
