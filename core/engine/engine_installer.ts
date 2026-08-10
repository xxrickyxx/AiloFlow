import fs from 'fs';
import os from 'os';
import path from 'path';
import { execFile } from 'child_process';
import { ComputerProfile } from '../hardware/types.js';
import { getConfigDirectory, updateConfig } from '../config/config.js';

/**
 * Installs a real inference engine so AILOFlow does not depend on any external
 * daemon.
 *
 * The compute kernels come from llama.cpp — per the project brief, hand-written
 * GPU kernels are not something to reinvent — but the binaries are fetched,
 * verified and managed by AILOFlow itself and run as our own child process.
 */

const RELEASES_API = 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest';

export type EngineVariant = 'cuda' | 'hip' | 'vulkan' | 'metal' | 'cpu';

export interface EngineCandidate {
  variant: EngineVariant;
  assetName: string;
  downloadUrl: string;
  sizeBytes: number;
  /** Why this build suits the detected hardware. */
  rationale: string;
  /** Extra archives that must be installed alongside (CUDA runtime). */
  companionAssets: Array<{ name: string; url: string; sizeBytes: number }>;
}

export interface InstalledEngine {
  variant: EngineVariant;
  release: string;
  serverPath: string;
  installedAt: string;
  version: string | null;
}

export interface DownloadProgress {
  phase: 'download' | 'extract' | 'verify';
  assetName: string;
  receivedBytes: number;
  totalBytes: number;
  percent: number;
  bytesPerSecond: number;
}

interface GithubAsset {
  name: string;
  browser_download_url: string;
  size: number;
}

interface GithubRelease {
  tag_name: string;
  assets: GithubAsset[];
}

export function getEngineDirectory(): string {
  return path.join(getConfigDirectory(), 'engines');
}

async function fetchLatestRelease(): Promise<GithubRelease> {
  const res = await fetch(RELEASES_API, {
    headers: { Accept: 'application/vnd.github+json', 'User-Agent': 'AILOFlow' },
  });
  if (!res.ok) {
    throw new Error(
      `Impossibile leggere le release di llama.cpp (HTTP ${res.status}). ` +
        'Serve una connessione a github.com per installare il motore.'
    );
  }
  return (await res.json()) as GithubRelease;
}

/**
 * Rank the available builds against the machine.
 *
 * The order matters: a vendor-specific build beats the portable one, and the
 * CPU build is always offered last as a guaranteed fallback.
 */
export async function listEngineCandidates(profile: ComputerProfile): Promise<{
  release: string;
  candidates: EngineCandidate[];
}> {
  const release = await fetchLatestRelease();
  const platform = os.platform();
  const arch = os.arch();

  const vendors = new Set(profile.gpus.map((g) => g.vendor));
  const candidates: EngineCandidate[] = [];

  const find = (predicate: (name: string) => boolean): GithubAsset | undefined =>
    release.assets.find((a) => predicate(a.name.toLowerCase()));

  const platformTag = platform === 'win32' ? 'win' : platform === 'darwin' ? 'macos' : 'ubuntu';
  const archTag = arch === 'arm64' ? 'arm64' : 'x64';

  const push = (
    variant: EngineVariant,
    asset: GithubAsset | undefined,
    rationale: string,
    companions: GithubAsset[] = []
  ) => {
    if (!asset) return;
    candidates.push({
      variant,
      assetName: asset.name,
      downloadUrl: asset.browser_download_url,
      sizeBytes: asset.size,
      rationale,
      companionAssets: companions.map((c) => ({ name: c.name, url: c.browser_download_url, sizeBytes: c.size })),
    });
  };

  if (vendors.has('NVIDIA')) {
    // The CUDA build needs the matching cudart archive shipped in the release.
    const cuda = find((n) => n.includes(`bin-${platformTag}-cuda`) && n.includes(archTag));
    const cudartVersion = cuda?.name.match(/cuda-([\d.]+)/)?.[1];
    const cudart = cudartVersion
      ? find((n) => n.startsWith('cudart-') && n.includes(cudartVersion))
      : undefined;
    push('cuda', cuda, 'GPU NVIDIA rilevata: CUDA è il backend più veloce.', cudart ? [cudart] : []);
  }

  if (vendors.has('AMD')) {
    push(
      'hip',
      find((n) => n.includes(`bin-${platformTag}-hip`) && n.includes(archTag)),
      'GPU AMD rilevata: la build HIP/ROCm usa i kernel nativi Radeon.'
    );
  }

  if (platform === 'darwin') {
    push('metal', find((n) => n.includes('macos') && n.includes(archTag)), 'Su macOS il backend Metal è integrato nella build.');
  }

  // Vulkan runs on AMD, Intel and NVIDIA alike: always a valid GPU option.
  if (platform !== 'darwin' && profile.gpus.length > 0) {
    push(
      'vulkan',
      find((n) => n.includes(`bin-${platformTag}-vulkan`) && n.includes(archTag)),
      'Vulkan funziona su qualsiasi GPU recente (AMD, Intel, NVIDIA).'
    );
  }

  push(
    'cpu',
    find((n) => n.includes(`bin-${platformTag}-cpu`) && n.includes(archTag)),
    'Esecuzione su CPU: sempre disponibile, nessuna GPU richiesta.'
  );

  if (candidates.length === 0) {
    throw new Error(
      `Nessuna build precompilata di llama.cpp per ${platform}/${arch} nella release ${release.tag_name}. ` +
        'Su questa piattaforma il motore va compilato a mano e indicato nelle impostazioni.'
    );
  }

  return { release: release.tag_name, candidates };
}

/** Stream a URL to disk, reporting real throughput as it goes. */
async function downloadFile(
  url: string,
  destination: string,
  assetName: string,
  onProgress?: (p: DownloadProgress) => void
): Promise<void> {
  const res = await fetch(url, { redirect: 'follow', headers: { 'User-Agent': 'AILOFlow' } });
  if (!res.ok || !res.body) throw new Error(`Download fallito (HTTP ${res.status}): ${assetName}`);

  const totalBytes = Number(res.headers.get('content-length') || 0);
  const handle = await fs.promises.open(destination, 'w');
  const reader = res.body.getReader();

  let received = 0;
  const startedAt = performance.now();
  let lastReport = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await handle.write(value);
      received += value.length;

      const now = performance.now();
      if (onProgress && now - lastReport > 250) {
        lastReport = now;
        const seconds = (now - startedAt) / 1000;
        onProgress({
          phase: 'download',
          assetName,
          receivedBytes: received,
          totalBytes,
          percent: totalBytes ? Number(((received / totalBytes) * 100).toFixed(1)) : 0,
          bytesPerSecond: seconds > 0 ? Math.round(received / seconds) : 0,
        });
      }
    }
  } finally {
    await handle.close();
  }
}

/**
 * Extract a zip archive.
 *
 * `tar` ships with Windows 10+, macOS and every Linux distro we target and
 * understands zip, so no third-party dependency is pulled in for this.
 */
function extractArchive(archivePath: string, targetDir: string): Promise<void> {
  return new Promise((resolve, reject) => {
    fs.mkdirSync(targetDir, { recursive: true });
    execFile('tar', ['-xf', archivePath, '-C', targetDir], { timeout: 300_000 }, (err, _stdout, stderr) => {
      if (err) {
        reject(new Error(`Estrazione di ${path.basename(archivePath)} fallita: ${stderr || err.message}`));
        return;
      }
      resolve();
    });
  });
}

/** Locate the llama-server executable anywhere under the extracted tree. */
function findServerBinary(root: string): string | null {
  const targets = os.platform() === 'win32' ? ['llama-server.exe'] : ['llama-server'];
  const stack = [root];

  while (stack.length) {
    const dir = stack.pop()!;
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) stack.push(full);
      else if (targets.includes(entry.name)) return full;
    }
  }

  return null;
}

function readEngineVersion(serverPath: string): Promise<string | null> {
  return new Promise((resolve) => {
    execFile(serverPath, ['--version'], { timeout: 20000 }, (err, stdout, stderr) => {
      const output = `${stdout || ''}${stderr || ''}`.trim();
      if (err && !output) return resolve(null);
      const line = output.split('\n').find((l) => l.toLowerCase().includes('version')) || output.split('\n')[0];
      resolve(line ? line.trim().slice(0, 120) : null);
    });
  });
}

/**
 * Download, extract and register an engine build. On success the config points
 * at the new binary, so llama.cpp becomes available without any manual step.
 */
export async function installEngine(
  candidate: EngineCandidate,
  release: string,
  onProgress?: (p: DownloadProgress) => void
): Promise<InstalledEngine> {
  const engineRoot = getEngineDirectory();
  const targetDir = path.join(engineRoot, `llama.cpp-${release}-${candidate.variant}`);
  const tempDir = path.join(engineRoot, '.downloads');
  fs.mkdirSync(tempDir, { recursive: true });

  const archives = [
    { name: candidate.assetName, url: candidate.downloadUrl },
    ...candidate.companionAssets.map((c) => ({ name: c.name, url: c.url })),
  ];

  try {
    for (const archive of archives) {
      const archivePath = path.join(tempDir, archive.name);
      await downloadFile(archive.url, archivePath, archive.name, onProgress);

      onProgress?.({
        phase: 'extract',
        assetName: archive.name,
        receivedBytes: 0,
        totalBytes: 0,
        percent: 100,
        bytesPerSecond: 0,
      });
      // Companion archives (CUDA runtime DLLs) extract into the same tree so the
      // server finds them next to itself.
      await extractArchive(archivePath, targetDir);
      fs.unlinkSync(archivePath);
    }

    const serverPath = findServerBinary(targetDir);
    if (!serverPath) {
      throw new Error(`Archivio estratto ma "llama-server" non trovato in ${targetDir}.`);
    }

    if (os.platform() !== 'win32') fs.chmodSync(serverPath, 0o755);

    onProgress?.({
      phase: 'verify',
      assetName: candidate.assetName,
      receivedBytes: 0,
      totalBytes: 0,
      percent: 100,
      bytesPerSecond: 0,
    });
    const version = await readEngineVersion(serverPath);

    const installed: InstalledEngine = {
      variant: candidate.variant,
      release,
      serverPath,
      installedAt: new Date().toISOString(),
      version,
    };

    updateConfig({ llamaServerPath: serverPath, installedEngine: installed });
    return installed;
  } finally {
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch {
      // Temp cleanup is best effort.
    }
  }
}

/** Engines already unpacked under the config directory. */
export function listInstalledEngines(): InstalledEngine[] {
  const root = getEngineDirectory();
  if (!fs.existsSync(root)) return [];

  const installed: InstalledEngine[] = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    const match = entry.name.match(/^llama\.cpp-(.+)-(cuda|hip|vulkan|metal|cpu)$/);
    if (!match) continue;

    const serverPath = findServerBinary(path.join(root, entry.name));
    if (!serverPath) continue;

    installed.push({
      variant: match[2] as EngineVariant,
      release: match[1],
      serverPath,
      installedAt: fs.statSync(serverPath).mtime.toISOString(),
      version: null,
    });
  }

  return installed;
}
