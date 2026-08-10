import fs from 'fs';
import os from 'os';
import { ChildProcess, execFile, spawn } from 'child_process';

/**
 * Vendor-neutral live GPU sampling.
 *
 * AILOFlow must monitor whatever GPU the machine actually has — AMD, Intel,
 * NVIDIA or Apple — so no single vendor tool is treated as the source of truth:
 *
 *   Windows  Performance counters (`GPU Engine`, `GPU Adapter Memory`). These
 *            are driver-independent and cover every adapter, which is why they
 *            are the primary path instead of nvidia-smi.
 *   Linux    amdgpu/i915 sysfs for AMD and Intel, nvidia-smi for NVIDIA.
 *   macOS    unified memory accounting.
 *
 * nvidia-smi is used only as an *enrichment* where present, because it is the
 * only source for NVIDIA die temperature.
 */

export type GpuSampleSource = 'windows-counters' | 'nvidia-smi' | 'amdgpu-sysfs' | 'i915-sysfs' | 'unified-memory';

export interface GpuLiveSample {
  /** Adapter this sample belongs to, or null when the source is system-wide. */
  adapterKey: string | null;
  /** Busiest engine utilisation, 0-100. null when the platform cannot report it. */
  utilizationPercent: number | null;
  /** Per-engine breakdown (3D, Compute, Copy, VideoDecode...) when available. */
  engineBreakdown: Record<string, number> | null;
  memoryUsedBytes: number | null;
  memoryTotalBytes: number | null;
  temperatureC: number | null;
  source: GpuSampleSource;
}

export interface GpuMonitorState {
  samples: GpuLiveSample[];
  /** Aggregate across every adapter, for the headline dashboard figure. */
  aggregate: {
    utilizationPercent: number | null;
    memoryUsedBytes: number | null;
    memoryTotalBytes: number | null;
    temperatureC: number | null;
  };
  /** Why no data is available, when that is the case. */
  unavailableReason: string | null;
}

// ---------------------------------------------------------------------------
// Windows: a single long-lived PowerShell that emits one JSON line per tick.
// Spawning a process per sample would cost ~1.5 s and is far too slow.
// ---------------------------------------------------------------------------

const WINDOWS_SAMPLER_SCRIPT = `
$ErrorActionPreference = 'SilentlyContinue'
while ($true) {
  $engines = @{}
  foreach ($s in (Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples) {
    if ($s.InstanceName -match 'engtype_([A-Za-z0-9]+)') {
      $type = $Matches[1]
      $engines[$type] = [double]$engines[$type] + [double]$s.CookedValue
    }
  }
  $adapters = @()
  foreach ($s in (Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples) {
    $adapters += @{ luid = $s.InstanceName; dedicatedBytes = [double]$s.CookedValue }
  }
  $shared = 0
  foreach ($s in (Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples) {
    $shared += [double]$s.CookedValue
  }
  (@{ engines = $engines; adapters = $adapters; sharedBytes = $shared } | ConvertTo-Json -Compress -Depth 4)
  Start-Sleep -Milliseconds 800
}
`;

interface WindowsSample {
  engines: Record<string, number>;
  adapters: Array<{ luid: string; dedicatedBytes: number }>;
  sharedBytes: number;
}

class WindowsCounterSampler {
  private proc: ChildProcess | null = null;
  private latest: WindowsSample | null = null;
  private buffer = '';
  private failed = false;
  private failureReason: string | null = null;

  start(): void {
    if (this.proc || this.failed) return;

    try {
      this.proc = spawn('powershell', ['-NoProfile', '-NonInteractive', '-Command', WINDOWS_SAMPLER_SCRIPT], {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      });
    } catch (err) {
      this.failed = true;
      this.failureReason = `Impossibile avviare il campionatore dei contatori GPU: ${(err as Error).message}`;
      return;
    }

    this.proc.stdout?.on('data', (chunk: Buffer) => {
      this.buffer += chunk.toString();
      // ConvertTo-Json -Compress emits one object per line.
      const lines = this.buffer.split('\n');
      this.buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('{')) continue;
        try {
          this.latest = JSON.parse(trimmed);
        } catch {
          // Partial or malformed frame; the next tick replaces it.
        }
      }
    });

    this.proc.on('exit', () => {
      this.proc = null;
      if (!this.latest) {
        this.failed = true;
        this.failureReason = 'I contatori prestazioni GPU di Windows non sono disponibili su questo sistema.';
      }
    });
  }

  stop(): void {
    this.proc?.kill();
    this.proc = null;
  }

  getLatest(): WindowsSample | null {
    return this.latest;
  }

  getFailureReason(): string | null {
    return this.failureReason;
  }
}

// ---------------------------------------------------------------------------
// nvidia-smi enrichment
// ---------------------------------------------------------------------------

interface NvidiaSample {
  index: number;
  name: string;
  utilizationPercent: number;
  memoryUsedBytes: number;
  memoryTotalBytes: number;
  temperatureC: number;
}

function sampleNvidia(): Promise<NvidiaSample[]> {
  return new Promise((resolve) => {
    execFile(
      'nvidia-smi',
      ['--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
      { timeout: 4000 },
      (err, stdout) => {
        if (err || !stdout) return resolve([]);

        const samples: NvidiaSample[] = [];
        for (const line of stdout.trim().split('\n')) {
          const parts = line.split(',').map((s) => s.trim());
          if (parts.length < 6) continue;
          const [index, name, util, used, total, temp] = parts;
          const numbers = [index, util, used, total, temp].map(Number);
          if (numbers.some((n) => Number.isNaN(n))) continue;

          samples.push({
            index: numbers[0],
            name,
            utilizationPercent: numbers[1],
            memoryUsedBytes: numbers[2] * 1024 * 1024,
            memoryTotalBytes: numbers[3] * 1024 * 1024,
            temperatureC: numbers[4],
          });
        }
        resolve(samples);
      }
    );
  });
}

// ---------------------------------------------------------------------------
// Linux sysfs: amdgpu and i915 expose real per-card counters
// ---------------------------------------------------------------------------

function readNumberFile(filePath: string): number | null {
  try {
    const value = Number(fs.readFileSync(filePath, 'utf8').trim());
    return Number.isNaN(value) ? null : value;
  } catch {
    return null;
  }
}

function sampleLinuxDrm(): GpuLiveSample[] {
  const samples: GpuLiveSample[] = [];
  let cards: string[];
  try {
    cards = fs.readdirSync('/sys/class/drm').filter((n) => /^card\d+$/.test(n));
  } catch {
    return samples;
  }

  for (const card of cards) {
    const device = `/sys/class/drm/${card}/device`;

    // amdgpu exposes utilisation and VRAM directly — real numbers, no vendor tool.
    const busy = readNumberFile(`${device}/gpu_busy_percent`);
    const vramUsed = readNumberFile(`${device}/mem_info_vram_used`);
    const vramTotal = readNumberFile(`${device}/mem_info_vram_total`);

    if (busy === null && vramUsed === null) continue;

    let temperature: number | null = null;
    try {
      const hwmonDir = fs.readdirSync(`${device}/hwmon`)[0];
      if (hwmonDir) {
        const milli = readNumberFile(`${device}/hwmon/${hwmonDir}/temp1_input`);
        temperature = milli === null ? null : Math.round(milli / 1000);
      }
    } catch {
      // No hwmon node for this card.
    }

    samples.push({
      adapterKey: card,
      utilizationPercent: busy,
      engineBreakdown: null,
      memoryUsedBytes: vramUsed,
      memoryTotalBytes: vramTotal,
      temperatureC: temperature,
      source: fs.existsSync(`${device}/mem_info_vram_total`) ? 'amdgpu-sysfs' : 'i915-sysfs',
    });
  }

  return samples;
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

export class GpuMonitor {
  private windows = new WindowsCounterSampler();
  private nvidia: NvidiaSample[] = [];
  private nvidiaProbed = false;
  private nvidiaAvailable = false;
  private handle: NodeJS.Timeout | null = null;

  /** Adapters with dedicated VRAM, used to attribute counter data. */
  private dedicatedAdapters: Array<{ key: string; name: string; vramTotalBytes: number | null }> = [];

  public setAdapters(adapters: Array<{ key: string; name: string; vramTotalBytes: number | null }>): void {
    this.dedicatedAdapters = adapters;
  }

  public start(intervalMs = 1500): void {
    if (os.platform() === 'win32') this.windows.start();
    if (this.handle) return;

    void this.sample();
    this.handle = setInterval(() => { void this.sample(); }, intervalMs);
    this.handle.unref?.();
  }

  public stop(): void {
    if (this.handle) clearInterval(this.handle);
    this.handle = null;
    this.windows.stop();
  }

  public async sample(): Promise<void> {
    // nvidia-smi is probed once: if it is absent, retrying every tick is waste.
    if (!this.nvidiaProbed) {
      this.nvidia = await sampleNvidia();
      this.nvidiaProbed = true;
      this.nvidiaAvailable = this.nvidia.length > 0;
    } else if (this.nvidiaAvailable) {
      this.nvidia = await sampleNvidia();
    }
  }

  public getState(): GpuMonitorState {
    const samples: GpuLiveSample[] = [];
    const platform = os.platform();

    // 1. NVIDIA cards report themselves precisely, including temperature.
    for (const gpu of this.nvidia) {
      samples.push({
        adapterKey: `nvidia-${gpu.index}`,
        utilizationPercent: gpu.utilizationPercent,
        engineBreakdown: null,
        memoryUsedBytes: gpu.memoryUsedBytes,
        memoryTotalBytes: gpu.memoryTotalBytes,
        temperatureC: gpu.temperatureC,
        source: 'nvidia-smi',
      });
    }

    // 2. Windows counters cover every adapter regardless of vendor.
    if (platform === 'win32') {
      const latest = this.windows.getLatest();
      if (latest) {
        const engineBreakdown: Record<string, number> = {};
        for (const [type, value] of Object.entries(latest.engines || {})) {
          engineBreakdown[type] = Number(Math.min(100, value).toFixed(1));
        }
        // The busiest engine is the meaningful headline: for LLM inference that
        // is usually Compute, while 3D stays near idle.
        const busiest = Object.values(engineBreakdown).length
          ? Math.max(...Object.values(engineBreakdown))
          : null;

        const used = (latest.adapters || []).reduce((sum, a) => sum + a.dedicatedBytes, 0);
        // Attribute memory to a single adapter only when that is unambiguous.
        const withVram = this.dedicatedAdapters.filter((a) => a.vramTotalBytes !== null);
        const soleAdapter = withVram.length === 1 ? withVram[0] : null;

        samples.push({
          adapterKey: soleAdapter ? soleAdapter.key : null,
          utilizationPercent: busiest,
          engineBreakdown,
          memoryUsedBytes: used,
          memoryTotalBytes: soleAdapter
            ? soleAdapter.vramTotalBytes
            : withVram.reduce((sum, a) => sum + (a.vramTotalBytes || 0), 0) || null,
          temperatureC: null,
          source: 'windows-counters',
        });
      }
    }

    // 3. Linux sysfs.
    if (platform === 'linux') {
      samples.push(...sampleLinuxDrm());
    }

    // 4. Apple unified memory: the GPU shares system RAM, which is a real figure.
    if (platform === 'darwin' && samples.length === 0) {
      samples.push({
        adapterKey: 'apple-0',
        utilizationPercent: null,
        engineBreakdown: null,
        memoryUsedBytes: os.totalmem() - os.freemem(),
        memoryTotalBytes: os.totalmem(),
        temperatureC: null,
        source: 'unified-memory',
      });
    }

    const utilisations = samples.map((s) => s.utilizationPercent).filter((v): v is number => v !== null);
    const memUsed = samples.map((s) => s.memoryUsedBytes).filter((v): v is number => v !== null);
    const memTotal = samples.map((s) => s.memoryTotalBytes).filter((v): v is number => v !== null);
    const temps = samples.map((s) => s.temperatureC).filter((v): v is number => v !== null);

    let unavailableReason: string | null = null;
    if (samples.length === 0) {
      unavailableReason =
        platform === 'win32'
          ? this.windows.getFailureReason() || 'Campionamento dei contatori GPU non ancora avviato.'
          : 'Nessuna sorgente di telemetria GPU disponibile su questa piattaforma.';
    }

    return {
      samples,
      aggregate: {
        utilizationPercent: utilisations.length ? Number(Math.max(...utilisations).toFixed(1)) : null,
        memoryUsedBytes: memUsed.length ? memUsed.reduce((a, b) => a + b, 0) : null,
        memoryTotalBytes: memTotal.length ? memTotal.reduce((a, b) => a + b, 0) : null,
        temperatureC: temps.length ? Math.max(...temps) : null,
      },
      unavailableReason,
    };
  }
}
