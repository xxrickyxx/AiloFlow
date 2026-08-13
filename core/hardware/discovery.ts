import fs from 'fs';
import os from 'os';
import { execSync } from 'child_process';
import { loadConfig } from '../config/config.js';
import {
  ComputerProfile,
  CpuInfo,
  GpuInfo,
  RamInfo,
  StorageDriveInfo,
  StoragePerformanceProfile,
} from './types.js';

/**
 * Run a PowerShell snippet and return its stdout.
 *
 * `exit 0` is appended deliberately: cmdlets run with -ErrorAction
 * SilentlyContinue still leave `$?` false, so PowerShell exits non-zero even
 * when it printed exactly the data we asked for — which would make execSync
 * throw and silently discard a perfectly good result. Callers validate the
 * payload by parsing it, so the process exit code carries no information here.
 */
function runPowerShell(script: string, timeoutMs = 10000): string | null {
  try {
    return execSync(`powershell -NoProfile -NonInteractive -Command "${script.replace(/"/g, '\\"')}; exit 0"`, {
      encoding: 'utf8',
      timeout: timeoutMs,
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
  } catch {
    return null;
  }
}

/**
 * Query the CPU's actual instruction-set support.
 *
 * Windows: IsProcessorFeaturePresent via P/Invoke — the same call the OS uses.
 * Linux:   /proc/cpuinfo flags.
 * macOS:   sysctl machdep.cpu features.
 *
 * Anything we cannot verify comes back false rather than assumed-true, so the
 * optimizer never selects a kernel the CPU may not support.
 */
function detectCpuInstructions(): CpuInfo['instructions'] {
  const platform = os.platform();
  const none = { sse: false, sse2: false, avx: false, avx2: false, avx512: false, neon: false };

  try {
    if (platform === 'win32') {
      // PF_* constants from winnt.h
      const script = [
        'Add-Type -Name PF -Namespace W -MemberDefinition',
        "'[System.Runtime.InteropServices.DllImport(\"kernel32.dll\")] public static extern bool IsProcessorFeaturePresent(uint f);'",
        ';',
        '$r = @{};',
        "foreach ($p in @{sse=6;sse2=10;avx=39;avx2=40;avx512=41;neon=19}.GetEnumerator()) { $r[$p.Key] = [W.PF]::IsProcessorFeaturePresent($p.Value) };",
        '$r | ConvertTo-Json -Compress',
      ].join(' ');

      const out = runPowerShell(script, 15000);
      if (!out) return none;
      const parsed = JSON.parse(out);
      return {
        sse: parsed.sse === true,
        sse2: parsed.sse2 === true,
        avx: parsed.avx === true,
        avx2: parsed.avx2 === true,
        avx512: parsed.avx512 === true,
        neon: parsed.neon === true,
      };
    }

    if (platform === 'linux') {
      const cpuinfo = fs.readFileSync('/proc/cpuinfo', 'utf8');
      const flagsLine = cpuinfo.split('\n').find((l) => l.startsWith('flags') || l.startsWith('Features')) || '';
      const flags = new Set(flagsLine.split(':')[1]?.trim().split(/\s+/) || []);
      return {
        sse: flags.has('sse'),
        sse2: flags.has('sse2'),
        avx: flags.has('avx'),
        avx2: flags.has('avx2'),
        avx512: flags.has('avx512f'),
        neon: flags.has('neon') || flags.has('asimd'),
      };
    }

    if (platform === 'darwin') {
      const features = execSync('sysctl -n machdep.cpu.features machdep.cpu.leaf7_features hw.optional.neon', {
        encoding: 'utf8',
        timeout: 5000,
        stdio: ['pipe', 'pipe', 'pipe'],
      }).toUpperCase();
      return {
        sse: features.includes('SSE'),
        sse2: features.includes('SSE2'),
        avx: features.includes('AVX1.0') || features.includes('AVX'),
        avx2: features.includes('AVX2'),
        avx512: features.includes('AVX512F'),
        neon: process.arch === 'arm64' || features.includes('\n1'),
      };
    }
  } catch {
    // Probe unavailable — fall through to the conservative default below.
  }

  // Apple Silicon and other ARM64 targets always have NEON by architecture.
  if (process.arch === 'arm64') return { ...none, neon: true };
  return none;
}

function detectPhysicalCores(logicalThreads: number): number | null {
  const platform = os.platform();
  try {
    if (platform === 'win32') {
      const out = runPowerShell('(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum', 8000);
      if (!out) return null;
      const cores = parseInt(out.trim(), 10);
      return Number.isNaN(cores) ? null : cores;
    }
    if (platform === 'linux') {
      const out = execSync('lscpu -p=Core,Socket | grep -v "^#" | sort -u | wc -l', {
        encoding: 'utf8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'],
      });
      const cores = parseInt(out.trim(), 10);
      return Number.isNaN(cores) ? null : cores;
    }
    if (platform === 'darwin') {
      const out = execSync('sysctl -n hw.physicalcpu', { encoding: 'utf8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] });
      const cores = parseInt(out.trim(), 10);
      return Number.isNaN(cores) ? null : cores;
    }
  } catch {
    // Unavailable.
  }
  return null;
}

export async function discoverCpu(): Promise<CpuInfo> {
  const cpus = os.cpus();
  const firstCpu = cpus[0] || { model: 'Unknown Processor', speed: 0 };
  const logicalThreads = cpus.length;
  const physicalCores = detectPhysicalCores(logicalThreads);

  const modelLower = firstCpu.model.toLowerCase();
  const instructions = detectCpuInstructions();

  let recommendedOptimization: CpuInfo['recommendedOptimization'] = 'GENERIC';
  if (instructions.neon) {
    recommendedOptimization = 'NEON';
  } else if (instructions.avx512) {
    recommendedOptimization = 'AVX512';
  } else if (instructions.avx2) {
    recommendedOptimization = 'AVX2';
  } else if (instructions.avx) {
    recommendedOptimization = 'AVX';
  } else {
    recommendedOptimization = 'GENERIC';
  }

  let vendor = 'Unknown';
  if (modelLower.includes('intel')) vendor = 'Intel';
  else if (modelLower.includes('amd')) vendor = 'AMD';
  else if (modelLower.includes('apple')) vendor = 'Apple';

  return {
    vendor,
    model: firstCpu.model.trim(),
    arch: process.arch,
    physicalCores,
    logicalThreads,
    baseFrequencyGHz: firstCpu.speed > 0 ? Number((firstCpu.speed / 1000).toFixed(2)) : null,
    instructions,
    recommendedOptimization,
  };
}

/** Physical memory module details, where the OS exposes them. */
function detectMemoryModules(): { speedMHz: number | null; moduleCount: number | null; memoryType: string | null } {
  const unknown = { speedMHz: null, moduleCount: null, memoryType: null };
  try {
    if (os.platform() === 'win32') {
      const out = runPowerShell(
        'Get-CimInstance Win32_PhysicalMemory | Select-Object Speed, SMBIOSMemoryType | ConvertTo-Json -Compress',
        8000
      );
      if (!out) return unknown;
      const parsed = JSON.parse(out);
      const modules = Array.isArray(parsed) ? parsed : [parsed];
      if (modules.length === 0 || !modules[0]) return unknown;

      // SMBIOS memory type codes (DSP0134): 26=DDR4, 34=DDR5, 24=DDR3.
      const typeNames: Record<number, string> = { 24: 'DDR3', 26: 'DDR4', 34: 'DDR5' };
      const speed = Number(modules[0].Speed);
      return {
        speedMHz: Number.isNaN(speed) ? null : speed,
        moduleCount: modules.length,
        memoryType: typeNames[Number(modules[0].SMBIOSMemoryType)] || null,
      };
    }
  } catch {
    // Requires elevation on some systems — report unknown rather than guess.
  }
  return unknown;
}

export async function discoverRam(): Promise<RamInfo> {
  const totalBytes = os.totalmem();
  const freeBytes = os.freemem();
  const usedBytes = totalBytes - freeBytes;
  const systemLoadPercent = Number(((usedBytes / totalBytes) * 100).toFixed(1));

  // Dedicate part of free RAM to the AILOFlow cache, leaving headroom for the OS.
  const recommendedCacheBytes = Math.floor(freeBytes * 0.65);
  const modules = detectMemoryModules();

  return {
    totalBytes,
    freeBytes,
    usedBytes,
    systemLoadPercent,
    recommendedCacheBytes,
    speedMHz: modules.speedMHz,
    moduleCount: modules.moduleCount,
    memoryType: modules.memoryType,
  };
}

/**
 * Real VRAM size per adapter, read from the display-class registry key that
 * the driver populates.
 *
 * WMI's Win32_VideoController.AdapterRAM is a uint32 and reports 4 GB for any
 * larger card (a 12 GB RX 6750 XT shows up as 4293918720), so it cannot be
 * used. `HardwareInformation.qwMemorySize` is 64-bit and accurate.
 *
 * Returns a map keyed by the lowercased `pci\ven_xxxx&dev_xxxx` fragment so it
 * can be joined against a device's PNPDeviceID.
 */
function readVramSizesFromRegistry(): Map<string, { vramBytes: number; driverDesc: string }> {
  const result = new Map<string, { vramBytes: number; driverDesc: string }>();
  if (os.platform() !== 'win32') return result;

  try {
    const script =
      "$b='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';" +
      "Get-ChildItem $b -EA SilentlyContinue | Where-Object { $_.PSChildName -match '^[0-9]{4}$' } | ForEach-Object {" +
      '  $p = Get-ItemProperty $_.PSPath -EA SilentlyContinue;' +
      '  if ($p.MatchingDeviceId) {' +
      '    [PSCustomObject]@{ id = $p.MatchingDeviceId; vram = $p.\'HardwareInformation.qwMemorySize\'; desc = $p.DriverDesc }' +
      '  }' +
      '} | ConvertTo-Json -Compress';

    const out = runPowerShell(script);
    if (!out) return result;

    const parsed = JSON.parse(out);
    const entries = Array.isArray(parsed) ? parsed : [parsed];

    for (const entry of entries) {
      if (!entry?.id) continue;
      const vram = Number(entry.vram);
      if (!vram || Number.isNaN(vram)) continue;
      // MatchingDeviceId carries extra qualifiers (&REV_C0) and inconsistent
      // casing, so both sides are reduced to the ven/dev pair before joining.
      const key = pnpMatchKey(String(entry.id));
      if (key) result.set(key, { vramBytes: vram, driverDesc: String(entry.desc || '') });
    }
  } catch {
    // Registry unreadable — VRAM stays unknown rather than guessed.
  }

  return result;
}

/**
 * Reduce a PNPDeviceID or a registry MatchingDeviceId to the `ven_xxxx&dev_xxxx`
 * pair they have in common. WMI reports
 * `PCI\VEN_1002&DEV_73DF&SUBSYS_2419148C&REV_C0\6&179EC9A8&0&00000008` while the
 * registry stores `PCI\VEN_1002&DEV_73DF&REV_C0`, so only this fragment joins
 * reliably across the two sources.
 */
function pnpMatchKey(deviceId: string): string | null {
  const match = deviceId.toLowerCase().match(/ven_([0-9a-f]{4})&dev_([0-9a-f]{4})/);
  return match ? `ven_${match[1]}&dev_${match[2]}` : null;
}

function classifyGpuVendor(name: string): {
  vendor: GpuInfo['vendor'];
  supportedBackends: GpuInfo['supportedBackends'];
  recommendedBackend: GpuInfo['recommendedBackend'];
} {
  const lower = name.toLowerCase();

  if (/nvidia|geforce|rtx|gtx|quadro|tesla/.test(lower)) {
    return { vendor: 'NVIDIA', supportedBackends: ['CUDA', 'Vulkan', 'CPU'], recommendedBackend: 'CUDA' };
  }
  if (/amd|radeon|ati|firepro/.test(lower)) {
    // HIP/ROCm is not shipped on consumer Windows, so Vulkan is the realistic
    // default while HIP stays selectable for ROCm installations.
    return { vendor: 'AMD', supportedBackends: ['Vulkan', 'HIP', 'CPU'], recommendedBackend: 'Vulkan' };
  }
  if (/intel|arc|iris|uhd graphics|hd graphics/.test(lower)) {
    return { vendor: 'Intel', supportedBackends: ['Vulkan', 'CPU'], recommendedBackend: 'Vulkan' };
  }
  if (/apple/.test(lower)) {
    return { vendor: 'Apple', supportedBackends: ['Metal', 'CPU'], recommendedBackend: 'Metal' };
  }
  return { vendor: 'Unknown', supportedBackends: ['Vulkan', 'CPU'], recommendedBackend: 'CPU' };
}

export async function discoverGpus(): Promise<GpuInfo[]> {
  const gpus: GpuInfo[] = [];
  const platform = os.platform();

  if (platform === 'win32') {
    // Enumerate every *active* adapter. Win32_VideoController lists what is
    // actually present, unlike the registry which also keeps stale entries for
    // cards that were removed.
    let controllers: Array<{ Name?: string; DriverVersion?: string; PNPDeviceID?: string; AdapterRAM?: number }> = [];
    try {
      const output = runPowerShell(
        'Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion, PNPDeviceID, AdapterRAM | ConvertTo-Json -Compress'
      );
      const parsed = JSON.parse(output || 'null');
      controllers = Array.isArray(parsed) ? parsed : parsed ? [parsed] : [];
    } catch {
      // Enumeration failed; no adapters are reported rather than invented.
    }

    const registryVram = readVramSizesFromRegistry();

    // nvidia-smi, when present, gives exact VRAM and live counters for NVIDIA.
    const nvidiaByName = new Map<string, { totalBytes: number; freeBytes: number; temp: number | null; util: number | null }>();
    try {
      const smiOutput = execSync(
        'nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu --format=csv,noheader,nounits',
        { encoding: 'utf8', timeout: 5000, stdio: ['pipe', 'pipe', 'pipe'] }
      );
      for (const line of smiOutput.trim().split('\n')) {
        const parts = line.split(',').map((s) => s.trim());
        if (parts.length < 5) continue;
        const totalMB = parseInt(parts[1], 10);
        const freeMB = parseInt(parts[2], 10);
        if (Number.isNaN(totalMB)) continue;
        const temp = parseInt(parts[3], 10);
        const util = parseInt(parts[4], 10);
        nvidiaByName.set(parts[0].toLowerCase(), {
          totalBytes: totalMB * 1024 * 1024,
          freeBytes: Number.isNaN(freeMB) ? 0 : freeMB * 1024 * 1024,
          temp: Number.isNaN(temp) ? null : temp,
          util: Number.isNaN(util) ? null : util,
        });
      }
    } catch {
      // No NVIDIA tooling: other vendors are still fully covered below.
    }

    controllers.forEach((item, index) => {
      if (!item?.Name) return;
      const name = String(item.Name);
      const { vendor, supportedBackends, recommendedBackend } = classifyGpuVendor(name);

      let vramTotalBytes: number | null = null;
      let vramFreeBytes: number | null = null;
      let vramSource: GpuInfo['vramSource'] = 'unknown';
      let utilizationPercent: number | null = null;
      let temperatureC: number | null = null;

      const smi = nvidiaByName.get(name.toLowerCase());
      if (smi) {
        vramTotalBytes = smi.totalBytes;
        vramFreeBytes = smi.freeBytes;
        vramSource = 'nvidia-smi';
        utilizationPercent = smi.util;
        temperatureC = smi.temp;
      } else {
        // Works for AMD, Intel and NVIDIA alike: the 64-bit registry value.
        const key = item.PNPDeviceID ? pnpMatchKey(String(item.PNPDeviceID)) : null;
        const fromRegistry = key ? registryVram.get(key) : undefined;
        if (fromRegistry) {
          vramTotalBytes = fromRegistry.vramBytes;
          vramSource = 'registry';
        } else {
          // Last resort: AdapterRAM, trustworthy only below the uint32 ceiling.
          const rawVram = item.AdapterRAM ? Number(item.AdapterRAM) : 0;
          if (rawVram > 0 && rawVram < 4 * 1024 * 1024 * 1024) {
            vramTotalBytes = rawVram;
            vramSource = 'wmi';
          }
        }
      }

      gpus.push({
        id: `gpu-${index}`,
        vendor,
        model: name,
        vramTotalBytes,
        vramFreeBytes,
        vramSource,
        driverVersion: item.DriverVersion || undefined,
        supportedBackends,
        recommendedBackend,
        utilizationPercent,
        temperatureC,
        pnpDeviceId: item.PNPDeviceID ? String(item.PNPDeviceID) : undefined,
      });
    });

    // Prefer the adapter with the most VRAM as the primary compute device.
    gpus.sort((a, b) => (b.vramTotalBytes || 0) - (a.vramTotalBytes || 0));
  } else if (platform === 'darwin') {
    // Apple Silicon shares system memory with the GPU; that is a real figure,
    // not an estimate, so it is reported as unified memory.
    gpus.push({
      id: 'gpu-0',
      vendor: 'Apple',
      model: 'Apple Unified GPU',
      vramTotalBytes: os.totalmem(),
      vramFreeBytes: os.freemem(),
      vramSource: 'unified-memory',
      supportedBackends: ['Metal', 'CPU'],
      recommendedBackend: 'Metal',
      temperatureC: null,
      utilizationPercent: null,
    });
  }

  // An empty list means "no discrete GPU detected" — the caller falls back to
  // CPU inference rather than being handed a fictional device.
  return gpus;
}

/**
 * Class-based bandwidth estimate from the drive's bus type. Explicitly flagged
 * as unmeasured; a real benchmark replaces it.
 */
function estimateProfileFromBusType(type: StorageDriveInfo['type']): StoragePerformanceProfile {
  const table: Record<string, Omit<StoragePerformanceProfile, 'measured' | 'healthStatus'>> = {
    NVMe: { seqReadMBps: 3500, seqWriteMBps: 3000, randReadMBps: 450, randWriteMBps: 400, latencyUs: 100, iops: 400000 },
    SATA: { seqReadMBps: 550, seqWriteMBps: 500, randReadMBps: 90, randWriteMBps: 85, latencyUs: 200, iops: 90000 },
    HDD: { seqReadMBps: 160, seqWriteMBps: 140, randReadMBps: 2, randWriteMBps: 2, latencyUs: 8000, iops: 120 },
    USB: { seqReadMBps: 300, seqWriteMBps: 250, randReadMBps: 30, randWriteMBps: 25, latencyUs: 600, iops: 20000 },
    UNKNOWN: { seqReadMBps: 500, seqWriteMBps: 450, randReadMBps: 60, randWriteMBps: 55, latencyUs: 300, iops: 50000 },
  };

  return { ...(table[type] || table.UNKNOWN), healthStatus: 'UNKNOWN', measured: false };
}

export async function discoverStorage(): Promise<StorageDriveInfo[]> {
  const drives: StorageDriveInfo[] = [];
  const platform = os.platform();

  if (platform === 'win32') {
    try {
      // 1. Get physical disk details
      // Physical disks keyed by their DeviceId, which is the same number
      // Get-Partition reports as DiskNumber.
      const disksById = new Map<number, { FriendlyName: string; MediaType: string; BusType: string; Size: number }>();
      try {
        const pdOutput = runPowerShell(
          'Get-PhysicalDisk | Select-Object DeviceId, FriendlyName, MediaType, BusType, Size | ConvertTo-Json -Compress'
        );
        const pdParsed = JSON.parse(pdOutput || 'null');
        for (const disk of Array.isArray(pdParsed) ? pdParsed : pdParsed ? [pdParsed] : []) {
          const id = Number(disk.DeviceId);
          if (!Number.isNaN(id)) disksById.set(id, disk);
        }
      } catch {
        // Storage subsystem query unavailable; bus type stays unknown.
      }

      // Drive letter -> disk number. This join is the whole point: matching the
      // Nth volume to the Nth physical disk by array position is arbitrary and
      // silently mislabels every drive when the orders differ.
      const diskNumberByLetter = new Map<string, number>();
      try {
        const partOutput = runPowerShell(
          'Get-Partition | Where-Object { $_.DriveLetter } | Select-Object DriveLetter, DiskNumber | ConvertTo-Json -Compress'
        );
        const partParsed = JSON.parse(partOutput || 'null');
        for (const part of Array.isArray(partParsed) ? partParsed : partParsed ? [partParsed] : []) {
          if (!part?.DriveLetter) continue;
          diskNumberByLetter.set(String(part.DriveLetter).toUpperCase(), Number(part.DiskNumber));
        }
      } catch {
        // Without the partition map we simply do not claim a bus type.
      }

      // 2. Get volume details
      const output = runPowerShell(
        'Get-Volume | Select-Object -Property DriveLetter, FileSystemLabel, FileSystem, Size, SizeRemaining | ConvertTo-Json -Compress'
      );
      const parsed = JSON.parse(output || 'null');
      const items = Array.isArray(parsed) ? parsed : parsed ? [parsed] : [];

      items.forEach((item, index) => {
        if (!item || !item.DriveLetter) return;
        const letter = `${item.DriveLetter}:/`;
        const totalSize = Number(item.Size) || 0;
        const freeSize = Number(item.SizeRemaining) || 0;

        // A volume with no size is a card reader or optical drive with nothing
        // in it. Reporting it as storage would put a phantom device in the
        // fabric and make every benchmark fail against it.
        if (totalSize === 0) return;

        // Resolve the actual physical disk behind this letter.
        const diskNumber = diskNumberByLetter.get(String(item.DriveLetter).toUpperCase());
        const physDisk = diskNumber === undefined ? undefined : disksById.get(diskNumber);
        const busTypeUpper = physDisk?.BusType?.toUpperCase() || '';
        const mediaTypeUpper = physDisk?.MediaType?.toUpperCase() || '';

        // Only claim a bus type the OS actually reported.
        let driveType: StorageDriveInfo['type'] = 'UNKNOWN';
        if (busTypeUpper.includes('NVME')) {
          driveType = 'NVMe';
        } else if (busTypeUpper.includes('USB')) {
          driveType = 'USB';
        } else if (busTypeUpper.includes('SATA') || busTypeUpper.includes('ATA')) {
          driveType = mediaTypeUpper === 'HDD' ? 'HDD' : 'SATA';
        } else if (mediaTypeUpper === 'HDD') {
          driveType = 'HDD';
        } else if (mediaTypeUpper === 'SSD') {
          driveType = 'SATA';
        }

        drives.push({
          id: `drive-${index}`,
          devicePath: diskNumber === undefined ? letter : `\\\\.\\PhysicalDrive${diskNumber}`,
          mountPoint: letter,
          label: physDisk?.FriendlyName || item.FileSystemLabel || `Disk ${item.DriveLetter}`,
          type: driveType,
          totalSizeBytes: totalSize,
          freeSizeBytes: freeSize,
          filesystem: item.FileSystem || 'NTFS',
          performanceProfile: estimateProfileFromBusType(driveType),
        });
      });
    } catch {
      // Enumeration failed; return whatever was collected rather than inventing a disk.
    }
  } else {
    // POSIX: enumerate mounted filesystems with df.
    try {
      const output = execSync('df -k -P', { encoding: 'utf8', timeout: 5000 });
      const lines = output.trim().split('\n').slice(1);

      lines.forEach((line, index) => {
        const cols = line.trim().split(/\s+/);
        if (cols.length < 6) return;
        const [device, totalKb, , availableKb] = cols;
        const mountPoint = cols.slice(5).join(' ');
        if (!device.startsWith('/dev/')) return;

        drives.push({
          id: `drive-${index}`,
          devicePath: device,
          mountPoint,
          label: device,
          type: 'UNKNOWN',
          totalSizeBytes: Number(totalKb) * 1024,
          freeSizeBytes: Number(availableKb) * 1024,
          filesystem: 'unknown',
          performanceProfile: estimateProfileFromBusType('UNKNOWN'),
        });
      });
    } catch {
      // df unavailable — report nothing rather than a fictional disk.
    }
  }

  // Replace estimates with real benchmark results previously measured here.
  const cachedBenchmarks = loadConfig().storageBenchmarks;
  for (const drive of drives) {
    const cached = cachedBenchmarks[drive.mountPoint];
    if (!cached) continue;
    drive.performanceProfile = {
      seqReadMBps: cached.seqReadMBps,
      seqWriteMBps: cached.seqWriteMBps,
      randReadMBps: cached.randReadMBps,
      randWriteMBps: cached.randWriteMBps,
      latencyUs: cached.latencyUs,
      iops: cached.iops,
      healthStatus: 'UNKNOWN',
      measured: true,
      measuredAt: cached.measuredAt,
      cacheInfluenced: cached.cacheInfluenced === true,
    };
  }

  return drives;
}

/**
 * Discovery is expensive: it shells out to PowerShell several times and takes
 * seconds — occasionally tens of seconds — to come back. The hardware it
 * describes does not change between two requests a second apart, so a request
 * that only needs to know what the machine is gets the last answer.
 *
 * This is what made the tuning panel unusable: every slider movement rebuilt
 * the plan, every plan rediscovered the machine, and the interface sat waiting
 * on PowerShell to re-enumerate disks that had not moved.
 */
const PROFILE_TTL_MS = 60_000;
let cachedProfile: { profile: ComputerProfile; at: number } | null = null;
let profileInFlight: Promise<ComputerProfile> | null = null;

/** Drop the cached profile, for when something is known to have changed. */
export function invalidateProfileCache(): void {
  cachedProfile = null;
}

export async function discoverComputerProfile(options?: { fresh?: boolean }): Promise<ComputerProfile> {
  if (options?.fresh) cachedProfile = null;

  const fresh = cachedProfile && Date.now() - cachedProfile.at < PROFILE_TTL_MS;
  if (fresh && cachedProfile) return cachedProfile.profile;

  // Concurrent callers share one discovery rather than each starting their own
  // storm of PowerShell processes.
  if (profileInFlight) return profileInFlight;

  profileInFlight = buildComputerProfile()
    .then((profile) => {
      cachedProfile = { profile, at: Date.now() };
      return profile;
    })
    .finally(() => {
      profileInFlight = null;
    });

  return profileInFlight;
}

async function buildComputerProfile(): Promise<ComputerProfile> {
  const [cpu, ram, gpus, storageDrives] = await Promise.all([
    discoverCpu(),
    discoverRam(),
    discoverGpus(),
    discoverStorage(),
  ]);

  const primaryGpu = gpus[0];
  const configuredBackend = loadConfig().backendOverride;
  const selectedBackend = configuredBackend || (primaryGpu ? primaryGpu.recommendedBackend : 'CPU');

  const totalStorageBytes = storageDrives.reduce((acc, d) => acc + d.totalSizeBytes, 0);

  // Only benchmarked drives contribute — an unmeasured fabric has no known bandwidth.
  const measuredDrives = storageDrives.filter((d) => d.performanceProfile?.measured);
  const measuredStorageReadBandwidthMBps = measuredDrives.length
    ? Number(measuredDrives.reduce((acc, d) => acc + (d.performanceProfile?.seqReadMBps || 0), 0).toFixed(1))
    : null;

  return {
    timestamp: new Date().toISOString(),
    os: {
      platform: os.platform(),
      release: os.release(),
      arch: os.arch(),
    },
    cpu,
    ram,
    gpus,
    storageDrives,
    selectedBackend,
    totalStorageBytes,
    measuredStorageReadBandwidthMBps,
  };
}
