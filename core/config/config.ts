import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * Persistent AILOFlow configuration.
 *
 * Everything here is user-controlled state that must survive restarts:
 * where models live, which inference engine binary to use, which model is
 * currently selected. Nothing in this file invents values — when a setting
 * is unknown it stays `null` so the UI can say "not configured" instead of
 * displaying a plausible-looking default.
 */
export interface AiloConfig {
  /** TCP port for the local OpenAI-compatible API. */
  apiPort: number;
  /** Directories scanned for *.gguf / *.sflow model files. */
  modelDirectories: string[];
  /** Absolute path to a llama.cpp `llama-server` binary, or null if not set. */
  llamaServerPath: string | null;
  /** Base URL of a local Ollama daemon. */
  ollamaBaseUrl: string;
  /** Whether the Ollama backend may be used. */
  ollamaEnabled: boolean;
  /** Model id currently selected for generation (backend-qualified). */
  activeModelId: string | null;
  /** Manual backend override; null = automatic selection. */
  backendOverride: 'CUDA' | 'Vulkan' | 'HIP' | 'Metal' | 'CPU' | null;
  /** Number of GPU layers to offload for llama.cpp; null = auto. */
  gpuLayers: number | null;
  /**
   * Context window the engine is started with, in tokens. null = pick
   * automatically from the model's own maximum, capped so the KV cache stays
   * affordable. IDE assistants send very large system prompts, so the
   * llama.cpp default of 4096 is far too small in practice.
   */
  contextLength: number | null;
  /**
   * Runtime tuning overrides. Every field is null by default, meaning the
   * runtime decides from the detected hardware; setting one hands that
   * decision to the user.
   */
  tuning: TuningOverridesRecord;
  /** Cached storage benchmark results keyed by drive mount point. */
  storageBenchmarks: Record<string, StoredBenchmark>;
  /** The llama.cpp build AILOFlow installed and manages itself. */
  installedEngine: InstalledEngineRecord | null;
  /** Where downloaded models are stored. */
  downloadDirectory: string | null;
  /** Measured RAM copy bandwidth, used by the performance estimator. */
  ramBandwidthMBps: number | null;
  /**
   * Bind address for the API. Stays on loopback unless the user deliberately
   * opens it up, because doing so exposes local models to the network.
   */
  apiHost: string;
}

/**
 * Mirrors TuningOverrides. Declared here rather than imported so the config
 * module stays free of dependencies on the tuning logic it merely stores.
 */
export interface TuningOverridesRecord {
  expertsPerToken: number | null;
  gpuLayers: number | null;
  cpuMoeLayers: number | null;
  contextLength: number | null;
  kvCacheType: 'f16' | 'q8_0' | 'q4_0' | null;
  threads: number | null;
  batchSize: number | null;
  ubatchSize: number | null;
  flashAttention: 'auto' | 'on' | 'off' | null;
  loadMode: 'mmap' | 'mlock' | 'mmap+mlock' | null;
}

export function emptyTuningOverrides(): TuningOverridesRecord {
  return {
    expertsPerToken: null,
    gpuLayers: null,
    cpuMoeLayers: null,
    contextLength: null,
    kvCacheType: null,
    threads: null,
    batchSize: null,
    ubatchSize: null,
    flashAttention: null,
    loadMode: null,
  };
}

export interface InstalledEngineRecord {
  variant: string;
  release: string;
  serverPath: string;
  installedAt: string;
  version: string | null;
}

export interface StoredBenchmark {
  seqReadMBps: number;
  seqWriteMBps: number;
  randReadMBps: number;
  randWriteMBps: number;
  latencyUs: number;
  iops: number;
  measuredAt: string;
  cacheInfluenced?: boolean;
}

function defaultModelDirectories(): string[] {
  const dirs: string[] = [];
  const home = os.homedir();

  // Ollama keeps real GGUF blobs here; they are readable model files.
  const ollamaModels = process.env.OLLAMA_MODELS || path.join(home, '.ollama', 'models');
  if (fs.existsSync(ollamaModels)) dirs.push(ollamaModels);

  const localModels = path.join(process.cwd(), 'models');
  if (fs.existsSync(localModels)) dirs.push(localModels);

  return dirs;
}

export function getConfigDirectory(): string {
  if (process.env.AILOFLOW_CONFIG_DIR) return process.env.AILOFLOW_CONFIG_DIR;

  const platform = os.platform();
  if (platform === 'win32') {
    return path.join(process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming'), 'AiloFlow');
  }
  if (platform === 'darwin') {
    return path.join(os.homedir(), 'Library', 'Application Support', 'AiloFlow');
  }
  return path.join(process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config'), 'ailoflow');
}

export function getConfigPath(): string {
  return path.join(getConfigDirectory(), 'config.json');
}

export function createDefaultConfig(): AiloConfig {
  return {
    // 11434 belongs to Ollama — AILOFlow must not squat on it.
    apiPort: 11500,
    modelDirectories: defaultModelDirectories(),
    llamaServerPath: null,
    ollamaBaseUrl: 'http://127.0.0.1:11434',
    ollamaEnabled: true,
    activeModelId: null,
    backendOverride: null,
    gpuLayers: null,
    contextLength: null,
    tuning: emptyTuningOverrides(),
    storageBenchmarks: {},
    installedEngine: null,
    downloadDirectory: null,
    ramBandwidthMBps: null,
    apiHost: '127.0.0.1',
  };
}

let cached: AiloConfig | null = null;

export function loadConfig(): AiloConfig {
  if (cached) return cached;

  const configPath = getConfigPath();
  const defaults = createDefaultConfig();

  try {
    const raw = fs.readFileSync(configPath, 'utf8');
    const parsed = JSON.parse(raw) as Partial<AiloConfig>;
    // Merge tuning field by field: a config written before a knob existed must
    // gain it as "auto" rather than as undefined.
    cached = {
      ...defaults,
      ...parsed,
      tuning: { ...emptyTuningOverrides(), ...(parsed.tuning || {}) },
    };
  } catch {
    // No config yet (or unreadable) — start from defaults and persist them.
    cached = defaults;
    try {
      saveConfig(cached);
    } catch {
      // Read-only environment: keep working purely in memory.
    }
  }

  return cached;
}

export function saveConfig(config: AiloConfig): void {
  const dir = getConfigDirectory();
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(getConfigPath(), JSON.stringify(config, null, 2), 'utf8');
  cached = config;
}

export function updateConfig(patch: Partial<AiloConfig>): AiloConfig {
  const next = { ...loadConfig(), ...patch };
  saveConfig(next);
  return next;
}

/** Drop the in-process cache so the next read hits disk (used by tests). */
export function resetConfigCache(): void {
  cached = null;
}
