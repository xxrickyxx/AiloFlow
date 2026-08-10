/**
 * Typed client for the local AILOFlow API.
 *
 * Every value the UI renders comes through here. There is no local mock data:
 * when the API is down the UI must show that it is down.
 */

export const API_BASE = '/v1';

// --- Shared shapes (mirror the server types) --------------------------------

export interface CpuInfo {
  vendor: string;
  model: string;
  arch: string;
  physicalCores: number | null;
  logicalThreads: number;
  baseFrequencyGHz: number | null;
  instructions: Record<string, boolean>;
  recommendedOptimization: string;
}

export interface RamInfo {
  totalBytes: number;
  freeBytes: number;
  usedBytes: number;
  systemLoadPercent: number;
  recommendedCacheBytes: number;
  speedMHz: number | null;
  moduleCount: number | null;
  memoryType: string | null;
}

export interface GpuInfo {
  id: string;
  vendor: string;
  model: string;
  vramTotalBytes: number | null;
  vramFreeBytes: number | null;
  vramSource: string;
  driverVersion?: string;
  supportedBackends: string[];
  recommendedBackend: string;
  temperatureC?: number | null;
  utilizationPercent?: number | null;
  pnpDeviceId?: string;
}

export interface GpuLiveSample {
  adapterKey: string | null;
  utilizationPercent: number | null;
  engineBreakdown: Record<string, number> | null;
  memoryUsedBytes: number | null;
  memoryTotalBytes: number | null;
  temperatureC: number | null;
  source: string;
}

export interface StoragePerformanceProfile {
  seqReadMBps: number;
  seqWriteMBps: number;
  randReadMBps: number;
  randWriteMBps: number;
  latencyUs: number;
  iops: number;
  healthStatus: string;
  measured: boolean;
  measuredAt?: string;
}

export interface StorageDriveInfo {
  id: string;
  devicePath: string;
  mountPoint: string;
  label: string;
  type: string;
  totalSizeBytes: number;
  freeSizeBytes: number;
  filesystem: string;
  performanceProfile?: StoragePerformanceProfile;
}

export interface ComputerProfile {
  timestamp: string;
  os: { platform: string; release: string; arch: string };
  cpu: CpuInfo;
  ram: RamInfo;
  gpus: GpuInfo[];
  storageDrives: StorageDriveInfo[];
  selectedBackend: string;
  totalStorageBytes: number;
  measuredStorageReadBandwidthMBps: number | null;
}

export interface OptimizerConfig {
  ramCacheBytes: number;
  vramCacheBytes: number;
  threadCount: number;
  backend: string;
  prefetchDepthLayers: number;
  ioQueueDepth: number;
  shardingStrategy: string;
  ioChunkSizeBytes: number;
  basedOnMeasuredStorage: boolean;
}

export interface TelemetrySnapshot {
  timestamp: string;
  cpuUsagePercent: number;
  ramUsagePercent: number;
  ramUsedBytes: number;
  ramTotalBytes: number;
  gpuUsagePercent: number | null;
  vramUsagePercent: number | null;
  vramUsedBytes: number | null;
  vramTotalBytes: number | null;
  gpuTemperatureC: number | null;
  gpus: GpuLiveSample[];
  gpuUnavailableReason: string | null;
  generation: {
    active: boolean;
    modelId: string | null;
    backendId: string | null;
    tokensPerSecond: number | null;
    promptTokensPerSecond: number | null;
    firstTokenLatencyMs: number | null;
    completionTokens: number | null;
    promptTokens: number | null;
    measuredAt: string | null;
  };
  storage: {
    bandwidthMBps: number | null;
    iops: number | null;
    totalBytesRead: number;
    queueDepth: number;
  };
  cache: {
    hitRatePercent: number | null;
    hits: number;
    misses: number;
    vramUsedBytes: number;
    ramUsedBytes: number;
  } | null;
  prefetch: { hitRatePercent: number | null; triggered: number; throttled: boolean } | null;
  bottleneck: {
    type: string;
    title: string;
    description: string;
    requestedBandwidthMBps?: number;
    availableBandwidthMBps?: number;
  };
}

export interface DiscoveredModel {
  id: string;
  source: 'ollama' | 'gguf' | 'sflow';
  displayName: string;
  filePath: string | null;
  fileSizeBytes: number | null;
  runnableWith: string[];
  inspectable: boolean;
  modifiedAt: string | null;
  /** Ordered parts when the model is published as a split file set. */
  splitParts: string[] | null;
  /** false when a split set is still missing parts. */
  complete: boolean;
}

export interface ModelInspection {
  id: string;
  source: string;
  filePath: string;
  fileSizeBytes: number;
  architecture: string;
  modelName: string;
  parameterCountBillions: number;
  quantization: string;
  blockCount: number;
  contextLength: number;
  embeddingLength: number;
  headCount: number;
  headCountKv: number;
  tensorCount: number;
  totalTensorDataBytes: number;
  estimatedRamRequiredBytes: number;
  estimatedVramRequiredBytes: number;
  estimatedStorageRequiredBytes: number;
  estimatedKvCacheBytes: number;
}

export interface BackendAvailability {
  id: string;
  name: string;
  available: boolean;
  reason: string | null;
  detail?: string;
}

export interface LoadedModelState {
  model: DiscoveredModel;
  backendId: string;
  backendName: string;
  loadedAt: string;
  loadDurationMs: number;
}

export interface EngineCandidate {
  variant: string;
  assetName: string;
  downloadUrl: string;
  sizeBytes: number;
  rationale: string;
  companionAssets: Array<{ name: string; url: string; sizeBytes: number }>;
}

export interface InstalledEngine {
  variant: string;
  release: string;
  serverPath: string;
  installedAt: string;
  version: string | null;
}

export type SizeClass = 'gigante' | 'grande' | 'medio' | 'piccolo';

export interface CatalogEntry {
  id: string;
  name: string;
  publisher: string;
  totalParamsB: number;
  activeParamsB: number;
  architecture: 'dense' | 'moe';
  contextLength: number;
  ggufRepos: string[];
  license: string;
  notes: string;
  sizeClass: SizeClass;
  tags: string[];
}

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
  splitParts?: string[];
  totalSizeBytes: number;
  quantization: string | null;
}

export interface TierPlacement {
  tier: 'VRAM' | 'RAM' | 'STORAGE';
  residentBytes: number;
  bytesPerToken: number;
  bandwidthMBps: number | null;
  secondsPerToken: number | null;
}

export interface PerformanceEstimate {
  model: { name: string; totalParamsB: number; activeParamsB: number; quantization: string; architecture: string };
  totalWeightBytes: number;
  activeBytesPerToken: number;
  kvCacheBytes: number;
  fitsInMemory: boolean;
  tiers: Array<{ tier: string; capacityBytes: number; bandwidthMBps: number | null; source: string; note: string }>;
  placement: TierPlacement[];
  sequentialTokensPerSecond: number | null;
  overlappedTokensPerSecond: number | null;
  bottleneckTier: string | null;
  bandwidthForTenTokensPerSecondMBps: number | null;
  confidence: 'measured' | 'partial' | 'speculative';
  assumptions: string[];
  warnings: string[];
}

export interface EstimateResponse {
  estimate: PerformanceEstimate;
  projections: Array<{ storageBandwidthMBps: number; estimate: PerformanceEstimate }>;
  weightBytesAtQuant: number;
}

export interface AiloConfig {
  apiPort: number;
  modelDirectories: string[];
  llamaServerPath: string | null;
  ollamaBaseUrl: string;
  ollamaEnabled: boolean;
  activeModelId: string | null;
  backendOverride: string | null;
  gpuLayers: number | null;
  contextLength: number | null;
  storageBenchmarks: Record<string, unknown>;
}

export interface BenchmarkResult {
  driveId: string;
  mountPoint: string;
  seqReadMBps: number;
  seqWriteMBps: number;
  randReadMBps: number;
  randWriteMBps: number;
  latencyUs: number;
  iops: number;
  benchmarkDurationMs: number;
  timestamp: string;
}

export interface LayerSweepResult {
  modelName: string;
  totalLayers: number;
  layersCompleted: number;
  bytesReadFromStorage: number;
  totalBytesRequested: number;
  durationMs: number;
  effectiveBandwidthMBps: number;
  averageLayerLatencyMs: number;
  cacheHits: number;
  cacheMisses: number;
  cacheHitRatePercent: number;
  prefetchTriggered: number;
  prefetchThrottled: boolean;
  drivesUsed: string[];
  prefetchDepth: number;
  bytesPerTokenStreamed: number;
  errors: string[];
}

// --- Transport ---------------------------------------------------------------

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
  });

  const text = await res.text();
  let body: any = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(`Risposta non valida da ${path}: ${text.slice(0, 200)}`);
  }

  if (!res.ok) throw new ApiError(body?.error || `HTTP ${res.status} su ${path}`);
  return body as T;
}

export const api = {
  health: () => request<{ status: string; version: string }>('/health'),

  system: () => request<{ profile: ComputerProfile; optimizer: OptimizerConfig }>('/system'),

  storage: () =>
    request<{
      totalStorageBytes: number;
      measuredReadBandwidthMBps: number | null;
      drives: StorageDriveInfo[];
    }>('/storage'),

  benchmarkStorage: (mountPoint?: string) =>
    request<{ results: BenchmarkResult[]; failures: Array<{ mountPoint: string; error: string }>; drives: StorageDriveInfo[] }>(
      '/storage/benchmark',
      { method: 'POST', body: JSON.stringify({ mountPoint }) }
    ),

  metrics: () => request<TelemetrySnapshot>('/metrics'),

  models: async (): Promise<DiscoveredModel[]> => {
    const body = await request<{ data: Array<{ ailoflow: DiscoveredModel }> }>('/models');
    return body.data.map((d) => d.ailoflow);
  },

  inspectModel: (id: string) =>
    request<ModelInspection>(`/models/inspect?id=${encodeURIComponent(id)}`),

  loadModel: (id: string) =>
    request<LoadedModelState>('/models/load', { method: 'POST', body: JSON.stringify({ id }) }),

  unloadModel: () => request<{ unloaded: boolean }>('/models/unload', { method: 'POST' }),

  backends: () =>
    request<{ backends: BackendAvailability[]; loaded: LoadedModelState | null }>('/backends'),

  config: () => request<AiloConfig>('/config'),

  updateConfig: (patch: Partial<AiloConfig>) =>
    request<AiloConfig>('/config', { method: 'PATCH', body: JSON.stringify(patch) }),

  lastSweep: () => request<{ lastSweep: LayerSweepResult | null }>('/pipeline/last'),

  engine: () =>
    request<{ installed: InstalledEngine[]; active: InstalledEngine | null; llamaServerPath: string | null }>('/engine'),

  engineCandidates: () =>
    request<{ release: string; candidates: EngineCandidate[] }>('/engine/candidates'),

  catalog: () =>
    request<{ entries: CatalogEntry[]; downloadDirectory: string; freeBytes: number | null }>('/catalog'),

  searchRepos: (query: string) =>
    request<{ results: HfModelSummary[] }>(`/catalog/search?q=${encodeURIComponent(query)}`),

  repoFiles: (repoId: string) =>
    request<{ repoId: string; files: HfFile[] }>(`/catalog/files?repo=${encodeURIComponent(repoId)}`),

  setDownloadDirectory: (directory: string) =>
    request<{ downloadDirectory: string; freeBytes: number | null }>('/catalog/directory', {
      method: 'POST',
      body: JSON.stringify({ directory }),
    }),

  estimate: (body: Record<string, unknown>) =>
    request<EstimateResponse>('/estimate', { method: 'POST', body: JSON.stringify(body) }),

  benchmarkMemory: () =>
    request<{ bandwidthMBps: number; copyMBps: number }>('/benchmark/memory', { method: 'POST' }),
};

// --- Streaming helpers -------------------------------------------------------

export interface ChatStreamCallbacks {
  onToken: (text: string) => void;
  /** Chain-of-thought from reasoning models, kept out of the answer. */
  onReasoning?: (text: string) => void;
  onMetrics?: (metrics: TelemetrySnapshot['generation']) => void;
  onError?: (message: string) => void;
}

export interface ChatRequest {
  model?: string;
  messages: Array<{ role: string; content: string }>;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  seed?: number;
  stop?: string[];
  repeat_penalty?: number;
  context_length?: number;
}

/** Stream a chat completion, forwarding real tokens and real timing metrics. */
export async function streamChat(
  body: ChatRequest,
  callbacks: ChatStreamCallbacks,
  signal?: AbortSignal
): Promise<void> {
  const res = await fetch(`${API_BASE}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, stream: true }),
    signal,
  });

  if (!res.ok) {
    const text = await res.text();
    let message = `HTTP ${res.status}`;
    try {
      message = JSON.parse(text).error || message;
    } catch { /* non-JSON error body */ }
    throw new ApiError(message);
  }
  if (!res.body) throw new ApiError('Nessun corpo di risposta dal server.');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event:')) {
        currentEvent = line.slice(6).trim();
        continue;
      }
      if (!line.startsWith('data:')) continue;

      const data = line.slice(5).trim();
      if (data === '[DONE]') return;

      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }

      if (currentEvent === 'error') {
        callbacks.onError?.(parsed.message || 'Errore sconosciuto');
        currentEvent = 'message';
        continue;
      }
      if (currentEvent === 'metrics') {
        callbacks.onMetrics?.(parsed);
        currentEvent = 'message';
        continue;
      }

      const delta = parsed.choices?.[0]?.delta;
      if (delta?.reasoning_content) callbacks.onReasoning?.(delta.reasoning_content);
      if (delta?.content) callbacks.onToken(delta.content);
    }
  }
}

export interface SseHandlers {
  [event: string]: (data: any) => void;
}

/** Consume a POST-initiated SSE endpoint (sharding, pipeline sweep). */
export async function streamPost(path: string, body: unknown, handlers: SseHandlers): Promise<void> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok || !res.body) throw new ApiError(`HTTP ${res.status} su ${path}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let currentEvent = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event:')) {
        currentEvent = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        try {
          handlers[currentEvent]?.(JSON.parse(line.slice(5).trim()));
        } catch { /* partial frame */ }
      }
    }
  }
}
