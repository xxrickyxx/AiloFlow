import fs from 'fs';
import os from 'os';
import path from 'path';
import { loadConfig } from '../config/config.js';
import { GgufMetadata, parseGgufHeader } from './gguf_parser.js';
import { SFlowContainer } from '../../formats/sflow/sflow_format.js';

export type ModelSource = 'ollama' | 'gguf' | 'sflow';

export interface DiscoveredModel {
  /** Stable, backend-qualified identifier, e.g. `ollama:qwen2.5:0.5b`. */
  id: string;
  /**
   * Short name for humans and IDE config files. A full id carries an absolute
   * path, which is unusable in a settings field, so every model also answers to
   * this alias.
   */
  alias: string;
  source: ModelSource;
  displayName: string;
  /** Real path of the underlying model file when one is known on disk. */
  filePath: string | null;
  /** Real byte size from the filesystem, or null when unknown. */
  fileSizeBytes: number | null;
  /** Engines that can actually execute this model right now. */
  runnableWith: Array<'ollama' | 'llama.cpp' | 'ailo-hierarchical'>;
  /** True when the GGUF header can be parsed for detailed metadata. */
  inspectable: boolean;
  modifiedAt: string | null;
  /**
   * For models published as an ordered set of files, every part in order.
   * null for ordinary single-file models.
   */
  splitParts: string[] | null;
  /** false when a split set is missing parts and therefore cannot be loaded. */
  complete: boolean;
}

export interface ModelInspection {
  id: string;
  source: ModelSource;
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
  /** KV cache bytes for the full context at f16, computed from real dims. */
  estimatedKvCacheBytes: number;
  /** Number of files the model is split across; 1 for an ordinary model. */
  splitPartCount: number;
}

const GGUF_MAGIC_LE = 0x46554747;

// ---------------------------------------------------------------------------
// Ollama discovery — real daemon, real blob files
// ---------------------------------------------------------------------------

interface OllamaTag {
  name: string;
  size?: number;
  modified_at?: string;
  digest?: string;
}

function ollamaModelsRoot(): string {
  return process.env.OLLAMA_MODELS || path.join(os.homedir(), '.ollama', 'models');
}

/**
 * Resolve the real GGUF blob backing an Ollama tag by reading the manifest
 * that Ollama itself wrote. Returns null when the manifest cannot be located,
 * which simply means we can't offer header inspection for that tag.
 */
export function resolveOllamaBlobPath(tag: string): string | null {
  const root = ollamaModelsRoot();
  const manifestsRoot = path.join(root, 'manifests');
  if (!fs.existsSync(manifestsRoot)) return null;

  const [namePart, tagPart = 'latest'] = tag.split(':');
  const segments = namePart.split('/');
  // Ollama stores manifests as <registry>/<namespace>/<name>/<tag>; the
  // registry and namespace are implicit for short names.
  const candidates: string[] = [];
  const registries = safeReadDir(manifestsRoot);

  for (const registry of registries) {
    if (segments.length === 1) {
      candidates.push(path.join(manifestsRoot, registry, 'library', segments[0], tagPart));
      for (const ns of safeReadDir(path.join(manifestsRoot, registry))) {
        candidates.push(path.join(manifestsRoot, registry, ns, segments[0], tagPart));
      }
    } else {
      candidates.push(path.join(manifestsRoot, registry, ...segments, tagPart));
    }
  }

  for (const candidate of candidates) {
    if (!fs.existsSync(candidate) || !fs.statSync(candidate).isFile()) continue;
    try {
      const manifest = JSON.parse(fs.readFileSync(candidate, 'utf8'));
      const modelLayer = (manifest.layers || []).find(
        (l: any) => l.mediaType === 'application/vnd.ollama.image.model'
      );
      if (!modelLayer?.digest) continue;
      const blobFile = String(modelLayer.digest).replace(':', '-');
      const blobPath = path.join(root, 'blobs', blobFile);
      if (fs.existsSync(blobPath)) return blobPath;
    } catch {
      // Malformed manifest — skip it rather than guessing.
    }
  }

  return null;
}

function safeReadDir(dir: string): string[] {
  try {
    return fs.readdirSync(dir);
  } catch {
    return [];
  }
}

function isLlamaServerAvailable(): boolean {
  const configured = loadConfig().llamaServerPath;
  return configured !== null && fs.existsSync(configured);
}

/**
 * Enumerate the models Ollama has already downloaded.
 *
 * The blobs are ordinary GGUF files, so when AILOFlow has its own llama.cpp
 * build these models stay usable even with the Ollama daemon stopped — the
 * daemon is a convenience, not a dependency.
 */
export async function discoverOllamaModels(baseUrl: string, timeoutMs = 2000): Promise<DiscoveredModel[]> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const llamaAvailable = isLlamaServerAvailable();

  let tags: OllamaTag[] = [];
  let daemonUp = false;

  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, '')}/api/tags`, { signal: controller.signal });
    if (res.ok) {
      const body = (await res.json()) as { models?: OllamaTag[] };
      tags = body.models || [];
      daemonUp = true;
    }
  } catch {
    // Daemon down: fall back to reading its manifests straight off disk.
  } finally {
    clearTimeout(timer);
  }

  if (!daemonUp) {
    tags = listOllamaTagsFromDisk().map((name) => ({ name }));
  }

  return tags.map((t) => {
    const blobPath = resolveOllamaBlobPath(t.name);
    const usableAsGguf = blobPath !== null && isGgufFile(blobPath);

    const runnableWith: Array<'ollama' | 'llama.cpp'> = [];
    if (daemonUp) runnableWith.push('ollama');
    if (usableAsGguf && llamaAvailable) runnableWith.push('llama.cpp');

    return {
      id: `ollama:${t.name}`,
      alias: t.name,
      source: 'ollama' as const,
      displayName: t.name,
      filePath: blobPath,
      fileSizeBytes: typeof t.size === 'number' ? t.size : blobPath ? statSize(blobPath) : null,
      runnableWith,
      inspectable: usableAsGguf,
      modifiedAt: t.modified_at || null,
      splitParts: null,
      complete: true,
    };
  });
}

/** Read tag names from Ollama's manifest tree when the daemon is not running. */
function listOllamaTagsFromDisk(): string[] {
  const manifestsRoot = path.join(ollamaModelsRoot(), 'manifests');
  if (!fs.existsSync(manifestsRoot)) return [];

  const tags: string[] = [];

  // Layout: manifests/<registry>/<namespace>/<name>/<tag>
  for (const registry of safeReadDir(manifestsRoot)) {
    const registryDir = path.join(manifestsRoot, registry);
    for (const namespace of safeReadDir(registryDir)) {
      const namespaceDir = path.join(registryDir, namespace);
      for (const name of safeReadDir(namespaceDir)) {
        const nameDir = path.join(namespaceDir, name);
        let entries: string[];
        try {
          if (!fs.statSync(nameDir).isDirectory()) continue;
          entries = fs.readdirSync(nameDir);
        } catch {
          continue;
        }
        for (const tag of entries) {
          // `library` is Ollama's implicit namespace for short names.
          const prefix = namespace === 'library' ? '' : `${namespace}/`;
          tags.push(`${prefix}${name}:${tag}`);
        }
      }
    }
  }

  return tags;
}

export async function isOllamaReachable(baseUrl: string, timeoutMs = 1500): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${baseUrl.replace(/\/$/, '')}/api/tags`, { signal: controller.signal });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

// ---------------------------------------------------------------------------
// Filesystem discovery — real .gguf and .sflow files
// ---------------------------------------------------------------------------

function statSize(filePath: string): number | null {
  try {
    return fs.statSync(filePath).size;
  } catch {
    return null;
  }
}

export function isGgufFile(filePath: string): boolean {
  let fd: number | null = null;
  try {
    fd = fs.openSync(filePath, 'r');
    const buf = Buffer.alloc(4);
    const read = fs.readSync(fd, buf, 0, 4, 0);
    return read === 4 && buf.readUInt32LE(0) === GGUF_MAGIC_LE;
  } catch {
    return false;
  } finally {
    if (fd !== null) {
      try { fs.closeSync(fd); } catch { /* already closed */ }
    }
  }
}

function walkForModels(dir: string, depth: number, out: string[]): void {
  if (depth < 0) return;
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      // Ollama blobs are enumerated through the manifest path instead.
      if (entry.name === 'blobs' || entry.name === 'manifests') continue;
      walkForModels(full, depth - 1, out);
    } else if (/\.(gguf|sflow)$/i.test(entry.name)) {
      out.push(full);
    }
  }
}

/** `model-00002-of-00005.gguf` → base name, part index and part count. */
const SPLIT_FILE_PATTERN = /^(.*)-(\d{5})-of-(\d{5})\.gguf$/i;

export function discoverFileModels(directories: string[], maxDepth = 3): DiscoveredModel[] {
  const found: string[] = [];
  for (const dir of directories) {
    if (fs.existsSync(dir)) walkForModels(dir, maxDepth, found);
  }

  const config = loadConfig();
  const llamaAvailable = config.llamaServerPath !== null && fs.existsSync(config.llamaServerPath);

  // A model larger than 50 GB is published as an ordered set of files, because
  // that is Hugging Face's per-file limit. Those files are one model: listing
  // them individually would offer five unloadable "models" instead of one.
  const splitGroups = new Map<string, { parts: string[]; expected: number }>();
  const standalone: string[] = [];

  for (const filePath of found) {
    const match = path.basename(filePath).match(SPLIT_FILE_PATTERN);
    if (!match) {
      standalone.push(filePath);
      continue;
    }
    const key = path.join(path.dirname(filePath), match[1]);
    const group = splitGroups.get(key) || { parts: [], expected: Number(match[3]) };
    group.parts.push(filePath);
    splitGroups.set(key, group);
  }

  const models: DiscoveredModel[] = [];

  for (const filePath of standalone) {
    const isSflow = /\.sflow$/i.test(filePath);
    let stats: fs.Stats | null = null;
    try { stats = fs.statSync(filePath); } catch { /* vanished mid-scan */ }

    models.push({
      id: `${isSflow ? 'sflow' : 'gguf'}:${filePath}`,
      alias: path.basename(filePath).replace(/\.(gguf|sflow)$/i, ''),
      source: (isSflow ? 'sflow' : 'gguf') as ModelSource,
      displayName: path.basename(filePath),
      filePath,
      fileSizeBytes: stats ? stats.size : null,
      // .sflow is an AILOFlow container executed via AILOFlow Active Weight Engine; GGUF supports both llama.cpp and AILOFlow engine.
      runnableWith: isSflow
        ? (['ailo-hierarchical'] as Array<'ailo-hierarchical'>)
        : llamaAvailable
        ? (['llama.cpp', 'ailo-hierarchical'] as Array<'llama.cpp' | 'ailo-hierarchical'>)
        : (['ailo-hierarchical'] as Array<'ailo-hierarchical'>),
      inspectable: isSflow ? true : isGgufFile(filePath),
      modifiedAt: stats ? stats.mtime.toISOString() : null,
      splitParts: null,
      complete: true,
    });
  }

  for (const [key, group] of splitGroups.entries()) {
    group.parts.sort();
    const first = group.parts[0];
    const totalBytes = group.parts.reduce((sum, p) => {
      try { return sum + fs.statSync(p).size; } catch { return sum; }
    }, 0);

    // Only the complete set is runnable: llama.cpp opens part 1 and expects to
    // find every sibling. An incomplete set must say so rather than fail later.
    const complete = group.parts.length === group.expected;
    let stats: fs.Stats | null = null;
    try { stats = fs.statSync(first); } catch { /* vanished mid-scan */ }

    models.push({
      id: `gguf:${first}`,
      alias: path.basename(key),
      source: 'gguf',
      displayName: `${path.basename(key)} (${group.parts.length}/${group.expected} parts)`,
      filePath: first,
      fileSizeBytes: totalBytes,
      runnableWith: complete && llamaAvailable ? (['llama.cpp'] as Array<'llama.cpp'>) : [],
      inspectable: isGgufFile(first),
      modifiedAt: stats ? stats.mtime.toISOString() : null,
      splitParts: group.parts,
      complete,
    });
  }

  return models;
}

// ---------------------------------------------------------------------------
// Combined registry
// ---------------------------------------------------------------------------

export async function discoverAllModels(): Promise<DiscoveredModel[]> {
  const config = loadConfig();
  const fileModels = discoverFileModels(config.modelDirectories);
  const ollamaModels = config.ollamaEnabled
    ? await discoverOllamaModels(config.ollamaBaseUrl)
    : [];

  const all = [...ollamaModels, ...fileModels];
  // Deduplicate on id, keeping the first (Ollama entries win — they are runnable).
  const seen = new Set<string>();
  return all.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
}

/**
 * Resolve a model by full id or by its short alias.
 *
 * IDE plugins and API clients configure a model by name, so both spellings must
 * work; the alias match is case-insensitive because config files rarely get the
 * capitalisation right.
 */
export async function findModelById(id: string): Promise<DiscoveredModel | null> {
  const models = await discoverAllModels();
  const exact = models.find((m) => m.id === id);
  if (exact) return exact;

  const lowered = id.toLowerCase();
  return models.find((m) => m.alias.toLowerCase() === lowered) || null;
}

/**
 * Compute KV cache size from the model's real dimensions:
 * 2 (K and V) × layers × context × kv_heads × head_dim × 2 bytes (f16).
 */
function estimateKvCacheBytes(meta: GgufMetadata): number {
  const headDim = meta.headCount > 0 ? meta.embeddingLength / meta.headCount : 0;
  const kvDim = meta.headCountKv * headDim;
  return Math.round(2 * meta.blockCount * meta.contextLength * kvDim * 2);
}

/**
 * Aggregate a split model's real figures.
 *
 * Architecture, layer count and context length are global and identical in
 * every part, so they come from the first. Tensor counts and byte totals are
 * per-file and must be summed — only then do parameter count and memory
 * requirement describe the actual model.
 */
function inspectSplitModel(model: DiscoveredModel, first: GgufMetadata): ModelInspection {
  const parts = model.splitParts!;
  let tensorCount = 0;
  let totalTensorDataBytes = 0;
  let totalElements = 0;
  let fileSizeBytes = 0;
  const problems: string[] = [];

  for (const part of parts) {
    try {
      const meta = parseGgufHeader(part);
      tensorCount += meta.tensorCount;
      totalTensorDataBytes += meta.totalTensorDataBytes;
      totalElements += meta.tensors.reduce((sum, t) => sum + Number(t.numElements), 0);
      fileSizeBytes += meta.fileSizeBytes;
    } catch (err) {
      // A part we cannot read makes the totals wrong; say so rather than
      // silently reporting a fraction of the model.
      problems.push(`${path.basename(part)}: ${(err as Error).message}`);
    }
  }

  if (problems.length > 0) {
    throw new Error(
      `Cannot read every part of this split model, so its size cannot be reported:\n  - ${problems.join('\n  - ')}`
    );
  }

  const headDim = first.headCount > 0 ? first.embeddingLength / first.headCount : 0;
  const kvDim = first.headCountKv * headDim;

  return {
    id: model.id,
    source: model.source,
    filePath: model.filePath!,
    fileSizeBytes,
    architecture: first.architecture,
    modelName: first.modelName,
    parameterCountBillions: Number((totalElements / 1e9).toFixed(2)),
    quantization: first.quantization,
    blockCount: first.blockCount,
    contextLength: first.contextLength,
    embeddingLength: first.embeddingLength,
    headCount: first.headCount,
    headCountKv: first.headCountKv,
    tensorCount,
    totalTensorDataBytes,
    estimatedRamRequiredBytes: Math.ceil(totalTensorDataBytes * 1.15),
    estimatedVramRequiredBytes: Math.ceil(totalTensorDataBytes * 0.25),
    estimatedStorageRequiredBytes: fileSizeBytes,
    estimatedKvCacheBytes: Math.round(2 * first.blockCount * first.contextLength * kvDim * 2),
    splitPartCount: parts.length,
  };
}

export async function inspectModel(id: string): Promise<ModelInspection> {
  const model = await findModelById(id);
  if (!model) throw new Error(`Model not found in registry: ${id}`);
  if (!model.filePath) {
    throw new Error(`Model "${model.displayName}" has no readable file on disk, so it cannot be inspected.`);
  }

  if (model.source === 'sflow') {
    const container = SFlowContainer.load(model.filePath);
    const m = container.manifest;
    return {
      id: model.id,
      source: 'sflow',
      filePath: model.filePath,
      fileSizeBytes: model.fileSizeBytes || 0,
      architecture: m.architecture,
      modelName: m.modelName,
      parameterCountBillions: m.parameterCountBillions,
      quantization: m.quantization,
      blockCount: m.blockCount,
      contextLength: m.contextLength,
      embeddingLength: 0,
      headCount: 0,
      headCountKv: 0,
      tensorCount: m.tensorMap.length,
      totalTensorDataBytes: m.totalSizeBytes,
      estimatedRamRequiredBytes: Math.ceil(m.totalSizeBytes * 1.15),
      estimatedVramRequiredBytes: Math.ceil(m.totalSizeBytes * 0.25),
      estimatedStorageRequiredBytes: m.totalSizeBytes,
      estimatedKvCacheBytes: 0,
      splitPartCount: 1,
    };
  }

  const meta = parseGgufHeader(model.filePath);

  // A split model's first file declares only its own share of the tensors.
  // Reporting those as the model's would understate a 357B model as a 79B one,
  // and its memory requirement by the same factor, so every part is read.
  if (model.splitParts && model.splitParts.length > 1) {
    return inspectSplitModel(model, meta);
  }

  return {
    id: model.id,
    source: model.source,
    filePath: model.filePath,
    fileSizeBytes: meta.fileSizeBytes,
    architecture: meta.architecture,
    modelName: meta.modelName,
    parameterCountBillions: meta.parameterCountBillions,
    quantization: meta.quantization,
    blockCount: meta.blockCount,
    contextLength: meta.contextLength,
    embeddingLength: meta.embeddingLength,
    headCount: meta.headCount,
    headCountKv: meta.headCountKv,
    tensorCount: meta.tensorCount,
    totalTensorDataBytes: meta.totalTensorDataBytes,
    estimatedRamRequiredBytes: meta.estimatedRamRequiredBytes,
    estimatedVramRequiredBytes: meta.estimatedVramRequiredBytes,
    estimatedStorageRequiredBytes: meta.estimatedStorageRequiredBytes,
    estimatedKvCacheBytes: estimateKvCacheBytes(meta),
    splitPartCount: 1,
  };
}
