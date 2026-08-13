// Must come first: libuv fixes its threadpool size on first use, and every
// storage read depends on how many slots it has.
import '../../core/runtime/threadpool.js';
import express, { Request, Response } from 'express';
import cors from 'cors';
import path from 'path';
import { discoverComputerProfile } from '../../core/hardware/discovery.js';
import { calculateOptimalConfiguration } from '../../core/hardware/optimizer.js';
import { benchmarkStorageDrive } from '../../core/benchmark/storage_benchmark.js';
import { TelemetryMonitor } from '../../core/telemetry/telemetry.js';
import { loadConfig, updateConfig } from '../../core/config/config.js';
import { discoverAllModels, inspectModel } from '../../core/model/model_registry.js';
import { parseGgufModel } from '../../core/model/gguf_parser.js';
import { createShardedSFlowModel } from '../../core/sharding/shard_manager.js';
import { SFlowContainer } from '../../formats/sflow/sflow_format.js';
import { GenerationOptions } from '../../inference/base.js';
import { AiloStreamingPipeline, LayerSweepResult } from '../../inference/custom_stream/stream_runner.js';
import { InferenceRegistry } from '../../inference/registry.js';
import { ComputerProfile } from '../../core/hardware/types.js';
import { installEngine, listEngineCandidates, listInstalledEngines } from '../../core/engine/engine_installer.js';
import { killOrphanedEngines } from '../../inference/llama_cpp/llama_cpp_backend.js';
import { MODEL_CATALOG, estimateWeightBytes, findCatalogEntry } from '../../core/models/catalog.js';
import {
  getDownloadDirectory,
  getDownloadDirectoryFreeBytes,
  listRepoGgufFiles,
  searchGgufRepos,
  setDownloadDirectory,
} from '../../core/models/downloader.js';
import { downloadManager } from '../../core/models/download_manager.js';
import { benchmarkMemoryBandwidth, getStoredRamBandwidth } from '../../core/benchmark/memory_benchmark.js';
import { estimatePerformance, ModelSpec, projectWithStorageBandwidth } from '../../core/estimator/performance_model.js';
import { suggestPresets } from '../../core/tuning/runtime_tuning.js';
import { registerOllamaCompatRoutes } from './ollama_compat.js';

const app = express();
app.use(cors({ origin: [/^http:\/\/localhost:\d+$/, /^http:\/\/127\.0\.0\.1:\d+$/] }));
app.use(express.json({ limit: '2mb' }));

const telemetry = new TelemetryMonitor();
telemetry.start(1000);

const inference = new InferenceRegistry();

let profile: ComputerProfile = await discoverComputerProfile();
telemetry.configureGpus(profile);

// Pick up any transfer that was still running when this process last stopped.
const resumedDownloads = downloadManager.resumeInterrupted();

// An engine orphaned by a previous run still holds the GPU and the model's
// memory, which makes the next load fail for no visible reason.
const reapedEngines = killOrphanedEngines();
let lastSweep: LayerSweepResult | null = null;
let lastGenerationAt: string | null = null;
let generationActive = false;

import { AiloHierarchicalBackend } from '../../inference/ailo_hierarchical/ailo_hierarchical_backend.js';

function telemetrySources() {
  const backend = inference.getActiveBackend();
  let ioStats;
  let prefetchStats;
  let cacheMetrics;

  if (backend instanceof AiloHierarchicalBackend) {
    ioStats = backend.getIoStats();
    prefetchStats = backend.getPrefetchStats();
    cacheMetrics = backend.getCacheMetrics();
  }

  return {
    ioStats,
    prefetchStats,
    cacheMetrics,
    lastGeneration: inference.getLastMetrics(),
    lastGenerationAt,
    generationActive,
    bytesPerTokenStreamed: lastSweep ? lastSweep.bytesPerTokenStreamed : null,
  };
}

function fail(res: Response, status: number, error: unknown): void {
  res.status(status).json({ error: error instanceof Error ? error.message : String(error) });
}

// ---------------------------------------------------------------------------
// System & configuration
// ---------------------------------------------------------------------------

app.get('/v1/health', (_req, res) => {
  res.json({ status: 'ok', service: 'ailoflow', version: '1.0.0' });
});

app.get('/v1/system', async (_req, res) => {
  try {
    profile = await discoverComputerProfile();
    telemetry.configureGpus(profile);
    // Live counters know how much VRAM is in use; the optimizer needs that to
    // budget a cache on non-NVIDIA adapters.
    telemetry.enrichProfileWithLiveGpu(profile);
    res.json({ profile, optimizer: calculateOptimalConfiguration(profile) });
  } catch (err) {
    fail(res, 500, err);
  }
});

app.get('/v1/config', (_req, res) => {
  res.json(loadConfig());
});

app.patch('/v1/config', (req, res) => {
  try {
    const next = updateConfig(req.body || {});
    inference.reconfigure();
    res.json(next);
  } catch (err) {
    fail(res, 400, err);
  }
});

app.get('/v1/backends', async (_req, res) => {
  try {
    res.json({ backends: await inference.listAvailability(), loaded: inference.getState() });
  } catch (err) {
    fail(res, 500, err);
  }
});

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

app.get('/v1/storage', (_req, res) => {
  res.json({
    totalStorageBytes: profile.totalStorageBytes,
    measuredReadBandwidthMBps: profile.measuredStorageReadBandwidthMBps,
    drives: profile.storageDrives,
  });
});

/** Run a real, non-destructive benchmark. `mountPoint` limits it to one drive. */
app.post('/v1/storage/benchmark', async (req, res) => {
  const { mountPoint } = req.body || {};
  const targets = mountPoint
    ? profile.storageDrives.filter((d) => d.mountPoint === mountPoint)
    : profile.storageDrives;

  if (targets.length === 0) {
    return fail(res, 404, new Error(`No drive matches mount point "${mountPoint}".`));
  }

  const results = [];
  const failures = [];
  for (const drive of targets) {
    try {
      results.push(await benchmarkStorageDrive(drive, { sampleSizeBytes: 32 * 1024 * 1024, iterations: 3 }));
    } catch (err) {
      failures.push({ mountPoint: drive.mountPoint, error: (err as Error).message });
    }
  }

  // The benchmark is exactly the thing that changes what discovery reports, so
  // this is one of the few places worth paying for a fresh sweep.
  profile = await discoverComputerProfile({ fresh: true });
  res.json({ results, failures, drives: profile.storageDrives });
});

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

app.get('/v1/models', async (_req, res) => {
  try {
    const models = await discoverAllModels();
    // OpenAI-compatible shape, with AILOFlow detail attached.
    res.json({
      object: 'list',
      data: models.map((m) => ({
        id: m.id,
        object: 'model',
        created: m.modifiedAt ? Math.floor(new Date(m.modifiedAt).getTime() / 1000) : null,
        owned_by: m.source,
        ailoflow: m,
      })),
    });
  } catch (err) {
    fail(res, 500, err);
  }
});

app.get('/v1/models/inspect', async (req, res) => {
  const id = String(req.query.id || '');
  if (!id) return fail(res, 400, new Error('Query parameter "id" is required.'));
  try {
    res.json(await inspectModel(id));
  } catch (err) {
    fail(res, 400, err);
  }
});

app.post('/v1/models/load', async (req, res) => {
  const { id, backendId } = req.body || {};
  if (!id) return fail(res, 400, new Error('Body field "id" is required.'));
  try {
    const state = await inference.loadModel(String(id), backendId ? String(backendId) : undefined);
    updateConfig({ activeModelId: String(id) });
    res.json(state);
  } catch (err) {
    fail(res, 400, err);
  }
});

/**
 * The launch plan for a model: what the runtime would choose, what the user
 * has overridden, and what it costs. Available before loading, so a decision
 * that takes minutes to act on can be made with the numbers in view.
 */
app.get('/v1/tuning', async (req, res) => {
  const id = String(req.query.id || '');
  if (!id) return fail(res, 400, new Error('Query parameter "id" is required.'));
  try {
    const plan = await inference.planForId(id);
    res.json({
      plan,
      overrides: loadConfig().tuning,
      presets: plan ? suggestPresets(plan) : [],
    });
  } catch (err) {
    fail(res, 400, err);
  }
});

/** Set or clear overrides. A null field hands the decision back to the runtime. */
app.patch('/v1/tuning', async (req, res) => {
  try {
    const current = loadConfig().tuning;
    const next = { ...current, ...(req.body?.overrides || {}) };
    updateConfig({ tuning: next });

    const id = req.body?.id ? String(req.body.id) : null;
    res.json({ overrides: next, plan: id ? await inference.planForId(id) : null });
  } catch (err) {
    fail(res, 400, err);
  }
});

/** Force-stop the generation in flight, freeing the GPU without a restart. */
app.post('/v1/generation/stop', (_req, res) => {
  res.json({ stopped: inference.stopGeneration() });
});

app.post('/v1/models/unload', async (_req, res) => {
  await inference.unload();
  updateConfig({ activeModelId: null });
  res.json({ unloaded: true });
});

/** Shard a real GGUF into a .sflow container, streaming progress over SSE. */
app.post('/v1/models/shard', async (req, res) => {
  const { modelPath, outputDirectory, distributeAcrossDrives } = req.body || {};
  if (!modelPath) return fail(res, 400, new Error('Body field "modelPath" is required.'));

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const meta = parseGgufModel(String(modelPath));
    const outDir = outputDirectory || path.join(process.cwd(), 'sflow-models', meta.filename.replace(/\.gguf$/i, ''));

    send('start', {
      model: meta.filename,
      layers: meta.blockCount,
      tensors: meta.tensorCount,
      sizeBytes: meta.fileSizeBytes,
      outputDirectory: outDir,
    });

    const container = await createShardedSFlowModel(
      meta,
      profile.storageDrives,
      outDir,
      (p) => send('progress', p),
      { distributeAcrossDrives: distributeAcrossDrives === true }
    );

    send('done', {
      sflowPath: container.sflowPath,
      shards: container.manifest.shards.length,
      drivesUsed: Array.from(new Set(container.manifest.shards.map((s) => s.targetMountPoint))),
      validation: container.validateShards(outDir),
    });
  } catch (err) {
    send('error', { message: (err as Error).message });
  } finally {
    res.end();
  }
});

/** Validate a .sflow container against the shard files actually on disk. */
app.get('/v1/models/sflow/validate', (req, res) => {
  const sflowPath = String(req.query.path || '');
  if (!sflowPath) return fail(res, 400, new Error('Query parameter "path" is required.'));
  try {
    const container = SFlowContainer.load(sflowPath);
    res.json({ manifest: container.manifest, validation: container.validateShards() });
  } catch (err) {
    fail(res, 400, err);
  }
});

// ---------------------------------------------------------------------------
// Storage pipeline benchmark (the real .sflow layer sweep)
// ---------------------------------------------------------------------------

app.post('/v1/pipeline/sweep', async (req, res) => {
  const { sflowPath } = req.body || {};
  if (!sflowPath) return fail(res, 400, new Error('Body field "sflowPath" is required.'));

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  const pipeline = new AiloStreamingPipeline(profile);
  try {
    await pipeline.load(String(sflowPath));
    send('start', { sflowPath });
    const result = await pipeline.runLayerSweep((p) => send('progress', p));
    lastSweep = result;
    send('done', result);
  } catch (err) {
    send('error', { message: (err as Error).message });
  } finally {
    pipeline.dispose();
    res.end();
  }
});

app.get('/v1/pipeline/last', (_req, res) => {
  res.json({ lastSweep });
});

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

app.get('/v1/metrics', (_req, res) => {
  res.json(telemetry.generateSnapshot(profile, telemetrySources()));
});

/** Server-sent metric stream so the GUI does not have to poll. */
app.get('/v1/metrics/stream', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders?.();

  const push = () => {
    res.write(`data: ${JSON.stringify(telemetry.generateSnapshot(profile, telemetrySources()))}\n\n`);
  };

  push();
  const handle = setInterval(push, 1000);
  req.on('close', () => clearInterval(handle));
});

// ---------------------------------------------------------------------------
// Inference — OpenAI compatible, backed by a real engine
// ---------------------------------------------------------------------------

interface ChatBody {
  messages?: Array<{ role: string; content: string }>;
  model?: string;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  max_tokens?: number;
  seed?: number;
  stop?: string[];
  repeat_penalty?: number;
  context_length?: number;
}

function buildOptions(body: ChatBody): GenerationOptions {
  const messages = body.messages || [];
  const systemPrompt = messages.find((m) => m.role === 'system')?.content;
  const conversation = messages.filter((m) => m.role === 'user' || m.role === 'assistant');
  const last = conversation[conversation.length - 1];

  if (!last || last.role !== 'user') {
    throw new Error('The last message must have role "user".');
  }

  return {
    prompt: last.content,
    systemPrompt,
    history: conversation.slice(0, -1) as Array<{ role: 'user' | 'assistant'; content: string }>,
    temperature: body.temperature,
    topP: body.top_p,
    topK: body.top_k,
    maxTokens: body.max_tokens,
    seed: body.seed,
    stopSequences: body.stop,
    repetitionPenalty: body.repeat_penalty,
    contextLength: body.context_length,
  };
}

async function handleChatCompletion(body: ChatBody, res: Response): Promise<void> {
  generationActive = true;
  try {
    // Honour an explicit model switch before generating; the request may name
    // the model by full id or by its short alias.
    const current = inference.getState()?.model;
    if (body.model && (!current || (current.id !== body.model && current.alias !== body.model))) {
      await inference.loadModel(body.model);
    }

    const backend = inference.getActiveBackend();
    const state = inference.getState();
    if (!backend || !state) {
      throw new Error(
        'No model is loaded. Select a model in the Models tab (or POST /v1/models/load) before generating.'
      );
    }

    const options = buildOptions(body);
    const stream = body.stream !== false;

    // Stop the engine when the client goes away, rather than letting it run on
    // and hold the GPU generating output nobody is reading.
    const abort = new AbortController();
    res.on('close', () => abort.abort());
    options.signal = abort.signal;

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders?.();

      const id = `chatcmpl-${Date.now()}`;
      const result = await backend.generateStream(options, (token) => {
        if (!token.token && !token.isFinished) return;
        const delta = token.isFinished
          ? {}
          : token.kind === 'reasoning'
            ? { reasoning_content: token.token }
            : { content: token.token };

        res.write(
          `data: ${JSON.stringify({
            id,
            object: 'chat.completion.chunk',
            created: Math.floor(Date.now() / 1000),
            model: state.model.id,
            choices: [{ index: 0, delta, finish_reason: token.isFinished ? 'stop' : null }],
          })}\n\n`
        );
      });

      inference.recordMetrics(result.metrics);
      lastGenerationAt = new Date().toISOString();
      res.write(`event: metrics\ndata: ${JSON.stringify(result.metrics)}\n\n`);
      res.write('data: [DONE]\n\n');
      res.end();
    } else {
      const result = await backend.generateStream(options, () => { /* buffered */ });
      inference.recordMetrics(result.metrics);
      lastGenerationAt = new Date().toISOString();

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: state.model.id,
        choices: [
          {
            index: 0,
            message: {
              role: 'assistant',
              content: result.text,
              ...(result.reasoning ? { reasoning_content: result.reasoning } : {}),
            },
            finish_reason: 'stop',
          },
        ],
        usage: {
          prompt_tokens: result.metrics.promptTokens,
          completion_tokens: result.metrics.completionTokens,
          total_tokens:
            result.metrics.promptTokens !== null && result.metrics.completionTokens !== null
              ? result.metrics.promptTokens + result.metrics.completionTokens
              : null,
        },
        ailoflow_metrics: result.metrics,
      });
    }
  } catch (err) {
    if ((err as Error).name === 'AbortError' || res.destroyed) return;
    if (res.headersSent) {
      res.write(`event: error\ndata: ${JSON.stringify({ message: (err as Error).message })}\n\n`);
      res.end();
    } else {
      fail(res, 400, err);
    }
  } finally {
    generationActive = false;
  }
}

app.post('/v1/chat/completions', (req: Request, res: Response) => {
  void handleChatCompletion((req.body || {}) as ChatBody, res);
});

app.post('/v1/completions', (req: Request, res: Response) => {
  const { prompt, ...rest } = req.body || {};
  if (typeof prompt !== 'string') {
    return fail(res, 400, new Error('Body field "prompt" is required.'));
  }
  void handleChatCompletion({ ...rest, messages: [{ role: 'user', content: prompt }] }, res);
});

// ---------------------------------------------------------------------------
// Engine management — AILOFlow installs and owns its own llama.cpp build
// ---------------------------------------------------------------------------

app.get('/v1/engine', async (_req, res) => {
  const config = loadConfig();
  res.json({
    installed: listInstalledEngines(),
    active: config.installedEngine,
    llamaServerPath: config.llamaServerPath,
  });
});

app.get('/v1/engine/candidates', async (_req, res) => {
  try {
    res.json(await listEngineCandidates(profile));
  } catch (err) {
    fail(res, 502, err);
  }
});

/** Download and install an engine build, streaming progress over SSE. */
app.post('/v1/engine/install', async (req, res) => {
  const { variant, release } = req.body || {};

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.flushHeaders?.();
  const send = (event: string, data: unknown) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  try {
    const { release: latest, candidates } = await listEngineCandidates(profile);
    const candidate = variant ? candidates.find((c) => c.variant === variant) : candidates[0];
    if (!candidate) throw new Error(`Nessuna build disponibile per la variante "${variant}".`);

    send('start', { candidate, release: release || latest });
    const installed = await installEngine(candidate, release || latest, (p) => send('progress', p));
    inference.reconfigure();
    send('done', installed);
  } catch (err) {
    send('error', { message: (err as Error).message });
  } finally {
    res.end();
  }
});

// ---------------------------------------------------------------------------
// Catalog and downloads
// ---------------------------------------------------------------------------

app.get('/v1/catalog', (_req, res) => {
  res.json({
    entries: MODEL_CATALOG,
    downloadDirectory: getDownloadDirectory(),
    freeBytes: getDownloadDirectoryFreeBytes(),
  });
});

app.get('/v1/catalog/search', async (req, res) => {
  const query = String(req.query.q || '');
  if (!query) return fail(res, 400, new Error('Query parameter "q" is required.'));
  try {
    res.json({ results: await searchGgufRepos(query) });
  } catch (err) {
    fail(res, 502, err);
  }
});

app.get('/v1/catalog/files', async (req, res) => {
  const repoId = String(req.query.repo || '');
  if (!repoId) return fail(res, 400, new Error('Query parameter "repo" is required.'));
  try {
    res.json({ repoId, files: await listRepoGgufFiles(repoId) });
  } catch (err) {
    fail(res, 502, err);
  }
});

app.post('/v1/catalog/directory', (req, res) => {
  const { directory } = req.body || {};
  if (!directory) return fail(res, 400, new Error('Body field "directory" is required.'));
  try {
    setDownloadDirectory(String(directory));
    res.json({ downloadDirectory: getDownloadDirectory(), freeBytes: getDownloadDirectoryFreeBytes() });
  } catch (err) {
    fail(res, 400, err);
  }
});

/**
 * Queue a download and return its job id immediately.
 *
 * The transfer runs in the process, not in the request: navigating away or
 * closing the browser must never abandon a partially fetched 300 GB model.
 * Clients read progress from GET /v1/downloads whenever they come back.
 */
app.post('/v1/downloads', (req, res) => {
  const { repoId, file } = req.body || {};

  try {
    if (!repoId || !file?.path) throw new Error('Body fields "repoId" and "file" are required.');

    const free = getDownloadDirectoryFreeBytes();
    if (free !== null && file.totalSizeBytes > free) {
      throw new Error(
        `Servono ${(file.totalSizeBytes / 1e9).toFixed(1)} GB ma ne restano ` +
          `${(free / 1e9).toFixed(1)} GB in ${getDownloadDirectory()}.`
      );
    }

    const job = downloadManager.start(String(repoId), file);
    res.json({ job, directory: getDownloadDirectory() });
  } catch (err) {
    fail(res, 400, err);
  }
});

app.get('/v1/downloads', (_req, res) => {
  res.json({ jobs: downloadManager.list(), directory: getDownloadDirectory() });
});

app.post('/v1/downloads/:id/cancel', (req, res) => {
  const cancelled = downloadManager.cancel(req.params.id);
  if (!cancelled) return fail(res, 404, new Error('Nessun download attivo con questo id.'));
  res.json({ cancelled: true });
});

app.post('/v1/downloads/clear', (_req, res) => {
  res.json({ removed: downloadManager.clearFinished() });
});

// ---------------------------------------------------------------------------
// Performance estimation
// ---------------------------------------------------------------------------

app.post('/v1/estimate', (req, res) => {
  const body = req.body || {};
  try {
    let spec: ModelSpec;

    if (body.catalogId) {
      const entry = findCatalogEntry(String(body.catalogId));
      if (!entry) throw new Error(`Modello "${body.catalogId}" non presente nel catalogo.`);
      spec = {
        name: entry.name,
        totalParamsB: entry.totalParamsB,
        activeParamsB: entry.activeParamsB,
        quantization: body.quantization || 'Q4_K_M',
        layers: body.layers || 64,
        contextLength: entry.contextLength,
        architecture: entry.architecture,
      };
    } else {
      if (!body.totalParamsB) throw new Error('Serve "catalogId" oppure "totalParamsB".');
      spec = {
        name: body.name || `${body.totalParamsB}B`,
        totalParamsB: Number(body.totalParamsB),
        activeParamsB: Number(body.activeParamsB || body.totalParamsB),
        quantization: body.quantization || 'Q4_K_M',
        layers: Number(body.layers || 64),
        contextLength: Number(body.contextLength || 32768),
        architecture: body.activeParamsB && body.activeParamsB < body.totalParamsB ? 'moe' : 'dense',
      };
    }

    // Only benchmarked drives contribute, and a page-cache-influenced result is
    // flagged so the estimate says so out loud.
    const measuredDrives = profile.storageDrives.filter((d) => d.performanceProfile?.measured);
    const storageBandwidthMBps = measuredDrives.length
      ? measuredDrives.reduce((sum, d) => sum + (d.performanceProfile?.seqReadMBps || 0), 0)
      : null;
    const cacheInfluenced = measuredDrives.some((d) => d.performanceProfile?.cacheInfluenced);

    const inputs = {
      profile,
      ramBandwidthMBps: getStoredRamBandwidth(),
      storageBandwidthMBps: body.storageBandwidthMBps ?? storageBandwidthMBps,
      storageBandwidthIsCacheInfluenced: body.storageBandwidthMBps ? false : cacheInfluenced,
      contextUsed: body.contextUsed,
    };

    const estimate = estimatePerformance(spec, inputs);

    // What-if projections: how the same model behaves on faster storage.
    const projections = (body.projectStorageMBps || [1600, 7000, 14000, 28000]).map((mbps: number) => ({
      storageBandwidthMBps: mbps,
      estimate: projectWithStorageBandwidth(spec, inputs, mbps),
    }));

    res.json({ estimate, projections, weightBytesAtQuant: estimateWeightBytes(spec.totalParamsB, spec.quantization) });
  } catch (err) {
    fail(res, 400, err);
  }
});

app.post('/v1/benchmark/memory', (_req, res) => {
  try {
    res.json(benchmarkMemoryBandwidth());
  } catch (err) {
    fail(res, 500, err);
  }
});

// ---------------------------------------------------------------------------
// Ollama-compatible API so IDE integrations can target AILOFlow
// ---------------------------------------------------------------------------

registerOllamaCompatRoutes(app, {
  inference,
  listModels: discoverAllModels,
  onGenerationStart: () => { generationActive = true; lastGenerationAt = new Date().toISOString(); },
  onGenerationEnd: () => { generationActive = false; },
  version: '1.0.0',
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

function startServer(port: number, attemptsLeft = 10): void {
  const host = loadConfig().apiHost || '127.0.0.1';
  const server = app.listen(port, host, () => {
    console.log(`AILOFlow API listening on http://${host}:${port}`);
    console.log(`  OpenAI-compatible: http://${host}:${port}/v1`);
    console.log(`  Ollama-compatible: http://${host}:${port}/api  (for IDE plugins)`);
    console.log(`  CPU:     ${profile.cpu.model} (${profile.cpu.logicalThreads} threads, ${profile.cpu.recommendedOptimization})`);
    console.log(`  GPUs:    ${profile.gpus.length ? profile.gpus.map((g) => g.model).join(', ') : 'none detected'}`);
    console.log(`  Drives:  ${profile.storageDrives.length}`);
    console.log(
      `  Storage bandwidth: ${
        profile.measuredStorageReadBandwidthMBps === null
          ? 'not benchmarked yet'
          : `${profile.measuredStorageReadBandwidthMBps} MB/s measured`
      }`
    );
    if (resumedDownloads > 0) {
      console.log(`  Resumed ${resumedDownloads} interrupted download(s)`);
    }
    if (reapedEngines > 0) {
      console.log(`  Cleaned up ${reapedEngines} orphaned engine process(es)`);
    }
  });

  server.on('error', (err: NodeJS.ErrnoException) => {
    if (err.code === 'EADDRINUSE' && attemptsLeft > 0) {
      console.log(`Port ${port} busy, trying ${port + 1}...`);
      startServer(port + 1, attemptsLeft - 1);
    } else {
      console.error('Server error:', err.message);
      process.exit(1);
    }
  });
}

const configuredPort = Number(process.env.PORT) || loadConfig().apiPort;
startServer(configuredPort);
