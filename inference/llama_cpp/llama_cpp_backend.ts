import fs from 'fs';
import net from 'net';
import os from 'os';
import path from 'path';
import { ChildProcess, spawn } from 'child_process';
import {
  BackendAvailability,
  GenerationOptions,
  GenerationResult,
  InferenceBackend,
  TokenCallback,
} from '../base.js';
import { execSync } from 'child_process';
import { loadConfig } from '../../core/config/config.js';
import { getEngineDirectory } from '../../core/engine/engine_installer.js';
import { parseGgufHeader } from '../../core/model/gguf_parser.js';
import { TuningPlan, tuningToEngineArgs } from '../../core/tuning/runtime_tuning.js';

const BINARY_NAMES = os.platform() === 'win32'
  ? ['llama-server.exe', 'server.exe']
  : ['llama-server', 'server'];

/**
 * Locate a real llama.cpp server binary: explicit config first, then PATH,
 * then a few conventional install locations. Returns null when none exists —
 * the caller must surface that instead of pretending to run.
 */
export function detectLlamaServerBinary(): string | null {
  const configured = loadConfig().llamaServerPath;
  if (configured && fs.existsSync(configured)) return configured;

  const pathDirs = (process.env.PATH || '').split(path.delimiter).filter(Boolean);
  const extraDirs = [
    path.join(process.cwd(), 'bin'),
    path.join(process.cwd(), 'llama.cpp', 'build', 'bin'),
    path.join(os.homedir(), 'llama.cpp', 'build', 'bin'),
    'C:/llama.cpp',
    'C:/Program Files/llama.cpp',
    '/usr/local/bin',
    '/opt/homebrew/bin',
  ];

  for (const dir of [...pathDirs, ...extraDirs]) {
    for (const name of BINARY_NAMES) {
      const candidate = path.join(dir, name);
      try {
        if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) return candidate;
      } catch {
        // Unreadable directory entry — keep looking.
      }
    }
  }

  return null;
}

async function findFreePort(preferred = 8080): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : preferred;
      server.close(() => resolve(port));
    });
  });
}

/**
 * Default ceiling for the automatically chosen context window.
 *
 * Large enough for the system prompts coding assistants send (10k+ tokens),
 * small enough that the KV cache stays affordable on consumer hardware. Models
 * advertising a million-token window would otherwise allocate far more memory
 * than the machine has.
 */
const DEFAULT_MAX_CONTEXT = 32768;

/**
 * Kill engine processes left behind by a previous run.
 *
 * A spawned llama-server is not killed automatically when its parent dies, so
 * restarting the API server orphans one holding the whole model in memory. The
 * next spawn then fails to allocate and looks like a mysterious startup crash.
 *
 * Only binaries living inside our own engines directory are touched: a
 * llama-server the user runs themselves is none of our business.
 */
export function killOrphanedEngines(): number {
  let killed = 0;
  for (const pid of listOwnEngines()) {
    try {
      process.kill(pid);
      killed++;
    } catch {
      // Already gone, or not ours to kill.
    }
  }
  return killed;
}

/**
 * Kill leftover engines and wait until they are actually gone.
 *
 * `process.kill` only delivers the signal; a llama-server holding tens of
 * gigabytes takes seconds to unmap them and hand the VRAM back. Spawning the
 * replacement in that window makes it fail to allocate, which surfaced as an
 * intermittent, unexplained startup crash whenever a large model was reloaded
 * with different settings. Waiting for the old process to disappear is what
 * makes back-to-back reconfiguration reliable.
 */
export async function killOrphanedEnginesAndWait(timeoutMs = 30000): Promise<number> {
  const killed = killOrphanedEngines();
  if (killed === 0) return 0;

  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (listOwnEngines().length === 0) return killed;
    await new Promise((r) => setTimeout(r, 250));
  }
  return killed;
}

/** PIDs of running llama-server processes that came out of our engines directory. */
function listOwnEngines(): number[] {
  const engineRoot = getEngineDirectory().toLowerCase();
  const pids: number[] = [];

  try {
    if (os.platform() === 'win32') {
      const output = execSync(
        'powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \\"Name=\'llama-server.exe\'\\" | ' +
          'Select-Object ProcessId, ExecutablePath | ConvertTo-Json -Compress; exit 0"',
        { encoding: 'utf8', timeout: 10000, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true }
      );
      const parsed = JSON.parse(output.trim() || 'null');
      for (const proc of Array.isArray(parsed) ? parsed : parsed ? [parsed] : []) {
        const execPath = String(proc.ExecutablePath || '').toLowerCase();
        if (!execPath.startsWith(engineRoot)) continue;
        pids.push(Number(proc.ProcessId));
      }
    } else {
      const output = execSync('pgrep -f llama-server || true', { encoding: 'utf8', timeout: 5000 });
      for (const line of output.trim().split('\n')) {
        const pid = Number(line.trim());
        if (!pid) continue;
        try {
          const exe = fs.readlinkSync(`/proc/${pid}/exe`).toLowerCase();
          if (!exe.startsWith(engineRoot)) continue;
          pids.push(pid);
        } catch {
          // Not inspectable or already gone.
        }
      }
    }
  } catch {
    // Enumeration unavailable; a stale engine will surface as a spawn failure.
  }

  return pids;
}

let cleanupRegistered = false;

/**
 * Make sure the engine dies with us on an orderly exit.
 *
 * This cannot cover SIGKILL or a hard crash, which is why the startup sweep
 * above exists as well; together they keep a stale multi-gigabyte process from
 * surviving indefinitely.
 */
function registerEngineCleanup(): void {
  if (cleanupRegistered) return;
  cleanupRegistered = true;

  const cleanup = () => {
    killOrphanedEngines();
  };

  process.once('exit', cleanup);
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGHUP'] as const) {
    process.once(signal, () => {
      cleanup();
      process.exit(0);
    });
  }
}

/**
 * How long to wait for the engine to become ready, from the model's real size.
 *
 * Loading is bound by how fast the weights can be pulled off disk, so the
 * budget is derived from total bytes at a deliberately pessimistic rate: a
 * model spread over several files on a SATA array is far slower to fault in
 * than one on NVMe, and being wrong here kills a load that was going to work.
 */
/** Total bytes of the model on disk, counting every part of a split set. */
function modelTotalBytes(modelPath: string): number {
  try {
    const dir = path.dirname(modelPath);
    const base = path.basename(modelPath);
    const split = base.match(/^(.*)-(\d{5})-of-(\d{5})\.gguf$/i);

    if (!split) return fs.statSync(modelPath).size;

    let total = 0;
    for (const entry of fs.readdirSync(dir)) {
      if (entry.startsWith(`${split[1]}-`) && entry.toLowerCase().endsWith('.gguf')) {
        total += fs.statSync(path.join(dir, entry)).size;
      }
    }
    return total;
  } catch {
    return 0;
  }
}

function startupBudgetMs(modelPath: string): number {
  const totalBytes = modelTotalBytes(modelPath);
  if (totalBytes === 0) return 300_000;

  // 100 MB/s is well below any SSD, leaving room for a cold cache and a busy
  // disk; floor at five minutes, cap at one hour.
  const estimated = (totalBytes / (100 * 1024 * 1024)) * 1000;
  return Math.min(3_600_000, Math.max(300_000, Math.round(estimated * 1.5)));
}

/**
 * Ceiling used when the model is larger than physical RAM.
 *
 * A disk-bound model lives or dies by the page cache: every expert kept warm
 * in memory is a read the next token does not pay for. The KV cache competes
 * for exactly that memory, and at 32k context over ~90 layers it claims
 * multiple gigabytes that would otherwise hold hot weights. A narrower window
 * costs prompt length; the wide one costs seconds per token. On a machine
 * where the model is several times the RAM, that trade is not close.
 */
const DISK_BOUND_MAX_CONTEXT = 8192;

/** Choose the context window: explicit setting, else the model's own maximum, capped. */
function resolveContextLength(modelPath: string, configured: number | null): number {
  // An explicit setting always wins — the user may know their prompts.
  if (configured !== null && configured > 0) return configured;

  const diskBound = modelTotalBytes(modelPath) > os.totalmem() * 0.9;
  const cap = diskBound ? DISK_BOUND_MAX_CONTEXT : DEFAULT_MAX_CONTEXT;

  try {
    const meta = parseGgufHeader(modelPath);
    if (meta.contextLength > 0) return Math.min(meta.contextLength, cap);
  } catch {
    // Header unreadable — fall through to the default.
  }

  return cap;
}

/**
 * Real llama.cpp execution. AILOFlow spawns an actual `llama-server` process
 * bound to localhost and streams tokens from it; all timing numbers come from
 * llama.cpp's own `timings` block.
 */
export class LlamaCppBackend implements InferenceBackend {
  public id = 'llama.cpp';
  public name = 'llama.cpp (native engine)';

  private proc: ChildProcess | null = null;
  private port = 0;
  private modelPath = '';
  /** Tuning arguments the running engine was started with, for reuse checks. */
  private launchSignature = '';
  private startupLog: string[] = [];
  private contextLength = 0;
  /** Abort handle for the generation currently in flight, if any. */
  private activeRun: AbortController | null = null;
  /** Tuning plan to start the next model with; set by the registry. */
  private tuningPlan: TuningPlan | null = null;

  /** Hand the engine the plan it should launch with. */
  public setTuningPlan(plan: TuningPlan | null): void {
    this.tuningPlan = plan;
  }

  public getTuningPlan(): TuningPlan | null {
    return this.tuningPlan;
  }

  public async checkAvailability(): Promise<BackendAvailability> {
    const binary = detectLlamaServerBinary();
    if (!binary) {
      return {
        id: this.id,
        name: this.name,
        available: false,
        reason:
          'No llama.cpp server binary found. Build llama.cpp and set its "llama-server" path in Settings, ' +
          'or put it on your PATH.',
      };
    }
    return { id: this.id, name: this.name, available: true, reason: null, detail: binary };
  }

  /** `modelRef` must be a real GGUF path (optionally prefixed with `gguf:`). */
  public async initialize(modelRef: string): Promise<boolean> {
    const modelPath = modelRef.startsWith('gguf:') ? modelRef.slice('gguf:'.length) : modelRef;

    const binary = detectLlamaServerBinary();
    if (!binary) {
      throw new Error(
        'llama.cpp backend unavailable: no "llama-server" binary configured or on PATH. ' +
          'Set the path in Settings before selecting a GGUF model.'
      );
    }
    if (!fs.existsSync(modelPath)) {
      throw new Error(`GGUF model file not found: ${modelPath}`);
    }

    const config = loadConfig();

    // The tuning plan decides every engine argument from the machine and the
    // model, with the user's overrides on top. Falling back to a bare context
    // setting only happens when the model's metadata cannot be read.
    const plan = this.tuningPlan;
    let tuningArgs: string[];
    let contextLength: number;

    if (plan) {
      contextLength = plan.contextLength.effective;
      tuningArgs = tuningToEngineArgs(plan);
    } else {
      contextLength = resolveContextLength(modelPath, config.contextLength);
      // llama.cpp defaults to 4096, which an IDE assistant blows past with its
      // system prompt alone. The window is fixed once the server starts.
      tuningArgs = ['--ctx-size', String(contextLength)];
      if (config.gpuLayers !== null) tuningArgs.push('--n-gpu-layers', String(config.gpuLayers));
    }

    // Reuse the running engine only when it was started the same way. Matching
    // on the model path alone meant a settings change was accepted, reported as
    // loaded, and quietly ignored — the engine kept running with the arguments
    // it was born with, and every measurement of the new setting was a
    // measurement of the old one.
    const signature = tuningArgs.join(' ');
    if (this.proc && this.modelPath === modelPath && this.launchSignature === signature) return true;
    if (this.proc) await this.dispose();

    this.port = await findFreePort();
    this.modelPath = modelPath;
    this.contextLength = contextLength;
    this.launchSignature = signature;
    this.startupLog = [];

    const args = ['--model', modelPath, '--host', '127.0.0.1', '--port', String(this.port), ...tuningArgs];

    // A previous engine still holding the GPU would make this spawn fail, and
    // it does not release it the instant it is signalled.
    await killOrphanedEnginesAndWait();

    this.proc = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    registerEngineCleanup();

    const capture = (data: Buffer) => {
      const text = data.toString();
      this.startupLog.push(text);
      if (this.startupLog.length > 200) this.startupLog.shift();
    };
    this.proc.stdout?.on('data', capture);
    this.proc.stderr?.on('data', capture);

    // An engine that rejects an argument dies in tens of milliseconds, and the
    // 'exit' event can beat the final 'data' events out of the pipes — which
    // used to leave the thrown error holding half a line of banner and none of
    // the actual complaint. Wait for both streams to close before reading the
    // log, so the reason survives.
    let exited = false;
    let exitCode: number | null = null;
    let openStreams = 2;
    const drained = new Promise<void>((resolve) => {
      const done = () => { if (--openStreams <= 0) resolve(); };
      this.proc?.stdout?.on('close', done);
      this.proc?.stderr?.on('close', done);
    });
    this.proc.on('exit', (code) => { exited = true; exitCode = code; });

    const exitReport = async (): Promise<string> => {
      await Promise.race([drained, new Promise((r) => setTimeout(r, 2000))]);
      const tail = this.startupLog.join('').trimEnd().slice(-2000);
      const status = exitCode === null ? 'was terminated' : `exited with code ${exitCode}`;
      return tail
        ? `llama-server ${status} during startup. Engine output:\n${tail}`
        : `llama-server ${status} during startup without producing any output. ` +
          `Arguments: ${args.join(' ')}`;
    };

    // llama.cpp loads weights before serving; poll /health until it is ready.
    //
    // The budget has to scale with the model: a 45 GB file is ready in under a
    // minute, but a 220 GB one spread over five files needs many minutes just to
    // fault its weights in. A fixed timeout silently killed exactly the large
    // models this runtime exists for.
    const deadline = Date.now() + startupBudgetMs(modelPath);
    while (Date.now() < deadline) {
      if (exited) throw new Error(await exitReport());
      try {
        const res = await fetch(`http://127.0.0.1:${this.port}/health`);
        if (res.ok) return true;
      } catch {
        // Not listening yet.
      }
      await new Promise((r) => setTimeout(r, 400));
    }

    const waited = Math.round(startupBudgetMs(modelPath) / 1000);
    await this.dispose();
    throw new Error(
      `llama-server did not become ready within ${waited}s. Last output:\n` +
        this.startupLog.join('').slice(-1200)
    );
  }

  public async generateStream(options: GenerationOptions, onToken: TokenCallback): Promise<GenerationResult> {
    if (!this.proc || !this.port) {
      throw new Error('LlamaCppBackend is not initialized. Load a GGUF model first.');
    }

    const messages: Array<{ role: string; content: string }> = [];
    if (options.systemPrompt) messages.push({ role: 'system', content: options.systemPrompt });
    for (const turn of options.history || []) messages.push({ role: turn.role, content: turn.content });
    messages.push({ role: 'user', content: options.prompt });

    const payload: Record<string, unknown> = { messages, stream: true };
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.topP !== undefined) payload.top_p = options.topP;
    if (options.topK !== undefined) payload.top_k = options.topK;
    if (options.repetitionPenalty !== undefined) payload.repeat_penalty = options.repetitionPenalty;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;
    if (options.seed !== undefined) payload.seed = options.seed;
    if (options.stopSequences?.length) payload.stop = options.stopSequences;

    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let fullText = '';
    let reasoningText = '';
    let timings: LlamaTimings | null = null;
    let usage: { prompt_tokens?: number; completion_tokens?: number } | null = null;

    // Only one generation can be in flight per engine, and a client that stops
    // listening does not always close its socket — some IDEs simply abandon the
    // response. Superseding the previous run guarantees a stuck generation is
    // released at the latest when the next request arrives.
    this.stopActiveRun();
    const run = new AbortController();
    this.activeRun = run;
    if (options.signal) {
      if (options.signal.aborted) run.abort();
      else options.signal.addEventListener('abort', () => run.abort(), { once: true });
    }

    // Aborting this fetch closes the socket to llama-server, which stops the
    // generation instead of leaving it running against a client that has gone.
    const res = await fetch(`http://127.0.0.1:${this.port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: run.signal,
    });

    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => '');
      throw new Error(`llama.cpp generation failed (HTTP ${res.status}): ${detail.slice(0, 300)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith('data:')) continue;
        const data = trimmed.slice(5).trim();
        if (data === '[DONE]') continue;

        let chunk: LlamaChunk;
        try {
          chunk = JSON.parse(data);
        } catch {
          continue;
        }

        const delta = chunk.choices?.[0]?.delta;

        // Thinking models stream their chain of thought on a separate field;
        // it counts for latency but must not be spliced into the answer.
        const reasoningPiece = delta?.reasoning_content || '';
        if (reasoningPiece) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          reasoningText += reasoningPiece;
          onToken({ token: reasoningPiece, isFinished: false, kind: 'reasoning' });
        }

        const piece = delta?.content || '';
        if (piece) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          fullText += piece;
          onToken({ token: piece, isFinished: false, kind: 'content' });
        }
        if (chunk.timings) timings = chunk.timings;
        if (chunk.usage) usage = chunk.usage;
      }
    }

    if (this.activeRun === run) this.activeRun = null;
    onToken({ token: '', isFinished: true });
    const totalDurationMs = performance.now() - startedAt;

    return {
      text: fullText,
      reasoning: reasoningText || undefined,
      metrics: {
        promptTokens: timings?.prompt_n ?? usage?.prompt_tokens ?? null,
        completionTokens: timings?.predicted_n ?? usage?.completion_tokens ?? null,
        firstTokenLatencyMs: firstTokenAt === null ? null : Number((firstTokenAt - startedAt).toFixed(1)),
        tokensPerSecond: timings?.predicted_per_second != null
          ? Number(timings.predicted_per_second.toFixed(2))
          : null,
        promptTokensPerSecond: timings?.prompt_per_second != null
          ? Number(timings.prompt_per_second.toFixed(2))
          : null,
        totalDurationMs: Number(totalDurationMs.toFixed(1)),
        backendId: this.id,
        modelId: `gguf:${this.modelPath}`,
      },
    };
  }

  /** Stop whatever is generating right now. Safe to call when nothing is. */
  public stopActiveRun(): boolean {
    if (!this.activeRun || this.activeRun.signal.aborted) return false;
    this.activeRun.abort();
    this.activeRun = null;
    return true;
  }

  public getLoadedModelPath(): string | null {
    return this.proc ? this.modelPath : null;
  }

  public async dispose(): Promise<void> {
    const proc = this.proc;
    this.proc = null;
    this.port = 0;
    this.modelPath = '';
    this.launchSignature = '';

    if (!proc) return;

    // Return only once the memory is actually back. An unload that resolves
    // while tens of gigabytes are still mapped makes the next load — or the
    // user watching the VRAM gauge — see a machine that has not freed anything.
    const exited = new Promise<void>((resolve) => {
      proc.once('exit', () => resolve());
    });
    proc.kill();
    await Promise.race([exited, new Promise((r) => setTimeout(r, 30000))]);
  }
}

interface LlamaTimings {
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
}

interface LlamaChunk {
  choices?: Array<{ delta?: { content?: string; reasoning_content?: string } }>;
  timings?: LlamaTimings;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}
