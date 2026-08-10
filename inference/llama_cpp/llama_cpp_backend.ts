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
  const engineRoot = getEngineDirectory().toLowerCase();
  let killed = 0;

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
        try {
          process.kill(Number(proc.ProcessId));
          killed++;
        } catch {
          // Already gone, or not ours to kill.
        }
      }
    } else {
      const output = execSync('pgrep -f llama-server || true', { encoding: 'utf8', timeout: 5000 });
      for (const line of output.trim().split('\n')) {
        const pid = Number(line.trim());
        if (!pid) continue;
        try {
          const exe = fs.readlinkSync(`/proc/${pid}/exe`).toLowerCase();
          if (!exe.startsWith(engineRoot)) continue;
          process.kill(pid);
          killed++;
        } catch {
          // Not inspectable or already gone.
        }
      }
    }
  } catch {
    // Enumeration unavailable; a stale engine will surface as a spawn failure.
  }

  return killed;
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

/** Choose the context window: explicit setting, else the model's own maximum, capped. */
function resolveContextLength(modelPath: string, configured: number | null): number {
  if (configured !== null && configured > 0) return configured;

  try {
    const meta = parseGgufHeader(modelPath);
    if (meta.contextLength > 0) return Math.min(meta.contextLength, DEFAULT_MAX_CONTEXT);
  } catch {
    // Header unreadable — fall through to the default.
  }

  return DEFAULT_MAX_CONTEXT;
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
  private startupLog: string[] = [];
  private contextLength = 0;

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

    if (this.proc && this.modelPath === modelPath) return true;
    if (this.proc) await this.dispose();

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
    this.port = await findFreePort();
    this.modelPath = modelPath;
    this.startupLog = [];

    const contextLength = resolveContextLength(modelPath, config.contextLength);
    this.contextLength = contextLength;

    const args = [
      '--model', modelPath,
      '--host', '127.0.0.1',
      '--port', String(this.port),
      // llama.cpp defaults to 4096, which an IDE assistant blows past with its
      // system prompt alone. The window is fixed once the server starts, so it
      // has to be sized correctly here rather than per request.
      '--ctx-size', String(contextLength),
    ];
    if (config.gpuLayers !== null) args.push('--n-gpu-layers', String(config.gpuLayers));

    // A previous engine still holding the GPU would make this spawn fail.
    killOrphanedEngines();

    this.proc = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true });
    registerEngineCleanup();

    const capture = (data: Buffer) => {
      const text = data.toString();
      this.startupLog.push(text);
      if (this.startupLog.length > 200) this.startupLog.shift();
    };
    this.proc.stdout?.on('data', capture);
    this.proc.stderr?.on('data', capture);

    let exited = false;
    this.proc.on('exit', () => { exited = true; });

    // llama.cpp loads weights before serving; poll /health until it is ready.
    const deadline = Date.now() + 180_000;
    while (Date.now() < deadline) {
      if (exited) {
        throw new Error(
          `llama-server exited during startup. Last output:\n${this.startupLog.join('').slice(-1500)}`
        );
      }
      try {
        const res = await fetch(`http://127.0.0.1:${this.port}/health`);
        if (res.ok) return true;
      } catch {
        // Not listening yet.
      }
      await new Promise((r) => setTimeout(r, 400));
    }

    await this.dispose();
    throw new Error('llama-server did not become ready within 180s.');
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

    // Aborting this fetch closes the socket to llama-server, which stops the
    // generation instead of leaving it running against a client that has gone.
    const res = await fetch(`http://127.0.0.1:${this.port}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: options.signal,
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

  public getLoadedModelPath(): string | null {
    return this.proc ? this.modelPath : null;
  }

  public async dispose(): Promise<void> {
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
    this.port = 0;
    this.modelPath = '';
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
