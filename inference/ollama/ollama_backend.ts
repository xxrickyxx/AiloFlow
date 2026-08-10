import {
  BackendAvailability,
  GenerationOptions,
  GenerationResult,
  InferenceBackend,
  TokenCallback,
} from '../base.js';

/**
 * Real inference against a locally running Ollama daemon.
 *
 * Every number reported back (token counts, eval durations) comes from the
 * daemon's own final response object — nothing is estimated here.
 */
export class OllamaBackend implements InferenceBackend {
  public id = 'ollama';
  public name = 'Ollama (local daemon)';

  private baseUrl: string;
  private modelTag = '';

  constructor(baseUrl = 'http://127.0.0.1:11434') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  public async checkAvailability(): Promise<BackendAvailability> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    try {
      const res = await fetch(`${this.baseUrl}/api/tags`, { signal: controller.signal });
      if (!res.ok) {
        return {
          id: this.id,
          name: this.name,
          available: false,
          reason: `Ollama responded with HTTP ${res.status}`,
        };
      }
      const body = (await res.json()) as { models?: unknown[] };
      return {
        id: this.id,
        name: this.name,
        available: true,
        reason: null,
        detail: `${body.models?.length ?? 0} models available at ${this.baseUrl}`,
      };
    } catch {
      return {
        id: this.id,
        name: this.name,
        available: false,
        reason: `No Ollama daemon reachable at ${this.baseUrl}. Start it with "ollama serve".`,
      };
    } finally {
      clearTimeout(timer);
    }
  }

  /** `modelRef` is an `ollama:<tag>` id or a bare Ollama tag. */
  public async initialize(modelRef: string): Promise<boolean> {
    this.modelTag = modelRef.startsWith('ollama:') ? modelRef.slice('ollama:'.length) : modelRef;
    if (!this.modelTag) throw new Error('OllamaBackend requires a model tag.');
    return true;
  }

  public async generateStream(options: GenerationOptions, onToken: TokenCallback): Promise<GenerationResult> {
    if (!this.modelTag) throw new Error('OllamaBackend is not initialized with a model.');

    const messages: Array<{ role: string; content: string }> = [];
    if (options.systemPrompt) messages.push({ role: 'system', content: options.systemPrompt });
    for (const turn of options.history || []) messages.push({ role: turn.role, content: turn.content });
    messages.push({ role: 'user', content: options.prompt });

    const ollamaOptions: Record<string, unknown> = {};
    if (options.temperature !== undefined) ollamaOptions.temperature = options.temperature;
    if (options.topP !== undefined) ollamaOptions.top_p = options.topP;
    if (options.topK !== undefined) ollamaOptions.top_k = options.topK;
    if (options.repetitionPenalty !== undefined) ollamaOptions.repeat_penalty = options.repetitionPenalty;
    if (options.maxTokens !== undefined) ollamaOptions.num_predict = options.maxTokens;
    if (options.contextLength !== undefined) ollamaOptions.num_ctx = options.contextLength;
    if (options.seed !== undefined) ollamaOptions.seed = options.seed;
    if (options.stopSequences?.length) ollamaOptions.stop = options.stopSequences;

    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let fullText = '';
    let reasoningText = '';
    let finalStats: OllamaFinalResponse | null = null;

    const res = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: this.modelTag, messages, stream: true, options: ollamaOptions }),
      signal: options.signal,
    });

    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => '');
      throw new Error(`Ollama generation failed (HTTP ${res.status}): ${detail.slice(0, 300)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      // The trailing element may be a partial JSON object; keep it buffered.
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        let chunk: OllamaStreamChunk;
        try {
          chunk = JSON.parse(trimmed);
        } catch {
          continue;
        }

        if (chunk.error) throw new Error(`Ollama error: ${chunk.error}`);

        // Ollama exposes reasoning models' chain of thought as `thinking`.
        const thinking = chunk.message?.thinking || '';
        if (thinking) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          reasoningText += thinking;
          onToken({ token: thinking, isFinished: false, kind: 'reasoning' });
        }

        const piece = chunk.message?.content || '';
        if (piece) {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          fullText += piece;
          onToken({ token: piece, isFinished: false, kind: 'content' });
        }

        if (chunk.done) {
          finalStats = chunk as OllamaFinalResponse;
          onToken({ token: '', isFinished: true });
        }
      }
    }

    const totalDurationMs = performance.now() - startedAt;
    const evalCount = finalStats?.eval_count ?? null;
    const evalDurationNs = finalStats?.eval_duration ?? null;
    const promptEvalCount = finalStats?.prompt_eval_count ?? null;
    const promptEvalDurationNs = finalStats?.prompt_eval_duration ?? null;

    return {
      text: fullText,
      reasoning: reasoningText || undefined,
      metrics: {
        promptTokens: promptEvalCount,
        completionTokens: evalCount,
        firstTokenLatencyMs: firstTokenAt === null ? null : Number((firstTokenAt - startedAt).toFixed(1)),
        tokensPerSecond:
          evalCount !== null && evalDurationNs
            ? Number((evalCount / (evalDurationNs / 1e9)).toFixed(2))
            : null,
        promptTokensPerSecond:
          promptEvalCount !== null && promptEvalDurationNs
            ? Number((promptEvalCount / (promptEvalDurationNs / 1e9)).toFixed(2))
            : null,
        totalDurationMs: Number(totalDurationMs.toFixed(1)),
        backendId: this.id,
        modelId: `ollama:${this.modelTag}`,
      },
    };
  }

  public async dispose(): Promise<void> {
    this.modelTag = '';
  }
}

interface OllamaStreamChunk {
  message?: { role: string; content: string; thinking?: string };
  done?: boolean;
  error?: string;
}

interface OllamaFinalResponse extends OllamaStreamChunk {
  eval_count?: number;
  eval_duration?: number;
  prompt_eval_count?: number;
  prompt_eval_duration?: number;
  total_duration?: number;
}
