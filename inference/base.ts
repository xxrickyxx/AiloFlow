export interface GenerationOptions {
  prompt: string;
  systemPrompt?: string;
  /** Prior turns, oldest first, excluding the current prompt. */
  history?: Array<{ role: 'user' | 'assistant'; content: string }>;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  stopSequences?: string[];
  seed?: number;
  contextLength?: number;
  /**
   * Aborts the run. A client that disconnects — an IDE cancelling a request, a
   * closed tab — must stop the engine too, otherwise it keeps generating into
   * nothing and holds the GPU for as long as the model feels like talking.
   */
  signal?: AbortSignal;
}

export interface StreamToken {
  token: string;
  isFinished: boolean;
  /**
   * Reasoning models (Qwen3, DeepSeek-R1, gpt-oss) emit their chain of thought
   * on a separate channel. Mixing it into the answer would corrupt the reply,
   * so it is tagged and kept apart.
   */
  kind?: 'content' | 'reasoning';
}

export type TokenCallback = (token: StreamToken) => void;

/**
 * Timings for a completed generation. Every field is measured, never assumed;
 * fields the engine did not report stay null so callers can render "n/d".
 */
export interface GenerationMetrics {
  promptTokens: number | null;
  completionTokens: number | null;
  /** Wall-clock time from request start to the first streamed token. */
  firstTokenLatencyMs: number | null;
  /** Completion tokens divided by the engine's own eval duration. */
  tokensPerSecond: number | null;
  /** Prompt tokens divided by the engine's prompt eval duration. */
  promptTokensPerSecond: number | null;
  totalDurationMs: number;
  backendId: string;
  modelId: string;
}

export interface GenerationResult {
  text: string;
  /** Chain-of-thought, when the model produced one. */
  reasoning?: string;
  metrics: GenerationMetrics;
}

export interface BackendAvailability {
  id: string;
  name: string;
  available: boolean;
  /** Human-readable explanation when `available` is false. */
  reason: string | null;
  detail?: string;
}

export interface InferenceBackend {
  id: string;
  name: string;
  /** Resolve whether this engine can actually run right now. */
  checkAvailability(): Promise<BackendAvailability>;
  initialize(modelRef: string): Promise<boolean>;
  generateStream(options: GenerationOptions, onToken: TokenCallback): Promise<GenerationResult>;
  dispose(): Promise<void>;
}
