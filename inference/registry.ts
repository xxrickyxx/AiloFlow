import { loadConfig } from '../core/config/config.js';
import { DiscoveredModel, findModelById } from '../core/model/model_registry.js';
import { BackendAvailability, GenerationMetrics, InferenceBackend } from './base.js';
import { LlamaCppBackend } from './llama_cpp/llama_cpp_backend.js';
import { OllamaBackend } from './ollama/ollama_backend.js';

export interface LoadedModelState {
  model: DiscoveredModel;
  backendId: string;
  backendName: string;
  loadedAt: string;
  loadDurationMs: number;
}

/**
 * Owns the single active inference engine.
 *
 * There is deliberately no fallback that produces text without a real engine:
 * if nothing can run the model, loading fails with an explanation.
 */
export class InferenceRegistry {
  private llamaCpp = new LlamaCppBackend();
  private ollama: OllamaBackend;

  private active: InferenceBackend | null = null;
  private state: LoadedModelState | null = null;
  private lastMetrics: GenerationMetrics | null = null;

  constructor() {
    this.ollama = new OllamaBackend(loadConfig().ollamaBaseUrl);
  }

  public async listAvailability(): Promise<BackendAvailability[]> {
    const config = loadConfig();
    const results: BackendAvailability[] = [await this.llamaCpp.checkAvailability()];

    if (config.ollamaEnabled) {
      results.push(await this.ollama.checkAvailability());
    } else {
      results.push({
        id: 'ollama',
        name: 'Ollama (local daemon)',
        available: false,
        reason: 'Disabled in settings.',
      });
    }

    return results;
  }

  public getState(): LoadedModelState | null {
    return this.state;
  }

  public getLastMetrics(): GenerationMetrics | null {
    return this.lastMetrics;
  }

  public getActiveBackend(): InferenceBackend | null {
    return this.active;
  }

  /** Load a model into a real engine chosen from what the model supports. */
  public async loadModel(modelId: string): Promise<LoadedModelState> {
    const model = await findModelById(modelId);
    if (!model) throw new Error(`Model not found: ${modelId}`);

    const startedAt = performance.now();

    let backend: InferenceBackend;
    let ref: string;

    if (model.source === 'ollama') {
      const availability = await this.ollama.checkAvailability();
      if (!availability.available) {
        throw new Error(availability.reason || 'Ollama backend is unavailable.');
      }
      backend = this.ollama;
      ref = model.id;
    } else if (model.source === 'gguf') {
      const availability = await this.llamaCpp.checkAvailability();
      if (!availability.available) {
        throw new Error(availability.reason || 'llama.cpp backend is unavailable.');
      }
      if (!model.filePath) throw new Error(`GGUF model has no file path: ${modelId}`);
      backend = this.llamaCpp;
      ref = model.filePath;
    } else {
      throw new Error(
        '.sflow containers describe how a model is sharded across storage; they are not directly ' +
          'executable. Load the source GGUF for generation, and use the storage pipeline benchmark ' +
          'to measure the .sflow container.'
      );
    }

    if (this.active && this.active !== backend) {
      await this.active.dispose();
    }

    await backend.initialize(ref);
    this.active = backend;
    this.state = {
      model,
      backendId: backend.id,
      backendName: backend.name,
      loadedAt: new Date().toISOString(),
      loadDurationMs: Number((performance.now() - startedAt).toFixed(1)),
    };

    return this.state;
  }

  public async unload(): Promise<void> {
    if (this.active) await this.active.dispose();
    this.active = null;
    this.state = null;
  }

  public recordMetrics(metrics: GenerationMetrics): void {
    this.lastMetrics = metrics;
  }

  /**
   * Stop whatever is generating right now.
   *
   * Needed because "the client went away" is not always observable: an IDE that
   * abandons a response without closing the socket leaves the engine running,
   * and the user needs a way to reclaim the GPU that does not involve
   * restarting the runtime.
   */
  public stopGeneration(): boolean {
    const backend = this.active;
    if (backend && backend instanceof LlamaCppBackend) return backend.stopActiveRun();
    return false;
  }

  /** Refresh the Ollama client when its base URL changes in settings. */
  public reconfigure(): void {
    this.ollama = new OllamaBackend(loadConfig().ollamaBaseUrl);
    if (this.state?.backendId === 'ollama') {
      this.active = this.ollama;
      void this.ollama.initialize(this.state.model.id);
    }
  }
}
