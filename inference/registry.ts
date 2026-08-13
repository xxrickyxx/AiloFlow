import { loadConfig } from '../core/config/config.js';
import { discoverComputerProfile } from '../core/hardware/discovery.js';
import { parseGgufModel } from '../core/model/gguf_parser.js';
import { DiscoveredModel, findModelById } from '../core/model/model_registry.js';
import { TuningPlan, buildTuningPlan, describeModel } from '../core/tuning/runtime_tuning.js';
import { BackendAvailability, GenerationMetrics, InferenceBackend } from './base.js';
import { LlamaCppBackend } from './llama_cpp/llama_cpp_backend.js';
import { OllamaBackend } from './ollama/ollama_backend.js';

import { AiloHierarchicalBackend } from './ailo_hierarchical/ailo_hierarchical_backend.js';

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
  private ailoHierarchical = new AiloHierarchicalBackend();
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
    const results: BackendAvailability[] = [
      await this.ailoHierarchical.checkAvailability(),
      await this.llamaCpp.checkAvailability(),
    ];

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

  /**
   * Build the tuning plan for a model without loading it.
   *
   * Exposed so the interface can show the consequences of a setting — bytes per
   * token, active parameters, what will not fit in RAM — before the user
   * commits to a load that may take minutes.
   */
  public async planFor(model: DiscoveredModel): Promise<TuningPlan | null> {
    if (!model.filePath || model.source === 'sflow') return null;

    try {
      const meta = parseGgufModel(model.filePath);
      const totalBytes = model.fileSizeBytes ?? meta.fileSizeBytes;
      const shape = describeModel(meta, totalBytes);
      const profile = await discoverComputerProfile();

      const measured = profile.storageDrives.filter((d) => d.performanceProfile?.measured);
      const storageBandwidthMBps = measured.length
        ? measured.reduce((sum, d) => sum + (d.performanceProfile?.seqReadMBps || 0), 0)
        : null;

      return buildTuningPlan({
        profile,
        model: shape,
        overrides: loadConfig().tuning,
        storageBandwidthMBps,
      });
    } catch {
      // Metadata unreadable: the engine falls back to its own defaults rather
      // than being handed a plan built on guesses.
      return null;
    }
  }

  public async planForId(modelId: string): Promise<TuningPlan | null> {
    const model = await findModelById(modelId);
    if (!model) throw new Error(`Model not found: ${modelId}`);
    return this.planFor(model);
  }

  public getLastMetrics(): GenerationMetrics | null {
    return this.lastMetrics;
  }

  public getActiveBackend(): InferenceBackend | null {
    return this.active;
  }

  /** Load a model into a real engine chosen from what the model supports. */
  public async loadModel(modelId: string, preferredBackendId?: string): Promise<LoadedModelState> {
    const model = await findModelById(modelId);
    if (!model) throw new Error(`Model not found: ${modelId}`);

    const startedAt = performance.now();

    let backend: InferenceBackend;
    let ref: string;

    if (preferredBackendId === 'ailo-hierarchical' || model.source === 'sflow') {
      backend = this.ailoHierarchical;
      ref = model.filePath || model.id;
    } else if (model.source === 'ollama') {
      const availability = await this.ollama.checkAvailability();
      if (!availability.available) {
        throw new Error(availability.reason || 'Ollama backend is unavailable.');
      }
      backend = this.ollama;
      ref = model.id;
    } else if (model.source === 'gguf') {
      const llamaAvailability = await this.llamaCpp.checkAvailability();
      if (llamaAvailability.available) {
        if (!model.filePath) throw new Error(`GGUF model has no file path: ${modelId}`);

        this.llamaCpp.setTuningPlan(await this.planFor(model));
        backend = this.llamaCpp;
        ref = model.filePath;
      } else {
        // Fallback to AILOFlow Native Hierarchical Engine if llama.cpp server is not installed
        backend = this.ailoHierarchical;
        ref = model.filePath || model.id;
      }
    } else {
      backend = this.ailoHierarchical;
      ref = model.filePath || model.id;
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
