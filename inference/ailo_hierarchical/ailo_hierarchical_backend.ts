import path from 'path';
import { SFlowContainer, SFlowManifest } from '../../formats/sflow/sflow_format.js';
import { AiloStorageFabric } from '../../core/storage/storage_fabric.js';
import { HierarchicalCache } from '../../core/cache/hierarchical_cache.js';
import { PrefetchEngine } from '../../core/prefetch/prefetch_engine.js';
import { discoverComputerProfile } from '../../core/hardware/discovery.js';
import { StorageDriveInfo } from '../../core/hardware/types.js';
import { parseGgufModel } from '../../core/model/gguf_parser.js';
import { AiloActiveWeightEngine } from '../../core/engine/active_weight_engine.js';
import {
  BackendAvailability,
  GenerationMetrics,
  GenerationOptions,
  GenerationResult,
  InferenceBackend,
  TokenCallback,
} from '../base.js';

export class AiloHierarchicalBackend implements InferenceBackend {
  public id = 'ailo-hierarchical';
  public name = 'AILOFlow Hierarchical Engine (Active Weights)';

  private container: SFlowContainer | null = null;
  private fabric: AiloStorageFabric | null = null;
  private cache: HierarchicalCache | null = null;
  private prefetcher: PrefetchEngine | null = null;
  private activeEngine: AiloActiveWeightEngine | null = null;
  private modelRef: string | null = null;
  private loadedModelName = 'Unknown Model';

  public async checkAvailability(): Promise<BackendAvailability> {
    return {
      id: this.id,
      name: this.name,
      available: true,
      reason: null,
      detail: 'Native AILOFlow Active Weight Hierarchical Memory Engine',
    };
  }

  public async initialize(modelRef: string): Promise<boolean> {
    this.modelRef = modelRef;
    const profile = await discoverComputerProfile();

    let manifest: SFlowManifest;
    let baseDir = process.cwd();

    if (modelRef.endsWith('.sflow') || modelRef.endsWith('.json')) {
      this.container = SFlowContainer.load(modelRef);
      manifest = this.container.manifest;
      baseDir = path.dirname(modelRef);
      this.loadedModelName = manifest.modelName;
    } else {
      // Build an inline SFlowManifest from GGUF metadata
      const meta = parseGgufModel(modelRef);
      this.loadedModelName = meta.architecture || path.basename(modelRef);
      baseDir = path.dirname(modelRef);

      const fallbackDrive: StorageDriveInfo = {
        id: 'drive0',
        devicePath: baseDir,
        mountPoint: baseDir,
        label: 'Default Storage',
        filesystem: 'NTFS',
        totalSizeBytes: 100 * 1024 * 1024 * 1024,
        freeSizeBytes: 50 * 1024 * 1024 * 1024,
        type: 'NVMe',
      };

      const drives = profile.storageDrives.length > 0 ? profile.storageDrives : [fallbackDrive];

      manifest = {
        formatVersion: '1.0',
        modelName: this.loadedModelName,
        architecture: meta.architecture,
        parameterCountBillions: Number(((meta.totalTensorDataBytes * 2) / 1e9).toFixed(1)),
        quantization: meta.quantization,
        totalSizeBytes: meta.fileSizeBytes,
        blockCount: meta.blockCount || 32,
        contextLength: meta.contextLength || 4096,
        createdAt: new Date().toISOString(),
        shards: [
          {
            shardId: 'shard-0',
            layerStart: 0,
            layerEnd: meta.blockCount || 32,
            tensorCount: meta.tensorCount,
            sizeBytes: meta.fileSizeBytes,
            targetDriveId: drives[0].id,
            targetMountPoint: drives[0].mountPoint,
            relFilePath: path.basename(modelRef),
            checksum: 'gguf-raw',
            priority: 'HIGH',
            accessFrequency: 1.0,
          },
        ],
        tensorMap: meta.tensors.map((t) => ({
          name: t.name,
          layerIndex: this.extractLayerIndex(t.name),
          shardId: 'shard-0',
          offsetInShard: Number(t.offset),
          sizeBytes: t.sizeBytes,
        })),
      };
    }

    const fallbackDrive: StorageDriveInfo = {
      id: 'drive0',
      devicePath: baseDir,
      mountPoint: baseDir,
      label: 'Default Storage',
      filesystem: 'NTFS',
      totalSizeBytes: 100 * 1024 * 1024 * 1024,
      freeSizeBytes: 50 * 1024 * 1024 * 1024,
      type: 'NVMe',
    };

    const drives = profile.storageDrives.length > 0 ? profile.storageDrives : [fallbackDrive];

    this.fabric = new AiloStorageFabric(drives, baseDir);

    const firstGpu = profile.gpus?.[0];
    const freeVram = firstGpu?.vramFreeBytes || firstGpu?.vramTotalBytes || (4096 * 1024 * 1024);
    const vramBudget = Math.max(1024 * 1024 * 1024, Math.floor(freeVram * 0.8));
    const ramBudget = Math.max(2048 * 1024 * 1024, Math.floor((profile.ram?.freeBytes || 8192 * 1024 * 1024) * 0.7));

    this.cache = new HierarchicalCache(vramBudget, ramBudget);
    this.prefetcher = new PrefetchEngine(this.cache, this.fabric, manifest, 2);
    this.activeEngine = new AiloActiveWeightEngine(manifest, this.fabric, this.cache, this.prefetcher);

    return true;
  }

  public getIoStats() {
    return this.fabric?.getIoStats();
  }

  public getPrefetchStats() {
    return this.prefetcher?.getStats();
  }

  public getCacheMetrics() {
    return this.cache?.getCacheMetrics();
  }

  private extractLayerIndex(tensorName: string): number {
    const match = tensorName.match(/blk\.(\d+)\./) || tensorName.match(/layers\.(\d+)\./);
    return match ? parseInt(match[1], 10) : 0;
  }

  public async generateStream(
    options: GenerationOptions,
    onToken: TokenCallback
  ): Promise<GenerationResult> {
    if (!this.activeEngine) {
      throw new Error('AILOFlow Hierarchical Engine is not initialized.');
    }

    const startedAt = performance.now();
    const promptTokenIds = options.prompt.split(/\s+/).map((_, i) => i + 1);

    const maxTokens = options.maxTokens || 128;
    let fullText = '';
    let firstTokenLatencyMs: number | null = null;

    for (let step = 0; step < maxTokens; step++) {
      if (options.signal?.aborted) {
        break;
      }

      const stepStart = performance.now();
      const stepMetrics = await this.activeEngine.generateStep(promptTokenIds, step, {
        temperature: options.temperature,
        topP: options.topP,
        topK: options.topK,
        repetitionPenalty: options.repetitionPenalty,
        promptText: options.prompt,
      });

      if (step === 0) {
        firstTokenLatencyMs = Number((performance.now() - startedAt).toFixed(2));
      }

      fullText += stepMetrics.tokenText;
      const isFinished = stepMetrics.isFinished || step === maxTokens - 1;

      onToken({
        token: stepMetrics.tokenText,
        isFinished,
        kind: 'content',
      });

      // Yield execution to the Node.js I/O event loop so Express flushes each
      // SSE token chunk down the socket immediately.
      await new Promise((resolve) => setImmediate(resolve));

      if (isFinished) {
        break;
      }
    }

    const totalDurationMs = Number((performance.now() - startedAt).toFixed(2));
    const completionTokens = maxTokens;
    const tokensPerSecond = Number(((completionTokens / totalDurationMs) * 1000).toFixed(2));

    const metrics: GenerationMetrics = {
      promptTokens: promptTokenIds.length,
      completionTokens,
      firstTokenLatencyMs,
      tokensPerSecond,
      promptTokensPerSecond: Number(((promptTokenIds.length / (firstTokenLatencyMs || 1)) * 1000).toFixed(2)),
      totalDurationMs,
      backendId: this.id,
      modelId: this.modelRef || 'unknown',
    };

    return {
      text: fullText.trim(),
      metrics,
    };
  }

  public async dispose(): Promise<void> {
    if (this.activeEngine) {
      this.activeEngine.reset();
    }
    this.activeEngine = null;
    this.cache = null;
    this.prefetcher = null;
    this.fabric = null;
    this.container = null;
  }
}
