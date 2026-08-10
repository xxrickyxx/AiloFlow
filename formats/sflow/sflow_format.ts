import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { GgufMetadata } from '../../core/model/gguf_parser.js';

export interface SFlowShardMetadata {
  shardId: string;
  layerStart: number;
  layerEnd: number;
  tensorCount: number;
  sizeBytes: number;
  targetDriveId: string;
  targetMountPoint: string;
  relFilePath: string;
  checksum: string;
  priority: 'HIGH' | 'MEDIUM' | 'LOW';
  accessFrequency: number;
}

export interface SFlowManifest {
  formatVersion: '1.0';
  modelName: string;
  architecture: string;
  parameterCountBillions: number;
  quantization: string;
  totalSizeBytes: number;
  blockCount: number;
  contextLength: number;
  createdAt: string;
  shards: SFlowShardMetadata[];
  tensorMap: Array<{
    name: string;
    layerIndex: number;
    shardId: string;
    offsetInShard: number;
    sizeBytes: number;
  }>;
}

export interface ShardValidation {
  valid: boolean;
  problems: string[];
  shardsPresent: number;
  shardsExpected: number;
  bytesOnDisk: number;
  bytesExpected: number;
}

export class SFlowContainer {
  public manifest: SFlowManifest;
  public sflowPath: string;

  constructor(sflowPath: string, manifest: SFlowManifest) {
    this.sflowPath = sflowPath;
    this.manifest = manifest;
  }

  /**
   * Check that every shard file exists and is at least as large as the
   * manifest claims. A container whose manifest advertises a 60 GB model while
   * the shards hold a few bytes is a stub, and callers must be able to tell.
   */
  public validateShards(containerDir?: string): ShardValidation {
    const baseDir = containerDir || path.dirname(this.sflowPath);
    const problems: string[] = [];
    let shardsPresent = 0;
    let bytesOnDisk = 0;
    let bytesExpected = 0;

    for (const shard of this.manifest.shards) {
      bytesExpected += shard.sizeBytes;
      const shardPath = path.isAbsolute(shard.relFilePath)
        ? shard.relFilePath
        : path.join(baseDir, shard.relFilePath);

      let size: number;
      try {
        size = fs.statSync(shardPath).size;
      } catch {
        problems.push(`Missing shard file: ${shard.relFilePath}`);
        continue;
      }

      shardsPresent++;
      bytesOnDisk += size;

      if (size < shard.sizeBytes) {
        problems.push(
          `Shard ${shard.shardId} holds ${size} bytes but the manifest declares ${shard.sizeBytes}.`
        );
      }
    }

    return {
      valid: problems.length === 0,
      problems,
      shardsPresent,
      shardsExpected: this.manifest.shards.length,
      bytesOnDisk,
      bytesExpected,
    };
  }

  public save(): void {
    const dir = path.dirname(this.sflowPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    const manifestPath = path.join(dir, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify(this.manifest, null, 2), 'utf8');
    fs.writeFileSync(this.sflowPath, JSON.stringify(this.manifest, null, 2), 'utf8');
  }

  public static load(sflowPath: string): SFlowContainer {
    const content = fs.readFileSync(sflowPath, 'utf8');
    const manifest: SFlowManifest = JSON.parse(content);
    return new SFlowContainer(sflowPath, manifest);
  }
}
