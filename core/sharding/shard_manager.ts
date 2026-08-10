import fs from 'fs';
import path from 'path';
import crypto from 'crypto';
import { GgufMetadata } from '../model/gguf_parser.js';
import { StorageDriveInfo } from '../hardware/types.js';
import { SFlowContainer, SFlowManifest, SFlowShardMetadata } from '../../formats/sflow/sflow_format.js';

export interface ShardProgressCallback {
  (progress: { percent: number; currentLayer: number; totalLayers: number; status: string; bytesCopied?: number }): void;
}

export interface ShardOptions {
  /**
   * Spread shard files across the storage fabric instead of keeping them all
   * under `outputDirectory`. This is what makes parallel multi-drive reads
   * possible: shards on the same physical disk contend for the same queue.
   */
  distributeAcrossDrives?: boolean;
  /** Directory name created on each target drive when distributing. */
  driveFolderName?: string;
}

const IO_CHUNK_SIZE = 8 * 1024 * 1024; // 8MB copy chunks for balanced speed/memory

/** Confirm a drive can actually be written to before shards are assigned to it. */
function isDriveWritable(mountPoint: string, folderName: string): string | null {
  try {
    const dir = path.join(mountPoint, folderName);
    fs.mkdirSync(dir, { recursive: true });
    const probe = path.join(dir, `.ailoflow_write_probe_${process.pid}`);
    fs.writeFileSync(probe, 'ok');
    fs.unlinkSync(probe);
    return dir;
  } catch {
    return null;
  }
}

export async function createShardedSFlowModel(
  modelMeta: GgufMetadata,
  drives: StorageDriveInfo[],
  outputDirectory: string,
  onProgress?: ShardProgressCallback,
  options: ShardOptions = {}
): Promise<SFlowContainer> {
  // Fastest drives first, so round-robin placement puts early (hot) layers on
  // the quickest devices.
  const sortedDrives = [...drives].sort((a, b) => {
    const sA = a.performanceProfile?.seqReadMBps || 0;
    const sB = b.performanceProfile?.seqReadMBps || 0;
    return sB - sA;
  });

  const folderName = options.driveFolderName || 'AiloFlowShards';

  // When distributing, only keep drives we can genuinely write to; a drive that
  // rejects writes must not silently receive shard assignments.
  const placements = new Map<string, string>(); // driveId -> absolute shard dir
  let targetDrives = sortedDrives;

  if (options.distributeAcrossDrives) {
    targetDrives = [];
    for (const drive of sortedDrives) {
      const dir = isDriveWritable(drive.mountPoint, path.join(folderName, modelMeta.filename.replace(/\.gguf$/i, '')));
      if (dir) {
        placements.set(drive.id, dir);
        targetDrives.push(drive);
      }
    }
    if (targetDrives.length === 0) {
      throw new Error('No storage device in the fabric is writable, so shards cannot be distributed.');
    }
  }

  const totalLayers = modelMeta.blockCount;
  const numDrives = Math.max(1, targetDrives.length);
  // Two shards per drive gives the scheduler room to interleave reads while
  // keeping each shard large enough for sequential throughput.
  const layersPerShard = Math.max(1, Math.ceil(totalLayers / (numDrives * 2)));

  // Sharding copies real tensor bytes; without the source GGUF there is
  // nothing to shard and we must not write placeholder files that would look
  // like a working container.
  if (!fs.existsSync(modelMeta.filePath)) {
    throw new Error(
      `Cannot shard "${modelMeta.filename}": source GGUF not found at ${modelMeta.filePath}.`
    );
  }

  // A large model is published across several files, and each tensor's offset
  // is relative to the file holding it. Reading everything from one descriptor
  // would copy the wrong bytes for every tensor outside the first part, so a
  // descriptor is kept per source file.
  const sourceFds = new Map<string, number>();
  const openSource = (file: string): number => {
    const existing = sourceFds.get(file);
    if (existing !== undefined) return existing;
    if (!fs.existsSync(file)) {
      throw new Error(`Cannot shard "${modelMeta.filename}": missing source part ${file}.`);
    }
    const fd = fs.openSync(file, 'r');
    sourceFds.set(file, fd);
    return fd;
  };

  const shards: SFlowShardMetadata[] = [];
  const tensorMap: SFlowManifest['tensorMap'] = [];

  // Create output directories
  const shardsDir = path.join(outputDirectory, 'shards');
  if (!fs.existsSync(shardsDir)) {
    fs.mkdirSync(shardsDir, { recursive: true });
  }

  let shardCounter = 0;
  let totalBytesCopied = 0;

  // Layer ranges to materialise. The leading (-1, -1) range collects tensors
  // that belong to no transformer block — token embeddings, the output head,
  // final norms — which must still live in the container for it to represent
  // the whole model.
  const ranges: Array<[number, number]> = [[-1, -1]];
  for (let l = 0; l < totalLayers; l += layersPerShard) {
    ranges.push([l, Math.min(totalLayers - 1, l + layersPerShard - 1)]);
  }

  try {
    for (const [layerStart, layerEnd] of ranges) {
      const driveIndex = shardCounter % targetDrives.length;
      const targetDrive = targetDrives[driveIndex];

      const isCommonShard = layerStart === -1;
      const shardId = isCommonShard
        ? `shard_${String(shardCounter).padStart(3, '0')}_common`
        : `shard_${String(shardCounter).padStart(3, '0')}_layer_${layerStart}_to_${layerEnd}`;

      // When distributing, the shard is written on its target device and the
      // manifest records an absolute path; otherwise everything stays local to
      // the container directory and the path stays relative (and portable).
      const driveDir = placements.get(targetDrive.id);
      const shardBinPath = driveDir
        ? path.join(driveDir, `${shardId}.bin`)
        : path.join(outputDirectory, 'shards', `${shardId}.bin`);
      const relFilePath = driveDir ? shardBinPath : `shards/${shardId}.bin`;

      // Get tensors belonging to this shard's layer range
      const layerTensors = modelMeta.tensors.filter((t) =>
        isCommonShard ? t.layerIndex < 0 : t.layerIndex >= layerStart && t.layerIndex <= layerEnd
      );
      if (layerTensors.length === 0) continue;
      const shardSizeBytes = layerTensors.reduce((sum, t) => sum + t.sizeBytes, 0);

      // Embeddings and the output head are touched by every token, so they rank
      // alongside the earliest layers; later blocks decay to LOW.
      const priority = isCommonShard || layerStart < 4
        ? 'HIGH'
        : layerStart < totalLayers * 0.5 ? 'MEDIUM' : 'LOW';

      // Write shard binary file — real data if source exists, header stub otherwise
      const shardHash = crypto.createHash('sha256');
      const shardFd = fs.openSync(shardBinPath, 'w');

      let offsetInShard = 0;
      for (const t of layerTensors) {
        // Copy the tensor's real bytes out of the GGUF and into the shard file,
        // reading from whichever part actually holds it.
        const sourceFd = openSource(t.sourceFile || modelMeta.filePath);
        let remaining = t.sizeBytes;
        let srcOffset = t.absoluteOffset;

        while (remaining > 0) {
          const chunkSize = Math.min(remaining, IO_CHUNK_SIZE);
          const chunk = Buffer.alloc(chunkSize);
          const bytesRead = fs.readSync(sourceFd, chunk, 0, chunkSize, srcOffset);
          if (bytesRead < chunkSize) {
            throw new Error(
              `Truncated read while sharding ${t.name}: expected ${chunkSize} bytes at ${srcOffset}, got ${bytesRead}.`
            );
          }
          fs.writeSync(shardFd, chunk, 0, chunkSize, offsetInShard + (t.sizeBytes - remaining));
          shardHash.update(chunk);
          remaining -= chunkSize;
          srcOffset += chunkSize;
          totalBytesCopied += chunkSize;
        }

        tensorMap.push({
          name: t.name,
          layerIndex: t.layerIndex,
          shardId,
          offsetInShard,
          sizeBytes: t.sizeBytes,
        });

        offsetInShard += t.sizeBytes;
      }

      fs.closeSync(shardFd);

      const checksum = shardHash.digest('hex').substring(0, 32);

      shards.push({
        shardId,
        layerStart,
        layerEnd,
        tensorCount: layerTensors.length,
        sizeBytes: shardSizeBytes,
        targetDriveId: targetDrive.id,
        targetMountPoint: targetDrive.mountPoint,
        relFilePath,
        checksum,
        priority,
        accessFrequency: priority === 'HIGH' ? 1.0 : priority === 'MEDIUM' ? 0.7 : 0.4,
      });

      shardCounter++;

      if (onProgress) {
        const percent = Number((Math.max(0, (layerEnd + 1) / totalLayers) * 100).toFixed(1));
        onProgress({
          percent,
          currentLayer: Math.max(0, layerEnd + 1),
          totalLayers,
          status: isCommonShard
            ? `Sharding embeddings and output head -> ${targetDrive.mountPoint}`
            : `Sharding layers ${layerStart}-${layerEnd} -> ${targetDrive.mountPoint}`,
          bytesCopied: totalBytesCopied,
        });
      }
    }
  } finally {
    for (const fd of sourceFds.values()) {
      try { fs.closeSync(fd); } catch { /* already closed */ }
    }
  }

  const manifest: SFlowManifest = {
    formatVersion: '1.0',
    modelName: modelMeta.filename,
    architecture: modelMeta.architecture,
    parameterCountBillions: modelMeta.parameterCountBillions,
    quantization: modelMeta.quantization,
    totalSizeBytes: modelMeta.fileSizeBytes,
    blockCount: modelMeta.blockCount,
    contextLength: modelMeta.contextLength,
    createdAt: new Date().toISOString(),
    shards,
    tensorMap,
  };

  const sflowFilePath = path.join(outputDirectory, `${modelMeta.filename.replace(/\.gguf$/i, '')}.sflow`);
  const container = new SFlowContainer(sflowFilePath, manifest);
  container.save();

  return container;
}
