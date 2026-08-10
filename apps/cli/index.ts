#!/usr/bin/env node
// Must come first: libuv fixes its threadpool size on first use, and every
// storage read depends on how many slots it has.
import '../../core/runtime/threadpool.js';
import { Command } from 'commander';
import path from 'path';
import { discoverComputerProfile, discoverStorage } from '../../core/hardware/discovery.js';
import { benchmarkStorageDrive } from '../../core/benchmark/storage_benchmark.js';
import { calculateOptimalConfiguration } from '../../core/hardware/optimizer.js';
import { parseGgufHeader, parseGgufModel } from '../../core/model/gguf_parser.js';
import { createShardedSFlowModel } from '../../core/sharding/shard_manager.js';
import { SFlowContainer } from '../../formats/sflow/sflow_format.js';
import { discoverAllModels, inspectModel } from '../../core/model/model_registry.js';
import { loadConfig, updateConfig, getConfigPath } from '../../core/config/config.js';
import { InferenceRegistry } from '../../inference/registry.js';
import { AiloStreamingPipeline } from '../../inference/custom_stream/stream_runner.js';
import { installEngine, listEngineCandidates, listInstalledEngines } from '../../core/engine/engine_installer.js';
import { MODEL_CATALOG, estimateWeightBytes, findCatalogEntry } from '../../core/models/catalog.js';
import {
  downloadGgufModel,
  getDownloadDirectory,
  getDownloadDirectoryFreeBytes,
  listRepoGgufFiles,
  searchGgufRepos,
} from '../../core/models/downloader.js';
import { benchmarkMemoryBandwidth, getStoredRamBandwidth } from '../../core/benchmark/memory_benchmark.js';
import { estimatePerformance, projectWithStorageBandwidth } from '../../core/estimator/performance_model.js';

const GB = 1024 ** 3;
const MB = 1024 ** 2;

const fmtBytes = (bytes: number | null): string =>
  bytes === null ? 'n/d' : bytes >= GB ? `${(bytes / GB).toFixed(2)} GB` : `${(bytes / MB).toFixed(1)} MB`;
const fmtNum = (n: number | null, suffix = '', decimals?: number): string =>
  n === null ? 'n/d' : `${decimals === undefined ? n : n.toFixed(decimals)}${suffix}`;

const program = new Command();

program
  .name('ailoflow')
  .description('Universal open-weight LLM runtime with dynamic streaming from storage')
  .version('1.0.0');

// ---------------------------------------------------------------------------
// system
// ---------------------------------------------------------------------------

const systemCmd = program.command('system').description('Hardware discovery and automatic tuning');

systemCmd
  .command('scan')
  .description('Detect CPU, RAM, GPU and storage, then show the automatic configuration')
  .action(async () => {
    console.log('\nScanning hardware...\n');
    const profile = await discoverComputerProfile();
    const config = calculateOptimalConfiguration(profile);

    const simd = Object.entries(profile.cpu.instructions)
      .filter(([, present]) => present)
      .map(([name]) => name.toUpperCase())
      .join(' ') || 'none detected';

    console.log('HARDWARE');
    console.log(`  CPU        ${profile.cpu.vendor} ${profile.cpu.model}`);
    console.log(`             ${fmtNum(profile.cpu.physicalCores)} cores / ${profile.cpu.logicalThreads} threads` +
      `${profile.cpu.baseFrequencyGHz ? ` @ ${profile.cpu.baseFrequencyGHz} GHz` : ''}`);
    console.log(`             SIMD: ${simd}  ->  kernel ${profile.cpu.recommendedOptimization}`);
    console.log(`  RAM        ${fmtBytes(profile.ram.totalBytes)} total, ${fmtBytes(profile.ram.freeBytes)} free` +
      `${profile.ram.speedMHz ? ` (${profile.ram.memoryType || 'RAM'} @ ${profile.ram.speedMHz} MHz x${profile.ram.moduleCount})` : ''}`);

    if (profile.gpus.length === 0) {
      console.log('  GPU        none detected — CPU inference only');
    } else {
      for (const gpu of profile.gpus) {
        const vram = gpu.vramTotalBytes === null
          ? 'VRAM unknown (not reported by the OS)'
          : `${fmtBytes(gpu.vramTotalBytes)} VRAM via ${gpu.vramSource}`;
        console.log(`  GPU        ${gpu.vendor} ${gpu.model} — ${vram}`);
        console.log(`             backends: ${gpu.supportedBackends.join(', ')} -> ${gpu.recommendedBackend}`);
      }
    }

    console.log(`  STORAGE    ${profile.storageDrives.length} drives, ${fmtBytes(profile.totalStorageBytes)} total`);
    console.log(`             measured read bandwidth: ${
      profile.measuredStorageReadBandwidthMBps === null
        ? 'not benchmarked (run "ailoflow storage benchmark")'
        : `${profile.measuredStorageReadBandwidthMBps} MB/s`
    }`);

    console.log('\nAUTOMATIC CONFIGURATION');
    console.log(`  RAM cache        ${fmtBytes(config.ramCacheBytes)}`);
    console.log(`  VRAM cache       ${config.vramCacheBytes > 0 ? fmtBytes(config.vramCacheBytes) : 'none (no usable VRAM figure)'}`);
    console.log(`  Threads          ${config.threadCount}`);
    console.log(`  Backend          ${config.backend}`);
    console.log(`  Prefetch depth   ${config.prefetchDepthLayers} layers`);
    console.log(`  I/O queue depth  ${config.ioQueueDepth}`);
    console.log(`  Shard strategy   ${config.shardingStrategy}`);
    if (!config.basedOnMeasuredStorage) {
      console.log('\n  Note: storage tuning is based on bus-type estimates. Run');
      console.log('        "ailoflow storage benchmark" to tune on measured throughput.');
    }
    console.log('');
  });

systemCmd
  .command('config')
  .description('Show or change persistent settings')
  .option('--set-llama-server <path>', 'Path to a llama.cpp llama-server binary')
  .option('--add-model-dir <dir>', 'Add a directory to scan for GGUF/.sflow models')
  .option('--port <port>', 'API server port')
  .action((options) => {
    const patch: Record<string, unknown> = {};
    if (options.setLlamaServer) patch.llamaServerPath = path.resolve(options.setLlamaServer);
    if (options.port) patch.apiPort = Number(options.port);
    if (options.addModelDir) {
      const dirs = new Set(loadConfig().modelDirectories);
      dirs.add(path.resolve(options.addModelDir));
      patch.modelDirectories = Array.from(dirs);
    }

    const config = Object.keys(patch).length ? updateConfig(patch) : loadConfig();
    console.log(`\nConfig file: ${getConfigPath()}\n`);
    console.log(JSON.stringify(config, null, 2));
    console.log('');
  });

// ---------------------------------------------------------------------------
// storage
// ---------------------------------------------------------------------------

const storageCmd = program.command('storage').description('Storage fabric inspection and benchmark');

storageCmd
  .command('scan')
  .description('List detected storage devices')
  .action(async () => {
    const drives = await discoverStorage();
    if (drives.length === 0) {
      console.log('\nNo storage devices detected.\n');
      return;
    }

    console.log('');
    for (const d of drives) {
      const p = d.performanceProfile;
      console.log(`[${d.mountPoint}] ${d.label} (${d.type})`);
      console.log(`    ${fmtBytes(d.totalSizeBytes)} total, ${fmtBytes(d.freeSizeBytes)} free, ${d.filesystem}`);
      if (p) {
        const tag = p.measured ? `measured ${p.measuredAt?.slice(0, 10)}` : 'ESTIMATE from bus type';
        console.log(`    read ${p.seqReadMBps} MB/s, latency ${p.latencyUs} us, ${p.iops} IOPS  [${tag}]`);
      }
      console.log('');
    }
  });

storageCmd
  .command('benchmark')
  .description('Run a non-destructive read/write/IOPS benchmark on every drive')
  .action(async () => {
    console.log('\nRunning non-destructive storage benchmark...\n');
    const drives = await discoverStorage();

    for (const drive of drives) {
      process.stdout.write(`  ${drive.mountPoint} (${drive.label})... `);
      try {
        const res = await benchmarkStorageDrive(drive, { sampleSizeBytes: 32 * 1024 * 1024, iterations: 3 });
        console.log('done');
        console.log(`      sequential read   ${res.seqReadMBps} MB/s`);
        console.log(`      sequential write  ${res.seqWriteMBps} MB/s`);
        console.log(`      random read (4K)  ${res.randReadMBps} MB/s`);
        console.log(`      latency           ${res.latencyUs} us`);
        console.log(`      IOPS              ${res.iops}`);

        if (res.readMethod === 'cold-file') {
          console.log(`      read measured against ${path.basename(res.coldReadSource || '')} (cache-cold)`);
        } else {
          console.log('      WARNING: no large pre-existing file was found on this volume, so the');
          console.log('      read figures come from the OS page cache and are an upper bound.');
          console.log(`      (page-cache read: ${res.cachedReadMBps} MB/s)`);
        }
        console.log('');
      } catch (err) {
        console.log(`FAILED: ${(err as Error).message}\n`);
      }
    }
  });

// ---------------------------------------------------------------------------
// model
// ---------------------------------------------------------------------------

const modelCmd = program.command('model').description('Model discovery, inspection and sharding');

modelCmd
  .command('list')
  .description('List every model AILOFlow can see (Ollama + configured directories)')
  .action(async () => {
    const models = await discoverAllModels();
    if (models.length === 0) {
      console.log('\nNo models found. Add a directory with "ailoflow system config --add-model-dir <dir>"');
      console.log('or start Ollama so its models become visible.\n');
      return;
    }

    console.log('');
    for (const m of models) {
      const runners = m.runnableWith.length ? m.runnableWith.join(', ') : 'no engine available';
      console.log(`  ${m.id}`);
      console.log(`      ${fmtBytes(m.fileSizeBytes)}  source: ${m.source}  runnable with: ${runners}`);
    }
    console.log('');
  });

modelCmd
  .command('inspect <model>')
  .description('Read real GGUF metadata (accepts a registry id or a file path)')
  .action(async (model: string) => {
    try {
      const meta = model.includes(':') && !path.isAbsolute(model)
        ? await inspectModel(model)
        : await inspectModel(`gguf:${path.resolve(model)}`).catch(() => {
            const parsed = parseGgufHeader(path.resolve(model));
            return {
              id: model,
              source: 'gguf' as const,
              filePath: parsed.filePath,
              fileSizeBytes: parsed.fileSizeBytes,
              architecture: parsed.architecture,
              modelName: parsed.modelName,
              parameterCountBillions: parsed.parameterCountBillions,
              quantization: parsed.quantization,
              blockCount: parsed.blockCount,
              contextLength: parsed.contextLength,
              embeddingLength: parsed.embeddingLength,
              headCount: parsed.headCount,
              headCountKv: parsed.headCountKv,
              tensorCount: parsed.tensorCount,
              totalTensorDataBytes: parsed.totalTensorDataBytes,
              estimatedRamRequiredBytes: parsed.estimatedRamRequiredBytes,
              estimatedVramRequiredBytes: parsed.estimatedVramRequiredBytes,
              estimatedStorageRequiredBytes: parsed.estimatedStorageRequiredBytes,
              estimatedKvCacheBytes: 0,
            };
          });

      console.log('');
      console.log(`  Name           ${meta.modelName}`);
      console.log(`  File           ${meta.filePath}`);
      console.log(`  Architecture   ${meta.architecture}`);
      console.log(`  Parameters     ${meta.parameterCountBillions}B`);
      console.log(`  Quantization   ${meta.quantization}`);
      console.log(`  Layers         ${meta.blockCount}`);
      console.log(`  Context        ${meta.contextLength}`);
      console.log(`  Tensors        ${meta.tensorCount}`);
      console.log(`  Size on disk   ${fmtBytes(meta.fileSizeBytes)}`);
      console.log(`  Weights in RAM ${fmtBytes(meta.estimatedRamRequiredBytes)}`);
      console.log(`  KV cache @ ctx ${fmtBytes(meta.estimatedKvCacheBytes)}`);
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

modelCmd
  .command('shard <modelPath>')
  .description('Copy a GGUF into a .sflow container distributed across the storage fabric')
  .option('-o, --output <dir>', 'Output directory for the manifest')
  .option('-d, --distribute', 'Write shards across every writable drive instead of one folder')
  .action(async (modelPath: string, options: { output?: string; distribute?: boolean }) => {
    try {
      const meta = parseGgufModel(path.resolve(modelPath));
      const drives = await discoverStorage();
      const outDir = options.output
        ? path.resolve(options.output)
        : path.join(process.cwd(), 'sflow-models', meta.filename.replace(/\.gguf$/i, ''));

      console.log(`\nSharding ${meta.filename} (${fmtBytes(meta.fileSizeBytes)}) into ${outDir}`);
      console.log(options.distribute ? 'Distributing shards across the storage fabric.\n' : '');

      const container = await createShardedSFlowModel(
        meta,
        drives,
        outDir,
        (p) => process.stdout.write(`\r  ${p.percent}%  ${p.status}   `),
        { distributeAcrossDrives: options.distribute === true }
      );

      const validation = container.validateShards(outDir);
      const byDrive = new Map<string, number>();
      for (const shard of container.manifest.shards) {
        byDrive.set(shard.targetMountPoint, (byDrive.get(shard.targetMountPoint) || 0) + shard.sizeBytes);
      }

      console.log(`\n\n  Container: ${container.sflowPath}`);
      console.log(`  Shards:    ${container.manifest.shards.length}`);
      console.log('  Placement:');
      for (const [mount, bytes] of byDrive.entries()) {
        console.log(`    ${mount}  ${fmtBytes(bytes)}`);
      }
      console.log(`  Integrity: ${validation.valid ? 'OK' : `FAILED\n    ${validation.problems.join('\n    ')}`}`);
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

modelCmd
  .command('validate <sflowPath>')
  .description('Verify that a .sflow container is backed by complete shard data')
  .action((sflowPath: string) => {
    try {
      const container = SFlowContainer.load(path.resolve(sflowPath));
      const v = container.validateShards();
      console.log('');
      console.log(`  Model      ${container.manifest.modelName}`);
      console.log(`  Shards     ${v.shardsPresent}/${v.shardsExpected} present`);
      console.log(`  Data       ${fmtBytes(v.bytesOnDisk)} on disk vs ${fmtBytes(v.bytesExpected)} declared`);
      console.log(`  Status     ${v.valid ? 'VALID' : 'INVALID'}`);
      for (const p of v.problems.slice(0, 10)) console.log(`    - ${p}`);
      if (v.problems.length > 10) console.log(`    ... and ${v.problems.length - 10} more`);
      console.log('');
      if (!v.valid) process.exitCode = 1;
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

// ---------------------------------------------------------------------------
// engine
// ---------------------------------------------------------------------------

const engineCmd = program.command('engine').description("Manage AILOFlow's own llama.cpp build");

engineCmd
  .command('list')
  .description('Show installed engines and the builds available for this hardware')
  .action(async () => {
    const installed = listInstalledEngines();
    console.log('');
    if (installed.length === 0) {
      console.log('  No engine installed yet.\n');
    } else {
      for (const engine of installed) {
        console.log(`  ${engine.variant.toUpperCase()} ${engine.release}`);
        console.log(`      ${engine.serverPath}`);
      }
      console.log('');
    }

    try {
      const profile = await discoverComputerProfile();
      const { release, candidates } = await listEngineCandidates(profile);
      console.log(`  Available in release ${release}:`);
      for (const c of candidates) {
        console.log(`    ${c.variant.padEnd(7)} ${fmtBytes(c.sizeBytes).padStart(9)}  ${c.rationale}`);
      }
      console.log('');
    } catch (err) {
      console.error(`  ${(err as Error).message}\n`);
    }
  });

engineCmd
  .command('install [variant]')
  .description('Download and install an engine build (cuda, hip, vulkan, metal, cpu)')
  .action(async (variant?: string) => {
    try {
      const profile = await discoverComputerProfile();
      const { release, candidates } = await listEngineCandidates(profile);
      const candidate = variant ? candidates.find((c) => c.variant === variant) : candidates[0];
      if (!candidate) throw new Error(`No build available for variant "${variant}".`);

      console.log(`\nInstalling ${candidate.variant.toUpperCase()} build from release ${release}\n`);
      const installed = await installEngine(candidate, release, (p) => {
        const label = p.phase === 'download' ? `${p.percent}%  ${fmtBytes(p.bytesPerSecond)}/s` : p.phase;
        process.stdout.write(`\r  ${p.assetName}: ${label}          `);
      });

      console.log(`\n\n  Installed: ${installed.serverPath}`);
      console.log(`  ${installed.version || 'version unknown'}\n`);
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

// ---------------------------------------------------------------------------
// catalog / download
// ---------------------------------------------------------------------------

const catalogCmd = program.command('catalog').description('Browse and download open-weight models');

catalogCmd
  .command('list')
  .description('Show the curated list of the largest open-weight models')
  .action(() => {
    console.log(`\n  Download directory: ${getDownloadDirectory()} (${fmtBytes(getDownloadDirectoryFreeBytes())} free)\n`);
    for (const entry of MODEL_CATALOG) {
      const q4 = estimateWeightBytes(entry.totalParamsB, 'Q4_K_M');
      console.log(`  ${entry.id.padEnd(20)} ${entry.name}`);
      console.log(
        `      ${entry.totalParamsB}B totali / ${entry.activeParamsB}B attivi (${entry.architecture}) — ${fmtBytes(q4)} in Q4_K_M`
      );
    }
    console.log('');
  });

catalogCmd
  .command('search <query>')
  .description('Search Hugging Face for GGUF repositories')
  .action(async (query: string) => {
    try {
      const results = await searchGgufRepos(query);
      console.log('');
      for (const r of results) console.log(`  ${r.repoId.padEnd(60)} ${r.downloads.toLocaleString()} download`);
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

catalogCmd
  .command('files <repoId>')
  .description('List the GGUF files published by a repository')
  .action(async (repoId: string) => {
    try {
      const files = await listRepoGgufFiles(repoId);
      console.log('');
      for (const f of files) {
        console.log(
          `  ${fmtBytes(f.totalSizeBytes).padStart(10)}  ${f.path}${f.splitParts ? `  (${f.splitParts.length} parti)` : ''}`
        );
      }
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

catalogCmd
  .command('download <repoId> <filePath>')
  .description('Download a GGUF, including every part of a split set')
  .action(async (repoId: string, filePath: string) => {
    try {
      const files = await listRepoGgufFiles(repoId);
      const file = files.find((f) => f.path === filePath || f.path.endsWith(`/${filePath}`));
      if (!file) throw new Error(`File "${filePath}" not found in ${repoId}.`);

      console.log(`\nDownloading ${fmtBytes(file.totalSizeBytes)} into ${getDownloadDirectory()}\n`);
      const result = await downloadGgufModel(repoId, file, (p) => {
        const eta = p.etaSeconds !== null ? `ETA ${Math.floor(p.etaSeconds / 60)}m${p.etaSeconds % 60}s` : '';
        process.stdout.write(
          `\r  ${p.percent.toFixed(1)}%  part ${p.partIndex}/${p.totalParts}  ${fmtBytes(p.bytesPerSecond)}/s  ${eta}        `
        );
      });

      console.log(`\n\n  Saved ${result.files.length} file(s), ${fmtBytes(result.totalBytes)} in ${(result.durationMs / 1000).toFixed(0)}s`);
      console.log(`  ${result.primaryPath}\n`);
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

// ---------------------------------------------------------------------------
// estimate
// ---------------------------------------------------------------------------

program
  .command('estimate')
  .description('Predict decode speed for a large model on this machine')
  .option('-c, --catalog <id>', 'Catalog entry id (see "ailoflow catalog list")')
  .option('-p, --params <b>', 'Total parameters in billions')
  .option('-a, --active <b>', 'Active parameters per token in billions (MoE)')
  .option('-q, --quant <name>', 'Quantization', 'Q4_K_M')
  .option('-l, --layers <n>', 'Layer count', '96')
  .option('--ctx <n>', 'Context length in use', '8192')
  .action(async (options) => {
    try {
      const profile = await discoverComputerProfile();
      const entry = options.catalog ? findCatalogEntry(options.catalog) : undefined;
      if (options.catalog && !entry) throw new Error(`Unknown catalog id "${options.catalog}".`);

      const totalParamsB = entry ? entry.totalParamsB : Number(options.params);
      if (!totalParamsB) throw new Error('Provide --catalog or --params.');
      const activeParamsB = entry ? entry.activeParamsB : Number(options.active || totalParamsB);

      const spec = {
        name: entry ? entry.name : `${totalParamsB}B`,
        totalParamsB,
        activeParamsB,
        quantization: options.quant,
        layers: Number(options.layers),
        contextLength: entry ? entry.contextLength : 131072,
        architecture: (activeParamsB < totalParamsB ? 'moe' : 'dense') as 'moe' | 'dense',
      };

      const measured = profile.storageDrives.filter((d) => d.performanceProfile?.measured);
      const inputs = {
        profile,
        ramBandwidthMBps: getStoredRamBandwidth(),
        storageBandwidthMBps: measured.length
          ? measured.reduce((sum, d) => sum + (d.performanceProfile?.seqReadMBps || 0), 0)
          : null,
        storageBandwidthIsCacheInfluenced: measured.some((d) => d.performanceProfile?.cacheInfluenced),
        contextUsed: Number(options.ctx),
      };

      const e = estimatePerformance(spec, inputs);

      console.log(`\n  ${spec.name} — ${spec.quantization}\n`);
      console.log(`  Peso totale          ${fmtBytes(e.totalWeightBytes)}`);
      console.log(`  Byte per token       ${fmtBytes(e.activeBytesPerToken)}`);
      console.log(`  KV cache             ${fmtBytes(e.kvCacheBytes)}`);
      console.log(`  Entra in memoria     ${e.fitsInMemory ? 'sì' : 'no'}`);
      console.log('');
      for (const p of e.placement) {
        console.log(
          `  ${p.tier.padEnd(8)} residenti ${fmtBytes(p.residentBytes).padStart(10)}  ` +
            `per token ${fmtBytes(p.bytesPerToken).padStart(10)}  ` +
            `banda ${p.bandwidthMBps ? `${(p.bandwidthMBps / 1024).toFixed(1)} GB/s` : 'n/d'}`
        );
      }
      console.log('');
      console.log(`  Stima sequenziale    ${fmtNum(e.sequentialTokensPerSecond, ' tok/s', 3)}`);
      console.log(`  Stima sovrapposta    ${fmtNum(e.overlappedTokensPerSecond, ' tok/s', 3)}`);
      console.log(`  Collo di bottiglia   ${e.bottleneckTier || 'n/d'}`);
      console.log(`  Confidenza           ${e.confidence}`);
      console.log('');

      for (const mbps of [1600, 7000, 14000, 28000, 56000]) {
        const p = projectWithStorageBandwidth(spec, inputs, mbps);
        console.log(`    storage ${(mbps / 1024).toFixed(1).padStart(5)} GB/s -> ${fmtNum(p.overlappedTokensPerSecond, ' tok/s', 3)}`);
      }
      console.log('');
      for (const a of e.assumptions) console.log(`  · ${a}`);
      for (const w of e.warnings) console.log(`  ! ${w}`);
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    }
  });

program
  .command('membench')
  .description('Measure system RAM bandwidth (input for the estimator)')
  .action(() => {
    const r = benchmarkMemoryBandwidth();
    console.log(`\n  Traffico memoria  ${(r.bandwidthMBps / 1024).toFixed(1)} GB/s`);
    console.log(`  Copia             ${(r.copyMBps / 1024).toFixed(1)} GB/s`);
    console.log(`  Buffer            ${fmtBytes(r.bufferSizeBytes)} x ${r.iterations} iterazioni`);
    console.log('\n  Nota: memcpy a thread singolo, quindi conservativa rispetto\n  alla banda aggregata che un motore multi-thread raggiunge.\n');
  });

// ---------------------------------------------------------------------------
// run / benchmark
// ---------------------------------------------------------------------------

program
  .command('run <modelId>')
  .description('Generate with a real engine (model id from "ailoflow model list")')
  .option('-p, --prompt <text>', 'Prompt to send', 'Spiega in breve cosa fa AILOFlow.')
  .option('-s, --system <text>', 'System prompt')
  .option('-t, --temperature <value>', 'Sampling temperature', '0.7')
  .action(async (modelId: string, options: { prompt: string; system?: string; temperature: string }) => {
    const registry = new InferenceRegistry();
    try {
      const state = await registry.loadModel(modelId);
      console.log(`\nEngine: ${state.backendName} — loaded in ${state.loadDurationMs} ms\n`);

      const backend = registry.getActiveBackend()!;
      const result = await backend.generateStream(
        { prompt: options.prompt, systemPrompt: options.system, temperature: Number(options.temperature) },
        (token) => process.stdout.write(token.token)
      );

      const m = result.metrics;
      console.log('\n');
      console.log(`  ${fmtNum(m.completionTokens)} tokens in ${(m.totalDurationMs / 1000).toFixed(2)}s`);
      console.log(`  ${fmtNum(m.tokensPerSecond, ' tok/s')} generation, ${fmtNum(m.promptTokensPerSecond, ' tok/s')} prompt`);
      console.log(`  first token after ${fmtNum(m.firstTokenLatencyMs, ' ms')}\n`);
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    } finally {
      await registry.unload();
    }
  });

program
  .command('benchmark <sflowPath>')
  .description('Stream every layer of a .sflow container off storage and measure the pipeline')
  .action(async (sflowPath: string) => {
    const profile = await discoverComputerProfile();
    const pipeline = new AiloStreamingPipeline(profile);

    try {
      await pipeline.load(path.resolve(sflowPath));
      console.log('\nStreaming layers through storage fabric -> cache -> prefetch...\n');

      const result = await pipeline.runLayerSweep((p) => {
        process.stdout.write(
          `\r  layer ${p.layerIndex + 1}/${p.totalLayers}  ${p.bandwidthMBps} MB/s  cache hit ${p.cacheHitRatePercent}%   `
        );
      });

      console.log('\n');
      console.log(`  Model                 ${result.modelName}`);
      console.log(`  Layers streamed       ${result.layersCompleted}/${result.totalLayers}`);
      console.log(`  Read from storage     ${fmtBytes(result.bytesReadFromStorage)} of ${fmtBytes(result.totalBytesRequested)} requested`);
      console.log(`  Duration              ${(result.durationMs / 1000).toFixed(2)} s`);
      console.log(`  Effective bandwidth   ${result.effectiveBandwidthMBps} MB/s`);
      console.log(`  Per-request rate      ${result.averageRequestMBps} MB/s (single read, reads overlap)`);
      console.log(`  Avg layer latency     ${result.averageLayerLatencyMs} ms`);
      console.log(`  Storage ceiling       ${result.sustainableTokensPerSecond} token/s at this bandwidth`);
      if (result.cacheInfluenced) {
        console.log('');
        console.log('  WARNING: the shard set is smaller than free RAM, so these reads were');
        console.log('  most likely served by the OS page cache. The figures above are an');
        console.log('  upper bound, not the speed of the physical drives.');
      }
      console.log(`  Cache hit rate        ${result.cacheHitRatePercent}% (${result.cacheHits} hits / ${result.cacheMisses} misses)`);
      console.log(
        `  Prefetch triggered    ${result.prefetchTriggered} ` +
          `(depth ${result.prefetchActiveDepth}/${result.prefetchDepth}` +
          `${result.prefetchThrottled ? ', reduced by memory pressure' : ''})`
      );
      console.log(`  Drives used           ${result.drivesUsed.join(', ') || 'none'}`);
      console.log(`  Weight bytes/token    ${fmtBytes(result.bytesPerTokenStreamed)}`);
      if (result.errors.length) {
        console.log(`\n  ${result.errors.length} read errors, first few:`);
        for (const e of result.errors.slice(0, 5)) console.log(`    - ${e}`);
      }
      console.log('');
    } catch (err) {
      console.error(`\n  ${(err as Error).message}\n`);
      process.exitCode = 1;
    } finally {
      pipeline.dispose();
    }
  });

program.parse(process.argv);
