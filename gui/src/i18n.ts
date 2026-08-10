import { createContext, useContext } from 'react';

/**
 * Minimal translation layer.
 *
 * English is the default because the UI ships to a general audience; Italian is
 * a full translation rather than a partial one, so switching never leaves a
 * screen half-translated. Missing keys fall back to English and are visible in
 * the console during development instead of rendering as blank.
 */

export type Language = 'en' | 'it';

export const LANGUAGE_STORAGE_KEY = 'ailoflow.language';

const en = {
  // Navigation and shell
  'nav.dashboard': 'Dashboard',
  'nav.storage': 'Storage Fabric',
  'nav.models': 'Models',
  'nav.catalog': 'Catalog',
  'nav.chat': 'Chat',
  'nav.settings': 'Settings',
  'nav.noModel': 'no model loaded',
  'nav.running': 'running: {name}',
  'conn.online': 'API CONNECTED',
  'conn.connecting': 'CONNECTING...',
  'conn.offline': 'API UNREACHABLE',

  // Shared
  'common.notAvailable': 'n/a',
  'common.loading': 'Loading...',
  'common.error': 'Error',
  'common.refresh': 'Refresh',
  'common.cancel': 'Cancel',
  'common.measured': 'measured',
  'common.estimate': 'estimate',
  'common.none': 'None',
  'common.close': 'Close',

  // Dashboard
  'dash.offlineTitle': 'AILOFlow API unreachable',
  'dash.offlineBody':
    'The interface does not invent data: until the server responds, no value is shown. Start the runtime with "npm run server" in the project folder.',
  'dash.loadedModel': 'LOADED MODEL',
  'dash.chooseModel': 'Pick a model in the Models tab to begin.',
  'dash.loadedIn': 'loaded in {duration}',
  'dash.genSpeed': 'GENERATION SPEED',
  'dash.genSpeedHint': 'tokens/s · measured {when}',
  'dash.noGeneration': 'no generation run yet',
  'dash.firstToken': 'FIRST TOKEN LATENCY',
  'dash.promptRate': 'prompt {rate} tok/s',
  'dash.awaitingData': 'awaiting data',
  'dash.cache': 'HIERARCHICAL CACHE',
  'dash.cacheHint': '{hits} hits / {misses} misses',
  'dash.cacheIdle': 'active only during .sflow streaming',
  'dash.hardware': 'LIVE HARDWARE MONITORING',
  'dash.sampledAt': 'sampled {time}',
  'dash.gpus': 'DETECTED GRAPHICS CARDS',
  'dash.noGpu': 'No GPU detected: inference will use the CPU.',
  'dash.utilization': 'Utilization',
  'dash.utilizationShared': 'Utilization (system-wide)',
  'dash.vramSource': 'source: {source}',
  'dash.counters': 'counters: {source}',
  'dash.autoConfig': 'AUTOMATIC CONFIGURATION',
  'dash.backend': 'Selected backend',
  'dash.ramCache': 'RAM dedicated to cache',
  'dash.vramCache': 'VRAM dedicated',
  'dash.vramNone': 'none (VRAM unknown)',
  'dash.threads': 'Worker threads',
  'dash.prefetch': 'Prefetch depth',
  'dash.prefetchLayers': '{n} layers',
  'dash.ioQueue': 'I/O queue depth',
  'dash.shardStrategy': 'Sharding strategy',
  'dash.tunedMeasured': 'tuning computed on measured bandwidth',
  'dash.tunedEstimated': 'tuning on bus-type estimates: run the storage benchmark',
  'dash.pipeline': '.SFLOW STORAGE PIPELINE',
  'dash.pipelineEmpty':
    'No measurement available. The streaming pipeline is measured from the Models tab, by building a .sflow container and running the layer sweep: the values above come only from real disk reads.',
  'dash.pipelineModel': 'Measured container',
  'dash.pipelineLayers': 'Layers transferred',
  'dash.pipelineRead': 'Read from storage',
  'dash.pipelineBandwidth': 'Effective bandwidth',
  'dash.pipelineLatency': 'Average layer latency',
  'dash.pipelineCache': 'Cache hit rate',
  'dash.pipelinePerToken': 'Weight bytes per token',
  'dash.pipelineDrives': 'Drives involved',
  'dash.cpuNote': '{threads} threads · {simd}',
  'dash.gpuNoCounter': 'no readable GPU counter',
  'dash.pipelineIdle': '.sflow pipeline idle',
  'dash.storageHint': '{iops} IOPS · queue {queue}',

  // Storage
  'storage.title': 'AILO STORAGE FABRIC',
  'storage.summary': '{count} devices · {total} total · measured bandwidth {bandwidth}',
  'storage.notMeasured': 'not measured yet',
  'storage.benchmarkAll': 'Benchmark every drive',
  'storage.benchmarking': 'Benchmarking...',
  'storage.benchmarkOne': 'Benchmark this drive',
  'storage.measuring': 'Measuring...',
  'storage.noDrives': 'No storage device detected',
  'storage.noDrivesBody':
    'The operating system returned no usable volumes. No fictional disk is shown in their place.',
  'storage.detecting': 'Detecting devices...',
  'storage.usage': 'Usage',
  'storage.seqRead': 'Sequential read',
  'storage.seqWrite': 'Sequential write',
  'storage.randRead': 'Random read 4K',
  'storage.latency': 'Latency',
  'storage.iops': 'IOPS',
  'storage.free': 'Free space',
  'storage.estimateWarning':
    'These values are estimates derived from the bus type, not measurements. Run the benchmark for real figures.',
  'storage.cacheWarning':
    'The read figures were served by the OS page cache, so they are an upper bound rather than device speed.',
  'storage.benchmarkDone': 'Benchmark completed in {duration} · {time}',

  // Models
  'models.detected': 'DETECTED MODELS',
  'models.scanning': 'Scanning Ollama and the configured folders...',
  'models.empty':
    'No model found. Add a folder containing .gguf files from Settings, or download one from the Catalog.',
  'models.noEngine': 'no engine',
  'models.selectTitle': 'Select a model',
  'models.selectBody':
    'The metadata shown here is read straight from the binary GGUF header: architecture, layers, quantization and memory requirements are computed from the real tensors.',
  'models.noLocalFile': 'no local file associated',
  'models.load': 'Load for inference',
  'models.shard': 'Shard into .sflow',
  'models.measurePipeline': 'Measure pipeline',
  'models.noEngineHint': 'No inference engine can run this model.',
  'models.readingHeader': 'Reading GGUF header...',
  'models.architecture': 'Architecture',
  'models.parameters': 'Parameters',
  'models.quantization': 'Quantization',
  'models.layers': 'Layers',
  'models.context': 'Context length',
  'models.tensors': 'Tensors',
  'models.fileSize': 'File size',
  'models.weightsInRam': 'Weights in RAM',
  'models.kvCache': 'KV cache (full context)',
  'models.heads': 'Heads / KV heads',
  'models.notInspectable':
    'This model exposes no readable GGUF header, so no metadata is shown: nothing is estimated in its place.',
  'models.operationLog': 'OPERATION LOG',
  'models.pipelineResult': 'REAL PIPELINE MEASUREMENT',
  'models.sweepErrors': '{count} read errors during the sweep: {first}',

  // Catalog
  'catalog.title': 'OPEN-WEIGHT CATALOG',
  'catalog.filterPlaceholder': 'Filter by name, publisher or capability (coding, reasoning...)',
  'catalog.noMatch': 'No model matches the filter.',
  'catalog.all': 'all',
  'catalog.gigante': 'giant',
  'catalog.grande': 'large',
  'catalog.medio': 'medium',
  'catalog.piccolo': 'small',
  'catalog.searchPlaceholder': 'Search Hugging Face (e.g. Kimi-K2 GGUF)',
  'catalog.noRepos': 'No GGUF repository found.',
  'catalog.downloads': '{count} downloads',
  'catalog.directory': 'download folder · {free} free',
  'catalog.spaceHint':
    'A 300B model in Q4 takes about 180 GB. Pick a volume with enough room from Settings before starting.',
  'catalog.estimate': 'Estimate',
  'catalog.computing': 'Computing on measured hardware...',
  'catalog.license': 'License: {license} · context {context} tokens',
  'catalog.totalParams': '{n}B total',
  'catalog.activeParams': '{n}B active',
  'catalog.inQ4': '{size} in Q4',
  'catalog.filesIn': 'GGUF files in {repo}',
  'catalog.readingRepo': 'Reading repository...',
  'catalog.noGgufFiles': 'No .gguf file in this repository.',
  'catalog.download': 'Download',
  'catalog.parts': '{n} parts',
  'catalog.emptyTitle': 'Pick a model on the left',
  'catalog.emptyBody':
    'You will see how much space it needs and how fast it would run on this computer, computed from the measured memory and storage bandwidth. Files are then downloaded straight from Hugging Face, resuming automatically if the connection drops.',

  // Estimate
  'est.totalWeight': 'Total weight',
  'est.bytesPerToken': 'Bytes per token',
  'est.kvCache': 'KV cache',
  'est.fitsInMemory': 'Fits in memory',
  'est.fitsYes': 'yes',
  'est.fitsNo': 'no, needs storage',
  'est.overlapped': 'Estimate (overlapped)',
  'est.bottleneck': 'Bottleneck',
  'est.placement': 'WEIGHT PLACEMENT',
  'est.resident': '{size} resident',
  'est.perToken': '{size}/token',
  'est.noBandwidth': 'bandwidth n/a',
  'est.projections': 'IF YOU IMPROVE STORAGE',
  'est.confidence': 'Confidence',

  // Chat
  'chat.noModelTitle': 'No model loaded',
  'chat.noModelBody':
    'The chat talks to a real inference engine (llama.cpp or Ollama): until you load a model from the Models tab, no text is generated.',
  'chat.placeholder': 'Write a message for the local model...',
  'chat.send': 'Send',
  'chat.stop': 'Stop',
  'chat.clear': 'Clear',
  'chat.intro':
    'Write a message: tokens stream from the local engine, and the metrics under each reply are the ones the engine itself reported.',
  'chat.reasoning': "Model's reasoning ({count} characters)",
  'chat.thinking': 'thinking...',
  'chat.parameters': 'PARAMETERS',
  'chat.systemPrompt': 'System prompt',
  'chat.systemNone': '(none)',
  'chat.temperature': 'Temperature',
  'chat.topP': 'Top-P',
  'chat.topK': 'Top-K',
  'chat.repeatPenalty': 'Repetition penalty',
  'chat.maxTokens': 'Max tokens',
  'chat.unlimitedTokens': 'No token limit (recommended for reasoning models)',
  'chat.contextLength': 'Context length',
  'chat.seed': 'Seed (empty = random)',
  'chat.seedRandom': 'random',
  'chat.stopSequences': 'Stop sequences (comma separated)',
  'chat.tokensPerSec': '{rate} tok/s',
  'chat.tokensNa': 'tok/s n/a',
  'chat.tokenCount': '{count} tokens',

  // Downloads
  'dl.active': 'Downloads',
  'dl.none': 'No download in progress.',
  'dl.part': 'part {index}/{total}',
  'dl.remaining': '{time} remaining',
  'dl.completedIn': 'Completed. The model appears in the Models tab.',
  'dl.cancelled': 'Cancelled',
  'dl.failed': 'Failed',
  'dl.running': 'Downloading',
  'dl.retrying': 'Connection lost, retrying (attempt {attempt})',
  'dl.completed': 'Completed',
  'dl.clearFinished': 'Clear finished',
  'dl.keepsRunning': 'Downloads keep running when you switch tabs or close this page.',

  // Settings
  'settings.engines': 'INFERENCE ENGINES',
  'settings.engineTitle': 'AILOFLOW ENGINE',
  'settings.engineNone':
    'No engine installed. AILOFlow can download the llama.cpp build matching the detected GPU on its own: from then on inference runs as a child process of the runtime, with no external daemon.',
  'settings.findBuilds': 'Find available builds',
  'settings.checkUpdates': 'Check for updates',
  'settings.release': 'Release {release} — builds ordered by fit for your hardware',
  'settings.install': 'Install',
  'settings.downloadPhase': 'Download',
  'settings.extractPhase': 'Extracting',
  'settings.verifyPhase': 'Verifying',
  'settings.llamaPath': 'Path to the llama-server binary (llama.cpp)',
  'settings.llamaPathHint':
    'Needed to run .gguf files directly. Without a binary, local GGUF models are reported as not runnable instead of appearing available.',
  'settings.save': 'Save',
  'settings.ollamaUrl': 'Ollama daemon URL',
  'settings.gpuBackend': 'GPU backend',
  'settings.automatic': 'Automatic ({backend})',
  'settings.useOllama': 'Use the Ollama daemon as an inference engine',
  'settings.modelFolders': 'MODEL FOLDERS',
  'settings.noFolders': 'No folder configured.',
  'settings.folderPlaceholder': 'Absolute path of a folder containing .gguf files',
  'settings.add': 'Add',
  'settings.runtime': 'RUNTIME',
  'settings.apiPort': 'API port (restart required)',
  'settings.gpuLayers': 'GPU layers (empty = automatic)',
  'settings.contextLength': 'Context window in tokens (empty = automatic, reload the model to apply)',
  'settings.contextLengthHint':
    "Automatic uses the model's own maximum, capped at 32768 so the KV cache stays affordable. IDE assistants send system prompts of 10k tokens and more, so a small window makes them fail.",
  'settings.auto': 'auto',
  'settings.privacy':
    'No prompt and no model leaves this machine: the API listens on 127.0.0.1 only and there is no remote telemetry.',
  'settings.saved': 'Settings saved.',
  'settings.language': 'Interface language',
  'settings.downloadFolder': 'Download folder',
  'settings.llamaPathPlaceholder': 'e.g. C:\\llama.cpp\\build\\bin\\llama-server.exe',

  // Sharding log lines
  'log.shardStart': 'Start: {model} · {layers} layers · {size} → {directory}',
  'log.shardProgress': '{percent}% — {status} ({copied} copied)',
  'log.shardDone': 'Done: {shards} shards in {path}',
  'log.shardDrives': 'Shards physically written to: {drives}',
  'log.integrityOk': 'Integrity verified: every shard matches the manifest.',
  'log.integrityFailed': 'Integrity FAILED: {problems}',
  'log.sweepLayer': 'layer {layer}/{total} · {bandwidth} MB/s · cache {cache}%',

  // Model manager extras
  'models.noEngineForModel': 'No engine available for this model',
  'models.splitParts': '{n} files',
  'models.incompleteSet':
    'This model is published as a split file set and some parts are still missing, so it cannot be loaded yet. Finish the download from the Catalog tab.',
  'models.sweepLayers': 'Layers transferred',
  'models.sweepRead': 'Read from disk',
  'models.sweepBandwidth': 'Effective bandwidth',
  'models.sweepDuration': 'Duration',
  'models.sweepLatency': 'Average latency/layer',
  'models.sweepCache': 'Cache hit',
  'models.sweepPrefetch': 'Prefetches triggered',
  'models.sweepPerToken': 'Weight per token',
  'models.context2': 'Context length',
} as const;

export type TranslationKey = keyof typeof en;

const it: Record<TranslationKey, string> = {
  'nav.dashboard': 'Dashboard',
  'nav.storage': 'Storage Fabric',
  'nav.models': 'Modelli',
  'nav.catalog': 'Catalogo',
  'nav.chat': 'Chat',
  'nav.settings': 'Impostazioni',
  'nav.noModel': 'nessun modello caricato',
  'nav.running': 'in esecuzione: {name}',
  'conn.online': 'API CONNESSA',
  'conn.connecting': 'CONNESSIONE...',
  'conn.offline': 'API NON RAGGIUNGIBILE',

  'common.notAvailable': 'n/d',
  'common.loading': 'Caricamento...',
  'common.error': 'Errore',
  'common.refresh': 'Aggiorna',
  'common.cancel': 'Annulla',
  'common.measured': 'misurato',
  'common.estimate': 'stima',
  'common.none': 'Nessuno',
  'common.close': 'Chiudi',

  'dash.offlineTitle': 'API AILOFlow non raggiungibile',
  'dash.offlineBody':
    'La GUI non inventa dati: finché il server non risponde non viene mostrato alcun valore. Avvia il runtime con "npm run server" nella cartella del progetto.',
  'dash.loadedModel': 'MODELLO CARICATO',
  'dash.chooseModel': 'Scegli un modello nella scheda Modelli per iniziare.',
  'dash.loadedIn': 'caricato in {duration}',
  'dash.genSpeed': 'VELOCITÀ GENERAZIONE',
  'dash.genSpeedHint': 'token/s · misurati {when}',
  'dash.noGeneration': 'nessuna generazione ancora eseguita',
  'dash.firstToken': 'LATENZA PRIMO TOKEN',
  'dash.promptRate': 'prompt {rate} tok/s',
  'dash.awaitingData': 'in attesa di dati',
  'dash.cache': 'CACHE GERARCHICA',
  'dash.cacheHint': '{hits} hit / {misses} miss',
  'dash.cacheIdle': 'attiva solo durante lo streaming .sflow',
  'dash.hardware': 'MONITORAGGIO HARDWARE IN TEMPO REALE',
  'dash.sampledAt': 'campionato {time}',
  'dash.gpus': 'SCHEDE GRAFICHE RILEVATE',
  'dash.noGpu': "Nessuna GPU rilevata: l'inferenza userà la CPU.",
  'dash.utilization': 'Utilizzo',
  'dash.utilizationShared': 'Utilizzo (aggregato di sistema)',
  'dash.vramSource': 'fonte: {source}',
  'dash.counters': 'contatori: {source}',
  'dash.autoConfig': 'CONFIGURAZIONE AUTOMATICA',
  'dash.backend': 'Backend selezionato',
  'dash.ramCache': 'RAM dedicata alla cache',
  'dash.vramCache': 'VRAM dedicata',
  'dash.vramNone': 'nessuna (VRAM non nota)',
  'dash.threads': 'Thread di lavoro',
  'dash.prefetch': 'Profondità prefetch',
  'dash.prefetchLayers': '{n} layer',
  'dash.ioQueue': 'Profondità coda I/O',
  'dash.shardStrategy': 'Strategia di sharding',
  'dash.tunedMeasured': 'tuning calcolato su banda misurata',
  'dash.tunedEstimated': 'tuning su stime da bus: esegui il benchmark storage',
  'dash.pipeline': 'PIPELINE STORAGE .SFLOW',
  'dash.pipelineEmpty':
    'Nessuna misura disponibile. La pipeline di streaming si misura dalla scheda Modelli, creando un container .sflow e avviando lo sweep dei layer: i valori qui sopra provengono solo da letture reali su disco.',
  'dash.pipelineModel': 'Container misurato',
  'dash.pipelineLayers': 'Layer trasferiti',
  'dash.pipelineRead': 'Letti da storage',
  'dash.pipelineBandwidth': 'Banda effettiva',
  'dash.pipelineLatency': 'Latenza media per layer',
  'dash.pipelineCache': 'Cache hit',
  'dash.pipelinePerToken': 'Peso per token',
  'dash.pipelineDrives': 'Dischi coinvolti',
  'dash.cpuNote': '{threads} thread · {simd}',
  'dash.gpuNoCounter': 'nessun contatore GPU leggibile',
  'dash.pipelineIdle': 'pipeline .sflow inattiva',
  'dash.storageHint': '{iops} IOPS · coda {queue}',

  'storage.title': 'AILO STORAGE FABRIC',
  'storage.summary': '{count} dispositivi · {total} totali · banda misurata {bandwidth}',
  'storage.notMeasured': 'non ancora rilevata',
  'storage.benchmarkAll': 'Benchmark di tutti i dischi',
  'storage.benchmarking': 'Benchmark in corso...',
  'storage.benchmarkOne': 'Misura questo disco',
  'storage.measuring': 'Misurazione...',
  'storage.noDrives': 'Nessun dispositivo di storage rilevato',
  'storage.noDrivesBody':
    'Il sistema operativo non ha restituito volumi utilizzabili. Nessun disco fittizio viene mostrato al loro posto.',
  'storage.detecting': 'Rilevamento dispositivi...',
  'storage.usage': 'Occupazione',
  'storage.seqRead': 'Lettura sequenziale',
  'storage.seqWrite': 'Scrittura seq.',
  'storage.randRead': 'Lettura casuale 4K',
  'storage.latency': 'Latenza',
  'storage.iops': 'IOPS',
  'storage.free': 'Spazio libero',
  'storage.estimateWarning':
    'Questi valori sono stime derivate dal tipo di bus, non misure. Esegui il benchmark per ottenere i dati reali.',
  'storage.cacheWarning':
    'Le letture sono state servite dalla cache del sistema operativo: sono un limite superiore, non la velocità del disco.',
  'storage.benchmarkDone': 'Benchmark completato in {duration} · {time}',

  'models.detected': 'MODELLI RILEVATI',
  'models.scanning': 'Scansione di Ollama e delle cartelle configurate...',
  'models.empty':
    'Nessun modello trovato. Aggiungi una cartella con file .gguf dalle Impostazioni, oppure scaricane uno dal Catalogo.',
  'models.noEngine': 'nessun motore',
  'models.selectTitle': 'Seleziona un modello',
  'models.selectBody':
    "I metadati mostrati qui vengono letti direttamente dall'header binario GGUF: architettura, layer, quantizzazione e fabbisogno di memoria sono calcolati sui tensori reali.",
  'models.noLocalFile': 'nessun file locale associato',
  'models.load': "Carica per l'inferenza",
  'models.shard': 'Shardizza in .sflow',
  'models.measurePipeline': 'Misura pipeline',
  'models.noEngineHint': "Nessun motore d'inferenza può eseguire questo modello.",
  'models.readingHeader': 'Lettura header GGUF...',
  'models.architecture': 'Architettura',
  'models.parameters': 'Parametri',
  'models.quantization': 'Quantizzazione',
  'models.layers': 'Layer',
  'models.context': 'Context length',
  'models.tensors': 'Tensori',
  'models.fileSize': 'Dimensione file',
  'models.weightsInRam': 'Pesi in RAM',
  'models.kvCache': 'KV cache (ctx pieno)',
  'models.heads': 'Head / KV head',
  'models.notInspectable':
    'Questo modello non espone un header GGUF leggibile, quindi non vengono mostrati metadati: nessun valore viene stimato al loro posto.',
  'models.operationLog': 'LOG OPERAZIONE',
  'models.pipelineResult': 'MISURA REALE DELLA PIPELINE',
  'models.sweepErrors': '{count} errori di lettura durante lo sweep: {first}',

  'catalog.title': 'CATALOGO OPEN-WEIGHT',
  'catalog.filterPlaceholder': 'Filtra per nome, editore o capacità (coding, reasoning...)',
  'catalog.noMatch': 'Nessun modello corrisponde al filtro.',
  'catalog.all': 'tutti',
  'catalog.gigante': 'gigante',
  'catalog.grande': 'grande',
  'catalog.medio': 'medio',
  'catalog.piccolo': 'piccolo',
  'catalog.searchPlaceholder': 'Cerca su Hugging Face (es. Kimi-K2 GGUF)',
  'catalog.noRepos': 'Nessun repository GGUF trovato.',
  'catalog.downloads': '{count} download',
  'catalog.directory': 'cartella di download · {free} liberi',
  'catalog.spaceHint':
    'Un modello da 300B in Q4 occupa circa 180 GB. Scegli un volume con spazio sufficiente dalle Impostazioni prima di iniziare.',
  'catalog.estimate': 'Stima',
  'catalog.computing': "Calcolo sull'hardware misurato...",
  'catalog.license': 'Licenza: {license} · contesto {context} token',
  'catalog.totalParams': '{n}B totali',
  'catalog.activeParams': '{n}B attivi',
  'catalog.inQ4': '{size} in Q4',
  'catalog.filesIn': 'File GGUF in {repo}',
  'catalog.readingRepo': 'Lettura del repository...',
  'catalog.noGgufFiles': 'Nessun file .gguf in questo repository.',
  'catalog.download': 'Scarica',
  'catalog.parts': '{n} parti',
  'catalog.emptyTitle': 'Scegli un modello a sinistra',
  'catalog.emptyBody':
    'Vedrai quanto occuperebbe e a che velocità girerebbe su questo computer, calcolata sulle bande di memoria e storage misurate. I file vengono poi scaricati direttamente da Hugging Face, con ripresa automatica se la connessione cade.',

  'est.totalWeight': 'Peso totale',
  'est.bytesPerToken': 'Byte per token',
  'est.kvCache': 'KV cache',
  'est.fitsInMemory': 'Entra in memoria',
  'est.fitsYes': 'sì',
  'est.fitsNo': 'no, serve storage',
  'est.overlapped': 'Stima (sovrapposto)',
  'est.bottleneck': 'Collo di bottiglia',
  'est.placement': 'DISTRIBUZIONE DEI PESI',
  'est.resident': '{size} residenti',
  'est.perToken': '{size}/token',
  'est.noBandwidth': 'banda n/d',
  'est.projections': 'SE MIGLIORI LO STORAGE',
  'est.confidence': 'Confidenza',

  'chat.noModelTitle': 'Nessun modello caricato',
  'chat.noModelBody':
    "La chat parla con un motore d'inferenza reale (llama.cpp o Ollama): finché non carichi un modello dalla scheda Modelli non viene generato alcun testo.",
  'chat.placeholder': 'Scrivi un messaggio per il modello locale...',
  'chat.send': 'Invia',
  'chat.stop': 'Ferma',
  'chat.clear': 'Pulisci',
  'chat.intro':
    'Scrivi un messaggio: i token arrivano in streaming dal motore locale, e le metriche sotto ogni risposta sono quelle riportate dal motore stesso.',
  'chat.reasoning': 'Ragionamento del modello ({count} caratteri)',
  'chat.thinking': 'sta ragionando...',
  'chat.parameters': 'PARAMETRI',
  'chat.systemPrompt': 'System prompt',
  'chat.systemNone': '(nessuno)',
  'chat.temperature': 'Temperature',
  'chat.topP': 'Top-P',
  'chat.topK': 'Top-K',
  'chat.repeatPenalty': 'Repetition penalty',
  'chat.maxTokens': 'Max token',
  'chat.unlimitedTokens': 'Nessun limite di token (consigliato per i modelli con ragionamento)',
  'chat.contextLength': 'Context length',
  'chat.seed': 'Seed (vuoto = casuale)',
  'chat.seedRandom': 'casuale',
  'chat.stopSequences': 'Stop sequences (separate da virgola)',
  'chat.tokensPerSec': '{rate} tok/s',
  'chat.tokensNa': 'tok/s n/d',
  'chat.tokenCount': '{count} token',

  'dl.active': 'Download',
  'dl.none': 'Nessun download in corso.',
  'dl.part': 'parte {index}/{total}',
  'dl.remaining': 'mancano {time}',
  'dl.completedIn': 'Completato. Il modello compare nella scheda Modelli.',
  'dl.cancelled': 'Annullato',
  'dl.failed': 'Fallito',
  'dl.running': 'In download',
  'dl.retrying': 'Connessione persa, nuovo tentativo (n. {attempt})',
  'dl.completed': 'Completato',
  'dl.clearFinished': 'Rimuovi completati',
  'dl.keepsRunning': 'I download proseguono anche cambiando scheda o chiudendo questa pagina.',

  'settings.engines': "MOTORI D'INFERENZA",
  'settings.engineTitle': 'MOTORE AILOFLOW',
  'settings.engineNone':
    "Nessun motore installato. AILOFlow può scaricare da solo la build di llama.cpp adatta alla GPU rilevata: da quel momento l'inferenza gira come processo figlio del runtime, senza alcun demone esterno.",
  'settings.findBuilds': 'Cerca build disponibili',
  'settings.checkUpdates': 'Cerca aggiornamenti',
  'settings.release': 'Release {release} — build ordinate per idoneità al tuo hardware',
  'settings.install': 'Installa',
  'settings.downloadPhase': 'Download',
  'settings.extractPhase': 'Estrazione',
  'settings.verifyPhase': 'Verifica',
  'settings.llamaPath': 'Percorso del binario llama-server (llama.cpp)',
  'settings.llamaPathHint':
    'Serve per eseguire direttamente file .gguf. Senza binario, i modelli GGUF locali risultano non eseguibili invece di sembrare disponibili.',
  'settings.save': 'Salva',
  'settings.ollamaUrl': 'URL demone Ollama',
  'settings.gpuBackend': 'Backend GPU',
  'settings.automatic': 'Automatico ({backend})',
  'settings.useOllama': "Usa il demone Ollama come motore d'inferenza",
  'settings.modelFolders': 'CARTELLE DEI MODELLI',
  'settings.noFolders': 'Nessuna cartella configurata.',
  'settings.folderPlaceholder': 'Percorso assoluto di una cartella con file .gguf',
  'settings.add': 'Aggiungi',
  'settings.runtime': 'RUNTIME',
  'settings.apiPort': 'Porta API (riavvio necessario)',
  'settings.gpuLayers': 'Layer su GPU (vuoto = automatico)',
  'settings.contextLength': 'Finestra di contesto in token (vuoto = automatico, ricarica il modello per applicare)',
  'settings.contextLengthHint':
    'In automatico usa il massimo del modello, con un tetto di 32768 per non far esplodere la KV cache. Gli assistenti negli IDE inviano system prompt da 10k token e oltre, quindi una finestra piccola li fa fallire.',
  'settings.auto': 'auto',
  'settings.privacy':
    "Nessun prompt e nessun modello lascia questa macchina: l'API resta in ascolto solo su 127.0.0.1 e non è prevista alcuna telemetria remota.",
  'settings.saved': 'Impostazioni salvate.',
  'settings.language': 'Lingua interfaccia',
  'settings.downloadFolder': 'Cartella di download',
  'settings.llamaPathPlaceholder': 'es. C:\\llama.cpp\\build\\bin\\llama-server.exe',

  'log.shardStart': 'Avvio: {model} · {layers} layer · {size} → {directory}',
  'log.shardProgress': '{percent}% — {status} ({copied} copiati)',
  'log.shardDone': 'Completato: {shards} shard in {path}',
  'log.shardDrives': 'Shard scritti fisicamente su: {drives}',
  'log.integrityOk': 'Integrità verificata: tutti gli shard corrispondono al manifest.',
  'log.integrityFailed': 'Integrità FALLITA: {problems}',
  'log.sweepLayer': 'layer {layer}/{total} · {bandwidth} MB/s · cache {cache}%',

  'models.noEngineForModel': 'Nessun motore disponibile per questo modello',
  'models.splitParts': '{n} file',
  'models.incompleteSet':
    'Questo modello è pubblicato in più file e alcune parti mancano ancora, quindi non può essere caricato. Completa il download dalla scheda Catalogo.',
  'models.sweepLayers': 'Layer trasferiti',
  'models.sweepRead': 'Letti da disco',
  'models.sweepBandwidth': 'Banda effettiva',
  'models.sweepDuration': 'Durata',
  'models.sweepLatency': 'Latenza media/layer',
  'models.sweepCache': 'Cache hit',
  'models.sweepPrefetch': 'Prefetch avviati',
  'models.sweepPerToken': 'Peso per token',
  'models.context2': 'Context length',
};

export const TRANSLATIONS: Record<Language, Record<TranslationKey, string>> = { en, it };

export const LANGUAGE_NAMES: Record<Language, string> = { en: 'English', it: 'Italiano' };

export type Translate = (key: TranslationKey, params?: Record<string, string | number>) => string;

export function createTranslator(language: Language): Translate {
  return (key, params) => {
    const table = TRANSLATIONS[language] || TRANSLATIONS.en;
    let text: string = table[key] ?? TRANSLATIONS.en[key] ?? key;
    if (params) {
      for (const [name, value] of Object.entries(params)) {
        text = text.replace(new RegExp(`\\{${name}\\}`, 'g'), String(value));
      }
    }
    return text;
  };
}

/** Pick the initial language: stored choice first, then the browser's. */
export function detectInitialLanguage(): Language {
  try {
    const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (stored === 'en' || stored === 'it') return stored;
  } catch {
    // Storage unavailable (private mode); fall through to detection.
  }
  return typeof navigator !== 'undefined' && navigator.language?.toLowerCase().startsWith('it') ? 'it' : 'en';
}

export interface I18nContextValue {
  language: Language;
  setLanguage: (language: Language) => void;
  t: Translate;
}

export const I18nContext = createContext<I18nContextValue>({
  language: 'en',
  setLanguage: () => undefined,
  t: createTranslator('en'),
});

export function useI18n(): I18nContextValue {
  return useContext(I18nContext);
}
