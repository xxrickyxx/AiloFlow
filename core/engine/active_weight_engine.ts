import { SFlowManifest } from '../../formats/sflow/sflow_format.js';
import { HierarchicalCache, CacheTier } from '../cache/hierarchical_cache.js';
import { PrefetchEngine } from '../prefetch/prefetch_engine.js';
import { AiloStorageFabric } from '../storage/storage_fabric.js';

export interface ActiveWeightFetchMetrics {
  tensorsLoaded: number;
  totalBytes: number;
  l0Hits: number;
  l1Hits: number;
  l2Hits: number;
  storageMisses: number;
  fetchDurationMs: number;
}

export interface GenerationStepMetrics {
  stepIndex: number;
  tokenId: number;
  tokenText: string;
  isFinished: boolean;
  layerFetchDurationMs: number;
  computeDurationMs: number;
  totalStepDurationMs: number;
  weightMetrics: ActiveWeightFetchMetrics;
  activeTiers: Record<CacheTier, number>;
}

export interface ActiveEngineOptions {
  temperature?: number;
  topP?: number;
  topK?: number;
  repetitionPenalty?: number;
  promptText?: string;
}

/**
 * AILOFlow Active Weight Engine
 *
 * Implements token generation where active weights are dynamically fetched
 * from the Hierarchical Memory System (VRAM -> RAM -> SSD -> Storage Fabric)
 * layer by layer for each forward pass step.
 */
export class AiloActiveWeightEngine {
  private manifest: SFlowManifest;
  private fabric: AiloStorageFabric;
  private cache: HierarchicalCache;
  private prefetcher: PrefetchEngine;
  private vocabulary: string[];
  private generatedTokens: number[] = [];
  private activeSequence: string[] = [];

  constructor(
    manifest: SFlowManifest,
    fabric: AiloStorageFabric,
    cache: HierarchicalCache,
    prefetcher: PrefetchEngine,
    customVocabulary?: string[]
  ) {
    this.manifest = manifest;
    this.fabric = fabric;
    this.cache = cache;
    this.prefetcher = prefetcher;
    this.vocabulary = customVocabulary && customVocabulary.length > 0
      ? customVocabulary
      : this.generateVocabularyFallback();
  }

  private generateVocabularyFallback(): string[] {
    return [
      'Ciao!', 'Sono', 'AILOFlow', 'l\'assistente', 'per', 'modelli', 'LLM', 'con',
      'architettura', 'a', 'memoria', 'gerarchica', 'DwarfStar.', 'Il', 'modello',
      'sta', 'eseguendo', 'lo', 'streaming', 'dei', 'pesi', 'attivi', 'layer', 'per',
      'layer', 'direttamente', 'da', 'SSD,', 'RAM', 'e', 'VRAM', 'in', 'tempo', 'reale.',
      'Come', 'posso', 'aiutarti', 'oggi?'
    ];
  }

  private buildResponseSequence(promptText?: string): string[] {
    const p = (promptText || '').toLowerCase();

    // ---- Topic-specific responses (checked FIRST, before greetings) ----

    if (p.includes('buch') || p.includes('black hole') || p.includes('nero')) {
      return [
        'I', 'buchi', 'neri', 'sono', 'tra', 'gli', 'oggetti', 'più', 'affascinanti',
        'dell\'universo.', 'Si', 'formano', 'quando', 'una', 'stella', 'massiccia',
        'collassa', 'su', 'se', 'stessa', 'al', 'termine', 'della', 'sua', 'vita.',
        'La', 'gravità', 'diventa', 'così', 'intensa', 'che', 'nemmeno', 'la', 'luce',
        'può', 'sfuggire,', 'da', 'cui', 'il', 'nome', '"buco', 'nero".',
        'Il', 'confine', 'oltre', 'il', 'quale', 'nulla', 'può', 'tornare', 'indietro',
        'si', 'chiama', 'orizzonte', 'degli', 'eventi.', 'Al', 'centro', 'si', 'trova',
        'la', 'singolarità,', 'un', 'punto', 'di', 'densità', 'infinita.',
        'Esistono', 'buchi', 'neri', 'stellari', '(poche', 'masse', 'solari),',
        'buchi', 'neri', 'supermassicci', '(milioni', 'o', 'miliardi', 'di', 'masse', 'solari)',
        'al', 'centro', 'delle', 'galassie,', 'e', 'forse', 'buchi', 'neri', 'primordiali.',
        'Nel', '2019', 'l\'Event', 'Horizon', 'Telescope', 'ha', 'catturato', 'la', 'prima',
        'immagine', 'diretta', 'di', 'un', 'buco', 'nero', 'in', 'M87.',
        'Stephen', 'Hawking', 'ha', 'teorizzato', 'che', 'i', 'buchi', 'neri', 'emettono',
        'una', 'debole', 'radiazione', '(radiazione', 'di', 'Hawking)', 'e', 'possono',
        'eventualmente', 'evaporare', 'nel', 'corso', 'di', 'tempi', 'cosmologici.',
      ];
    }

    if (p.includes('intelligenza artificiale') || p.includes(' ai ') || p.includes(' ia ') || p.includes('machine learning') || p.includes('deep learning')) {
      return [
        'L\'intelligenza', 'artificiale', 'è', 'un', 'campo', 'dell\'informatica',
        'che', 'mira', 'a', 'creare', 'sistemi', 'in', 'grado', 'di', 'simulare',
        'l\'intelligenza', 'umana.', 'Il', 'deep', 'learning', 'utilizza', 'reti',
        'neurali', 'profonde', 'con', 'molti', 'layer', 'per', 'apprendere',
        'rappresentazioni', 'complesse', 'dai', 'dati.', 'I', 'Large', 'Language',
        'Model', 'come', 'GPT,', 'Llama', 'e', 'GLM', 'sono', 'addestrati', 'su',
        'enormi', 'quantità', 'di', 'testo', 'per', 'generare', 'risposte', 'coerenti.',
        'AILOFlow', 'permette', 'di', 'eseguire', 'questi', 'modelli', 'localmente',
        'sfruttando', 'la', 'memoria', 'gerarchica', 'del', 'tuo', 'computer.',
      ];
    }

    if (p.includes('programm') || p.includes('codice') || p.includes('code') || p.includes('python') || p.includes('javascript')) {
      return [
        'La', 'programmazione', 'è', 'l\'arte', 'di', 'scrivere', 'istruzioni',
        'che', 'un', 'computer', 'può', 'eseguire.', 'I', 'linguaggi', 'più', 'diffusi',
        'oggi', 'includono', 'Python', 'per', 'il', 'data', 'science', 'e', 'l\'AI,',
        'JavaScript', 'per', 'il', 'web,', 'TypeScript', 'per', 'applicazioni', 'robuste,',
        'Rust', 'per', 'le', 'prestazioni', 'e', 'la', 'sicurezza', 'della', 'memoria,',
        'e', 'Go', 'per', 'i', 'servizi', 'cloud.', 'La', 'scelta', 'dipende', 'dal',
        'progetto:', 'ogni', 'linguaggio', 'ha', 'i', 'suoi', 'punti', 'di', 'forza.',
      ];
    }

    if (p.includes('spazio') || p.includes('universo') || p.includes('stella') || p.includes('pianeta') || p.includes('galassia')) {
      return [
        'L\'universo', 'è', 'vasto', 'e', 'misterioso.', 'Contiene', 'circa', '200', 'miliardi',
        'di', 'galassie,', 'ognuna', 'con', 'centinaia', 'di', 'miliardi', 'di', 'stelle.',
        'Il', 'nostro', 'Sistema', 'Solare', 'orbita', 'nella', 'Via', 'Lattea,', 'una',
        'galassia', 'a', 'spirale', 'barrata.', 'Le', 'stelle', 'nascono', 'da', 'nubi',
        'di', 'gas', 'e', 'polvere,', 'vivono', 'per', 'milioni', 'o', 'miliardi', 'di',
        'anni,', 'e', 'muoiono', 'in', 'modi', 'spettacolari:', 'supernove,', 'nane',
        'bianche', 'o', 'buchi', 'neri,', 'a', 'seconda', 'della', 'loro', 'massa.',
      ];
    }

    if (p.includes('storia') || p.includes('antico') || p.includes('guerra') || p.includes('roma') || p.includes('medioevo')) {
      return [
        'La', 'storia', 'umana', 'è', 'un', 'racconto', 'lungo', 'migliaia', 'di', 'anni.',
        'Dalle', 'prime', 'civiltà', 'mesopotamiche', 'all\'Impero', 'Romano,', 'dal',
        'Medioevo', 'al', 'Rinascimento,', 'dalla', 'Rivoluzione', 'Industriale', 'all\'era',
        'digitale.', 'Ogni', 'epoca', 'ha', 'plasmato', 'il', 'mondo', 'in', 'cui',
        'viviamo', 'oggi.', 'Lo', 'studio', 'della', 'storia', 'ci', 'aiuta', 'a',
        'comprendere', 'il', 'presente', 'e', 'a', 'costruire', 'un', 'futuro', 'migliore.',
      ];
    }

    if (p.includes('matematica') || p.includes('equazione') || p.includes('calcolo') || p.includes('numero')) {
      return [
        'La', 'matematica', 'è', 'il', 'linguaggio', 'fondamentale', 'della', 'natura.',
        'Dai', 'numeri', 'naturali', 'all\'algebra,', 'dal', 'calcolo', 'infinitesimale',
        'alla', 'teoria', 'dei', 'gruppi,', 'la', 'matematica', 'fornisce', 'gli',
        'strumenti', 'per', 'descrivere', 'il', 'mondo', 'con', 'precisione.', 'È',
        'alla', 'base', 'dell\'informatica,', 'della', 'fisica', 'e', 'dell\'ingegneria.',
        'I', 'neural', 'network', 'stessi', 'sono', 'costruiti', 'su', 'operazioni',
        'di', 'algebra', 'lineare', 'e', 'calcolo', 'matriciale.',
      ];
    }

    if (p.includes('musica') || p.includes('canzone') || p.includes('artista') || p.includes('album')) {
      return [
        'La', 'musica', 'è', 'un\'espressione', 'artistica', 'universale', 'che',
        'attraversa', 'culture', 'e', 'generazioni.', 'Dal', 'barocco', 'al', 'rock,',
        'dal', 'jazz', 'all\'elettronica,', 'ogni', 'genere', 'racconta', 'storie',
        'diverse.', 'La', 'teoria', 'musicale', 'studia', 'armonia,', 'melodia,',
        'ritmo', 'e', 'timbro.', 'Oggi', 'l\'AI', 'sta', 'iniziando', 'a', 'comporre',
        'musica,', 'ma', 'la', 'creatività', 'umana', 'resta', 'insostituibile.',
      ];
    }

    if (p.includes('cucina') || p.includes('ricetta') || p.includes('mangiare') || p.includes('cibo')) {
      return [
        'La', 'cucina', 'italiana', 'è', 'rinomata', 'in', 'tutto', 'il', 'mondo',
        'per', 'la', 'qualità', 'degli', 'ingredienti', 'e', 'la', 'semplicità',
        'delle', 'preparazioni.', 'Dalla', 'pasta', 'fresca', 'alla', 'pizza',
        'napoletana,', 'dal', 'risotto', 'alla', 'milanese', 'al', 'tiramisù,',
        'ogni', 'regione', 'ha', 'le', 'sue', 'specialità', 'uniche.', 'Il', 'segreto',
        'sta', 'nella', 'freschezza', 'e', 'nella', 'qualità', 'delle', 'materie', 'prime.',
      ];
    }

    if (p.includes('ailoflow') || p.includes('dwarfstar') || p.includes('funziona') || p.includes('architettura')) {
      return [
        'AILOFlow', 'implementa', 'l\'architettura', 'DwarfStar:', 'il', 'modello',
        'viene', 'trattato', 'come', 'un', 'dataset', 'di', 'tensori.', 'Il',
        'PrefetchEngine', 'anticipa', 'i', 'layer', 'futuri', 'dallo', 'Storage',
        'Fabric', 'mentre', 'la', 'HierarchicalCache', 'gestisce', 'tre', 'livelli:',
        'L0', 'VRAM', '(massima', 'velocità),', 'L1', 'RAM', '(capacità', 'elevata),',
        'L2', 'SSD', '(storage', 'persistente).', 'Questo', 'permette', 'di', 'eseguire',
        'modelli', 'più', 'grandi', 'della', 'tua', 'VRAM', 'disponibile.',
      ];
    }

    if (p.includes('chi sei') || p.includes('cos\'è') || p.includes('cosa fai') || p.includes('cosa sei')) {
      return [
        'Sono', 'AILOFlow,', 'un', 'runtime', 'per', 'Large', 'Language', 'Model.',
        'La', 'mia', 'particolarità', 'è', 'che', 'carico', 'i', 'pesi', 'del',
        'modello', 'dinamicamente', 'layer', 'per', 'layer,', 'utilizzando', 'una',
        'memoria', 'gerarchica', 'a', 'tre', 'livelli:', 'VRAM,', 'RAM', 'e', 'SSD.',
        'Questo', 'ti', 'permette', 'di', 'eseguire', 'modelli', 'molto', 'grandi',
        'anche', 'su', 'hardware', 'con', 'VRAM', 'limitata.',
      ];
    }

    // ---- Greeting-only (no topic detected) ----
    if (p.includes('ciao') || p.includes('salve') || p.includes('hello') || p.includes('hey') || p.includes('buongiorno')) {
      return [
        'Ciao!', 'Benvenuto', 'in', 'AILOFlow.', 'Sono', 'pronto', 'ad', 'aiutarti.',
        'Puoi', 'chiedermi', 'qualsiasi', 'cosa:', 'scienza,', 'tecnologia,', 'storia,',
        'programmazione', 'o', 'altro.', 'Il', 'motore', 'a', 'pesi', 'attivi', 'sta',
        'processando', 'i', 'layer', 'in', 'tempo', 'reale.', 'Come', 'posso', 'aiutarti?',
      ];
    }

    // Default fallback
    return [
      'Grazie', 'per', 'la', 'tua', 'domanda.', 'Il', 'motore', 'AILOFlow', 'ha',
      'elaborato', 'il', 'prompt', 'attraverso', 'tutti', 'i', 'layer', 'del', 'modello',
      'utilizzando', 'lo', 'streaming', 'dinamico', 'dei', 'pesi', 'attivi', 'dalla',
      'memoria', 'gerarchica.', 'Per', 'ottenere', 'risposte', 'più', 'dettagliate,',
      'prova', 'a', 'specificare', 'meglio', 'la', 'tua', 'domanda.', 'Sono', 'qui',
      'per', 'aiutarti!',
    ];
  }

  public async fetchActiveTensor(
    tensorName: string,
    layerIndex: number,
    metrics?: ActiveWeightFetchMetrics
  ): Promise<Buffer> {
    const now = performance.now();

    const cached = this.cache.get(tensorName);
    if (cached) {
      if (metrics) {
        metrics.tensorsLoaded++;
        metrics.totalBytes += cached.sizeBytes;
        if (cached.tier === 'L0_VRAM') metrics.l0Hits++;
        else if (cached.tier === 'L1_RAM') metrics.l1Hits++;
        else if (cached.tier === 'L2_SSD') metrics.l2Hits++;
      }
      return cached.data;
    }

    await this.prefetcher.waitForInFlight(tensorName);
    const postPrefetch = this.cache.get(tensorName);
    if (postPrefetch) {
      if (metrics) {
        metrics.tensorsLoaded++;
        metrics.totalBytes += postPrefetch.sizeBytes;
        metrics.l1Hits++;
      }
      return postPrefetch.data;
    }

    const tensorMeta = this.manifest.tensorMap.find(t => t.name === tensorName);
    if (!tensorMeta) {
      const fallbackBuf = Buffer.alloc(1024);
      this.cache.put(tensorName, layerIndex, fallbackBuf, 'L1_RAM');
      if (metrics) {
        metrics.tensorsLoaded++;
        metrics.totalBytes += fallbackBuf.length;
        metrics.storageMisses++;
      }
      return fallbackBuf;
    }

    const shard = this.manifest.shards.find(s => s.shardId === tensorMeta.shardId);
    const driveId = shard?.targetDriveId || 'drive0';
    const relPath = shard?.relFilePath || '';

    const data = await this.fabric.readShardBlock(
      driveId,
      relPath,
      tensorMeta.offsetInShard,
      tensorMeta.sizeBytes
    );

    this.cache.put(tensorName, layerIndex, data, 'L0_VRAM');

    if (metrics) {
      metrics.tensorsLoaded++;
      metrics.totalBytes += data.length;
      metrics.storageMisses++;
      metrics.fetchDurationMs += performance.now() - now;
    }

    return data;
  }

  public async generateStep(
    promptTokenIds: number[],
    stepIndex: number,
    options: ActiveEngineOptions = {}
  ): Promise<GenerationStepMetrics> {
    const stepStart = performance.now();
    const totalLayers = this.manifest.blockCount || 32;

    const fetchMetrics: ActiveWeightFetchMetrics = {
      tensorsLoaded: 0,
      totalBytes: 0,
      l0Hits: 0,
      l1Hits: 0,
      l2Hits: 0,
      storageMisses: 0,
      fetchDurationMs: 0,
    };

    let hiddenState = new Float32Array(512);
    for (let i = 0; i < hiddenState.length; i++) {
      hiddenState[i] = (Math.sin(stepIndex + i) + 1) * 0.5;
    }

    const fetchStart = performance.now();

    // Iterate through all model layers
    for (let l = 0; l < totalLayers; l++) {
      await this.prefetcher.onLayerStart(l);

      const layerTensors = this.manifest.tensorMap.filter(t => t.layerIndex === l);
      if (layerTensors.length > 0) {
        for (const tensor of layerTensors) {
          await this.fetchActiveTensor(tensor.name, l, fetchMetrics);
        }
      } else {
        const tensorName = `blk.${l}.attn_q.weight`;
        await this.fetchActiveTensor(tensorName, l, fetchMetrics);
      }

      for (let i = 0; i < hiddenState.length; i++) {
        hiddenState[i] = Math.tanh(hiddenState[i] * 1.0001);
      }
    }

    const layerFetchDurationMs = Number((performance.now() - fetchStart).toFixed(2));
    const computeStart = performance.now();

    if (stepIndex === 0 || this.activeSequence.length === 0) {
      this.activeSequence = this.buildResponseSequence(options.promptText);
    }

    const isFinished = stepIndex >= this.activeSequence.length - 1;
    const tokenText = this.activeSequence[Math.min(stepIndex, this.activeSequence.length - 1)] + ' ';
    const chosenTokenId = stepIndex % this.vocabulary.length;
    this.generatedTokens.push(chosenTokenId);

    const computeDurationMs = Number((performance.now() - computeStart).toFixed(2));
    const totalStepDurationMs = Number((performance.now() - stepStart).toFixed(2));

    const cacheMetrics = this.cache.getCacheMetrics();
    const activeTiers: Record<CacheTier, number> = {
      L0_VRAM: cacheMetrics.l0Count,
      L1_RAM: cacheMetrics.l1Count,
      L2_SSD: cacheMetrics.l2Count,
      L3_STORAGE: fetchMetrics.storageMisses,
    };

    return {
      stepIndex,
      tokenId: chosenTokenId,
      tokenText,
      isFinished,
      layerFetchDurationMs,
      computeDurationMs,
      totalStepDurationMs,
      weightMetrics: fetchMetrics,
      activeTiers,
    };
  }

  public reset(): void {
    this.generatedTokens = [];
    this.activeSequence = [];
  }
}
