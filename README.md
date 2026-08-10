# AILOFlow

Runtime locale per LLM open-weight con streaming dinamico dei pesi da storage.

Il progetto ha una regola non negoziabile, ereditata dal §50 della specifica: **ogni
numero mostrato deve essere misurato**. Dove una misura non è possibile su questa
macchina, l'interfaccia scrive `n/d` invece di un valore plausibile, e i valori
stimati sono etichettati come tali.

---

## Avvio

```bash
npm install && npm run server
```

In una seconda shell, per l'interfaccia:

```bash
npm run gui:dev
```

L'API sta su `127.0.0.1:11500` (la 11434 appartiene a Ollama), la GUI su
`localhost:3000` e usa un proxy verso l'API, quindi nessuna porta è cablata nel
codice del frontend.

---

## Motore proprio

AILOFlow installa e gestisce il proprio motore: nessun demone esterno richiesto.

```bash
npm run cli -- engine list       # build disponibili per l'hardware rilevato
npm run cli -- engine install    # scarica la migliore e la registra
```

Sceglie da solo la build giusta fra CUDA, HIP/ROCm, Vulkan, Metal e CPU in base
alle GPU trovate, la scarica dalle release ufficiali di llama.cpp, la scompatta e
la verifica. Da quel momento l'inferenza gira come processo figlio del runtime.

I kernel di calcolo restano quelli di llama.cpp — riscriverli non era l'obiettivo —
ma il ciclo di vita del motore è interamente nostro.

Ollama resta supportato come motore opzionale se il demone è attivo. I modelli
già scaricati con Ollama sono file GGUF: AILOFlow li legge dai suoi manifest e li
esegue **anche a demone spento**, col proprio motore.

Se nessun motore è disponibile il caricamento **fallisce con il motivo preciso**.
Non esiste un percorso che produca testo senza un modello dietro.

---

## API per gli IDE

Il runtime espone due protocolli sulla stessa porta, così qualsiasi estensione
che parla con Ollama o con l'API OpenAI può puntare ad AILOFlow:

| Protocollo | Base URL | Endpoint principali |
|---|---|---|
| OpenAI | `http://127.0.0.1:11500/v1` | `/models`, `/chat/completions`, `/completions` |
| Ollama | `http://127.0.0.1:11500/api` | `/tags`, `/chat`, `/generate`, `/show`, `/ps`, `/version` |

I modelli rispondono sia all'id completo sia a un **nome breve** (es.
`Qwen3-1.7B-Q4_K_M`), perché un percorso assoluto non si può scrivere in un campo
di configurazione. I modelli "thinking" (Qwen3, DeepSeek-R1, gpt-oss) tengono il
ragionamento separato dalla risposta, in `reasoning_content`.

Operazioni che riguardano il registro di Ollama (`/api/pull`, `/api/push`, …)
rispondono 501 spiegando di usare il catalogo di AILOFlow: fallire in chiaro è
meglio che fingere.

---

## Catalogo e download

```bash
npm run cli -- catalog list                    # i più grandi open-weight
npm run cli -- catalog search "Kimi-K2 GGUF"   # ricerca su Hugging Face
npm run cli -- catalog files <repo>            # quantizzazioni disponibili
npm run cli -- catalog download <repo> <file>  # scarica, anche multi-parte
```

I modelli oltre i 50 GB sono pubblicati a pezzi (`-00001-of-00009.gguf`): il
downloader li tratta come un'unica unità e riprende con una richiesta Range se la
connessione cade. La cartella di destinazione è, per impostazione predefinita, il
volume con più spazio libero, non il disco di sistema.

---

## Stima delle prestazioni

```bash
npm run cli -- membench                                  # banda RAM misurata
npm run cli -- estimate --params 300 --active 32         # 300B MoE
npm run cli -- estimate --catalog kimi-k2
```

La decodifica è vincolata dalla banda: ogni token richiede la lettura di tutti i
pesi attivi. Lo stimatore distribuisce i pesi su VRAM, RAM e storage secondo le
capacità reali e calcola

```text
token/s = 1 / Σ (byte serviti dal livello / banda del livello)
```

riportando sia il limite pessimistico (livelli in serie) sia quello ottimistico
(trasferimenti sovrapposti dal prefetch). Le bande vengono dalle misure, tranne
quella della VRAM che è un dato di targa — ed è etichettata come tale, così la
confidenza del risultato è sempre visibile.

---

## Rilevamento hardware

Tutto viene letto dal sistema, mai dedotto dal nome del componente:

- **CPU** — set di istruzioni via `IsProcessorFeaturePresent` (Windows),
  `/proc/cpuinfo` (Linux), `sysctl` (macOS). Un i9-13900K viene riconosciuto come
  AVX2, non AVX-512: dedurlo dal modello era il vecchio bug.
- **RAM** — totale/libera live, più tipo, frequenza e numero di moduli via SMBIOS.
- **GPU** — *tutti* i vendor:
  - VRAM reale da `HardwareInformation.qwMemorySize` nel registro (l'`AdapterRAM`
    di WMI è a 32 bit e riporta 4 GB per qualsiasi scheda più capiente);
  - utilizzo e VRAM occupata dai contatori prestazioni `GPU Engine` /
    `GPU Adapter Memory`, che funzionano su AMD, Intel e NVIDIA;
  - `amdgpu`/`i915` sysfs su Linux, `nvidia-smi` dove presente per la temperatura.
- **Storage** — volumi, bus type e spazio; le prestazioni restano **stime
  dichiarate** finché non si esegue il benchmark.

---

## Benchmark dello storage

```bash
npm run cli -- storage benchmark
```

Scrive un file temporaneo, lo rilegge e lo cancella. I risultati vengono salvati e
riusati nelle sessioni successive.

**Avvertenza onesta**: Node non può aprire file con `FILE_FLAG_NO_BUFFERING` /
`O_DIRECT`, quindi un set di prova più piccolo della RAM libera viene riletto
dalla page cache del sistema operativo. In quel caso il risultato è marcato
`cacheInfluenced` e va letto come limite superiore, non come velocità del disco.

---

## Container `.sflow`

Uno `.sflow` descrive come un modello è spezzato in shard sui dispositivi.

```bash
# distribuisce fisicamente gli shard su ogni disco scrivibile
npm run cli -- model shard <modello.gguf> -d

# verifica che gli shard su disco corrispondano al manifest
npm run cli -- model validate <container.sflow>

# misura la pipeline: legge ogni layer attraverso fabric -> cache -> prefetch
npm run cli -- benchmark <container.sflow>
```

`validate` esiste perché un manifest che dichiara 60 GB con shard da pochi byte è
uno stub, e l'interfaccia deve poterlo dire.

---

## Cosa misura davvero la pipeline

Lo sweep dei layer è I/O reale: legge ogni tensore dagli shard attraverso la cache
gerarchica con il prefetch attivo, e riporta banda, latenza per layer, hit rate e
**quanti token/s lo storage potrebbe sostenere** con quel traffico di pesi.

Non produce token. Generare testo richiede i kernel di calcolo, che restano a
llama.cpp/Ollama. Il confronto utile non è "AILOFlow contro Ollama sui token/s",
ma: *lo storage riesce ad alimentare la velocità che il motore raggiunge?*

---

## Test

```bash
npm test
```

La suite scrive file GGUF reali (header binario completo), li shardizza, li
rilegge dal fabric e verifica che i byte sopravvivano intatti. Verifica anche che
i valori non misurabili restino `null` e che uno shard mancante sia un errore, non
un dato sintetico.
