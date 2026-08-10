# AILOFlow

**A local runtime for open-weight LLMs, with dynamic weight streaming from storage.**

Run models larger than your RAM by treating them as a dataset spread across every
SSD in the machine, and measure honestly what that actually costs.

[🇬🇧 English](#english) · [🇮🇹 Italiano](#italiano)

---

<a name="english"></a>
## English

### The one rule

**Every number shown is measured.** Where a value cannot be measured on this
machine, the interface prints `n/a` instead of a plausible-looking placeholder,
and figures that are estimates are labelled as estimates. A benchmark that was
served by the OS page cache says so. This is not a demo that simulates a
runtime — nothing renders a number that did not come from a real measurement.

### Quick start

```bash
npm install
npm run server        # API on 127.0.0.1:11500
npm run gui:dev       # interface on localhost:3000
```

Then, in the interface: **Settings → Install engine**, **Catalog → download a
model**, **Models → load**, **Chat**.

### Bring your own engine — or let AILOFlow fetch one

AILOFlow installs and manages its own inference engine. No external daemon.

```bash
npm run cli -- engine list      # builds matching the detected hardware
npm run cli -- engine install   # download, unpack, verify, register
```

It picks between CUDA, HIP/ROCm, Vulkan, Metal and CPU based on the GPUs it
finds, fetches the build from the official llama.cpp releases and runs it as a
child process of the runtime. The compute kernels are llama.cpp's — rewriting
them was never the goal — but the engine's lifecycle is entirely ours.

Ollama is supported as an optional engine. Models already downloaded with Ollama
are ordinary GGUF files: AILOFlow reads them from Ollama's manifests and runs
them **even with the daemon stopped**.

If no engine can run a model, loading fails with the reason. There is no path
that produces text without a model behind it.

### Use it from your IDE

Two protocols on one port, so anything that speaks Ollama or the OpenAI API can
point at AILOFlow:

| Protocol | Base URL | Endpoints |
|---|---|---|
| OpenAI | `http://127.0.0.1:11500/v1` | `/models`, `/chat/completions`, `/completions` |
| Ollama | `http://127.0.0.1:11500` | `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/ps` |

Tested with Cline and Continue. Models answer to a **short name**
(`Qwen3-1.7B-Q4_K_M`) as well as their full id, because an absolute path cannot
be typed into a settings field. Reasoning models keep their chain of thought in
`thinking` / `reasoning_content`, separate from the answer.

The context window is sized from the model's own maximum (capped at 32768 by
default, adjustable). IDE assistants routinely send 12k-token system prompts, so
llama.cpp's 4096 default is far too small.

Cancelling a request in the IDE stops the engine. That sounds obvious, but a
disconnected client whose generation keeps running holds the GPU for as long as
the model feels like talking — with a reasoning model that is minutes of tokens
nobody will read. The disconnect is forwarded to the engine and utilisation
falls within seconds.

Whatever base URL you configure works: `.../v1`, `.../api`, or the bare host.
Providers append `/api/chat` to whatever you typed, and the resulting duplicated
prefix is normalised rather than answered with an HTML 404 that surfaces inside
the IDE as an unreadable error.

### Hardware discovery

Everything is read from the system; nothing is inferred from a component's name.

- **CPU** — instruction sets via `IsProcessorFeaturePresent` (Windows),
  `/proc/cpuinfo` (Linux), `sysctl` (macOS). An i9-13900K is correctly reported
  as AVX2, not AVX-512.
- **RAM** — live totals plus type, speed and module count via SMBIOS.
- **GPU** — *every* vendor. Real VRAM from the driver registry key
  (`AdapterRAM` is 32-bit and reports 4 GB for anything larger); utilisation and
  memory from the `GPU Engine` / `GPU Adapter Memory` performance counters,
  which work on AMD, Intel and NVIDIA alike; `amdgpu`/`i915` sysfs on Linux.
- **Storage** — volumes joined to physical disks through the partition table, so
  a drive is never mislabelled by array position. Volumes with no media are
  excluded rather than reported as phantom storage.

### Storage benchmark

```bash
npm run cli -- storage benchmark
```

Reading back a file you just wrote measures the page cache, not the disk — that
is how a SATA SSD "benchmarks" at 9 GB/s. AILOFlow instead finds a large
pre-existing file on the volume and reads scattered chunks of it. When no such
file exists it says so and marks the result as an upper bound.

Node cannot open files with `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`, and that
limitation is stated rather than hidden.

### `.sflow` containers

A `.sflow` describes how a model is split across devices.

```bash
npm run cli -- model shard <model.gguf> -d   # distribute across every writable drive
npm run cli -- model validate <container.sflow>
npm run cli -- benchmark <container.sflow>   # stream every layer, measure
```

`validate` exists because a manifest claiming 60 GB while its shards hold a few
bytes is a stub, and the interface must be able to say so.

### What the pipeline measures, and what it does not

The layer sweep is real I/O: it reads every tensor from the storage fabric
through the hierarchical cache with the prefetcher running, and reports
bandwidth, per-layer latency, hit rate and **how many tokens/s storage alone
could sustain**.

It does not generate text. Producing tokens needs compute kernels, which remain
llama.cpp's. **Chatting with a model does not activate the pipeline**: llama.cpp
loads weights its own way, and AILOFlow's cache and prefetcher are not in that
path. The useful question is not "AILOFlow versus llama.cpp on tokens/s" but
*can storage feed the rate the engine achieves?*

### Measured on the development machine

i9-13900K · 32 GB DDR5-6000 · Radeon RX 6750 XT · 1 NVMe + 4 SATA SSD

| | |
|---|---|
| SATA SSD sequential read (cache-cold) | 371–462 MB/s, 220 µs latency |
| RAM bandwidth (single-threaded memcpy) | 35.7 GB/s |
| Qwen3-1.7B via our Vulkan engine | 235–240 tok/s |
| Qwen3-Next-80B-A3B via our Vulkan engine | 19.4 tok/s |
| 80B sharded across 5 SSDs, full layer sweep | 44.97 GB in 117 s |

The 80B sweep is genuinely cache-cold — the shard set is larger than RAM.
Aggregate throughput stays near single-drive speed because the RAM cache cannot
hold a model this size, so the prefetcher spends the run throttled. That is a
real limitation of the current design, not a tuning parameter.

### Estimating a model before downloading it

```bash
npm run cli -- membench
npm run cli -- estimate --params 300 --active 32
npm run cli -- estimate --catalog kimi-k2
```

Decode is bandwidth-bound: one token requires reading every active weight. The
estimator places weights across VRAM, RAM and storage by real capacity and
computes

```
tokens/s = 1 / Σ (bytes served by tier / bandwidth of tier)
```

reporting both the pessimistic bound (tiers in series) and the optimistic one
(overlapped by prefetch). Bandwidths come from measurements, except VRAM which
is a manufacturer figure — and is labelled as such, so the confidence of each
result is always visible.

A worked example, on the machine above:

| Model | Bytes per token | Estimate |
|---|---|---|
| 300B dense, Q4 | 169 GB | 0.13 tok/s |
| 300B MoE, 32B active | 18 GB | 1.2 tok/s |
| gpt-oss-120b, 5.1B active | 2.9 GB | 12.6 tok/s |

A dense 300B streamed from disk would need 1345 GB/s to reach 10 tok/s — two
orders of magnitude beyond any consumer array. Mixture-of-Experts models are the
ones this architecture can actually serve.

### Catalog

40 curated open-weight models — Kimi K2, DeepSeek R1/V3, Qwen3 235B/480B,
Llama 4, GLM-4.6, MiniMax-M2, gpt-oss and more — filterable by size class and
capability, plus free search over Hugging Face. Every repository id is verified
against the API in the test suite.

Downloads are owned by the runtime, not by the browser tab: they survive
navigation, page reloads and server restarts, retry automatically on network
failures with exponential backoff, and resume from the exact byte. Models
published as split file sets are handled as one model.

### Tests

```bash
npm test
```

The suite writes real GGUF files with a complete binary header, shards them,
reads them back through the fabric and checks the bytes survive intact. It also
verifies that unmeasurable values stay `null`, that a missing shard is an error
rather than synthetic data, and that a page-cache-influenced measurement always
carries a warning.

### Privacy

No prompt and no model leaves the machine. The API listens on `127.0.0.1` only,
and there is no remote telemetry of any kind.

### Licence

Dual-licensed: **CC BY-NC-SA 4.0** for non-commercial use, separate agreement
for commercial use. See [LICENSE](LICENSE).

Author: Riccardo Sparacino — [LinkedIn](https://www.linkedin.com/in/riccardo-sparacino/)

---

<a name="italiano"></a>
## Italiano

### La regola

**Ogni numero mostrato è misurato.** Dove un valore non è misurabile su questa
macchina, l'interfaccia scrive `n/d` invece di un valore plausibile, e le stime
sono etichettate come tali. Un benchmark servito dalla cache del sistema
operativo lo dichiara. Non è una demo che simula un runtime: nulla mostra un
numero che non provenga da una misura reale.

### Avvio rapido

```bash
npm install
npm run server        # API su 127.0.0.1:11500
npm run gui:dev       # interfaccia su localhost:3000
```

Poi, nell'interfaccia: **Impostazioni → Installa motore**, **Catalogo → scarica
un modello**, **Modelli → carica**, **Chat**.

### Motore proprio

AILOFlow installa e gestisce il proprio motore d'inferenza. Nessun demone
esterno.

```bash
npm run cli -- engine list      # build adatte all'hardware rilevato
npm run cli -- engine install   # scarica, scompatta, verifica, registra
```

Sceglie da solo fra CUDA, HIP/ROCm, Vulkan, Metal e CPU in base alle GPU
trovate, scarica la build dalle release ufficiali di llama.cpp e la esegue come
processo figlio del runtime. I kernel di calcolo restano quelli di llama.cpp —
riscriverli non era l'obiettivo — ma il ciclo di vita del motore è interamente
nostro.

Ollama è supportato come motore opzionale. I modelli già scaricati con Ollama
sono normali file GGUF: AILOFlow li legge dai suoi manifest e li esegue **anche
a demone spento**.

Se nessun motore può eseguire un modello, il caricamento fallisce spiegando il
motivo. Non esiste un percorso che produca testo senza un modello dietro.

### Uso dagli IDE

Due protocolli sulla stessa porta, così qualunque strumento che parla Ollama o
API OpenAI può puntare ad AILOFlow:

| Protocollo | Base URL | Endpoint |
|---|---|---|
| OpenAI | `http://127.0.0.1:11500/v1` | `/models`, `/chat/completions`, `/completions` |
| Ollama | `http://127.0.0.1:11500` | `/api/tags`, `/api/chat`, `/api/generate`, `/api/show`, `/api/ps` |

Provato con Cline e Continue. I modelli rispondono sia all'id completo sia a un
**nome breve** (`Qwen3-1.7B-Q4_K_M`), perché un percorso assoluto non si può
scrivere in un campo di configurazione. I modelli con ragionamento tengono il
flusso di pensiero in `thinking` / `reasoning_content`, separato dalla risposta.

La finestra di contesto viene dimensionata sul massimo del modello (con un tetto
predefinito di 32768, modificabile). Gli assistenti negli IDE inviano
abitualmente system prompt da 12k token, quindi il default di 4096 di llama.cpp
è ampiamente insufficiente.

Annullare una richiesta nell'IDE ferma il motore. Sembra ovvio, ma un client
disconnesso la cui generazione prosegue tiene occupata la GPU per tutto il tempo
che il modello ha voglia di parlare — con un modello che ragiona, minuti di token
che nessuno leggerà. La disconnessione viene propagata al motore e l'utilizzo
scende in pochi secondi.

Qualunque base URL tu configuri funziona: `.../v1`, `.../api` o l'host nudo. I
provider aggiungono `/api/chat` a ciò che hai scritto, e il prefisso duplicato
che ne risulta viene normalizzato invece di produrre un 404 HTML che dentro
l'IDE appare come un errore illeggibile.

### Rilevamento hardware

Tutto viene letto dal sistema, mai dedotto dal nome del componente.

- **CPU** — set di istruzioni via `IsProcessorFeaturePresent` (Windows),
  `/proc/cpuinfo` (Linux), `sysctl` (macOS). Un i9-13900K risulta correttamente
  AVX2, non AVX-512.
- **RAM** — totali live più tipo, frequenza e numero di moduli via SMBIOS.
- **GPU** — *tutti* i vendor. VRAM reale dalla chiave di registro del driver
  (`AdapterRAM` è a 32 bit e riporta 4 GB per qualsiasi scheda più capiente);
  utilizzo e memoria dai contatori `GPU Engine` / `GPU Adapter Memory`, che
  funzionano su AMD, Intel e NVIDIA; sysfs `amdgpu`/`i915` su Linux.
- **Storage** — volumi uniti ai dischi fisici attraverso la tabella delle
  partizioni, così un disco non viene mai etichettato male per posizione
  nell'array. I volumi senza supporto vengono esclusi invece di comparire come
  storage fantasma.

### Benchmark dello storage

```bash
npm run cli -- storage benchmark
```

Rileggere un file appena scritto misura la page cache, non il disco — è così che
un SSD SATA "risulta" a 9 GB/s. AILOFlow cerca invece un file grande già
presente sul volume e ne legge blocchi sparsi. Quando un file simile non esiste,
lo dichiara e marca il risultato come limite superiore.

Node non può aprire file con `FILE_FLAG_NO_BUFFERING` / `O_DIRECT`, e questo
limite viene dichiarato invece che nascosto.

### Container `.sflow`

Uno `.sflow` descrive come un modello è distribuito sui dispositivi.

```bash
npm run cli -- model shard <modello.gguf> -d   # distribuisce su ogni disco scrivibile
npm run cli -- model validate <container.sflow>
npm run cli -- benchmark <container.sflow>     # legge ogni layer e misura
```

`validate` esiste perché un manifest che dichiara 60 GB con shard da pochi byte
è uno stub, e l'interfaccia deve poterlo dire.

### Cosa misura la pipeline, e cosa no

Lo sweep dei layer è I/O reale: legge ogni tensore dallo storage fabric
attraverso la cache gerarchica con il prefetch attivo, e riporta banda, latenza
per layer, hit rate e **quanti token/s lo storage da solo potrebbe sostenere**.

Non genera testo. Produrre token richiede kernel di calcolo, che restano di
llama.cpp. **Chattare con un modello non attiva la pipeline**: llama.cpp carica
i pesi a modo suo, e la cache e il prefetcher di AILOFlow non sono su quel
percorso. La domanda utile non è "AILOFlow contro llama.cpp sui token/s" ma
*lo storage riesce ad alimentare la velocità che il motore raggiunge?*

### Misure sulla macchina di sviluppo

i9-13900K · 32 GB DDR5-6000 · Radeon RX 6750 XT · 1 NVMe + 4 SSD SATA

| | |
|---|---|
| Lettura sequenziale SSD SATA (cache fredda) | 371–462 MB/s, 220 µs di latenza |
| Banda RAM (memcpy a thread singolo) | 35,7 GB/s |
| Qwen3-1.7B col nostro motore Vulkan | 235–240 tok/s |
| Qwen3-Next-80B-A3B col nostro motore Vulkan | 19,4 tok/s |
| 80B distribuito su 5 SSD, sweep completo | 44,97 GB in 117 s |

Lo sweep del modello da 80B è genuinamente a cache fredda: gli shard superano la
RAM. La banda aggregata resta vicina a quella di un singolo disco perché la
cache RAM non può contenere un modello di questa taglia, e il prefetcher passa
l'intera esecuzione in throttling. È un limite reale del progetto attuale, non
un parametro da tarare.

### Stimare un modello prima di scaricarlo

```bash
npm run cli -- membench
npm run cli -- estimate --params 300 --active 32
npm run cli -- estimate --catalog kimi-k2
```

La decodifica è vincolata dalla banda: un token richiede la lettura di tutti i
pesi attivi. Lo stimatore distribuisce i pesi su VRAM, RAM e storage secondo le
capacità reali e calcola

```
token/s = 1 / Σ (byte serviti dal livello / banda del livello)
```

riportando sia il limite pessimistico (livelli in serie) sia quello ottimistico
(sovrapposti dal prefetch). Le bande vengono dalle misure, tranne quella della
VRAM che è un dato di targa — ed è etichettata come tale, così la confidenza del
risultato è sempre visibile.

Un esempio concreto, sulla macchina qui sopra:

| Modello | Byte per token | Stima |
|---|---|---|
| 300B denso, Q4 | 169 GB | 0,13 tok/s |
| 300B MoE, 32B attivi | 18 GB | 1,2 tok/s |
| gpt-oss-120b, 5,1B attivi | 2,9 GB | 12,6 tok/s |

Un 300B denso in streaming da disco richiederebbe 1345 GB/s per arrivare a 10
tok/s: due ordini di grandezza oltre qualsiasi array consumer. I modelli
Mixture-of-Experts sono quelli che questa architettura può davvero servire.

### Catalogo

40 modelli open-weight selezionati — Kimi K2, DeepSeek R1/V3, Qwen3 235B/480B,
Llama 4, GLM-4.6, MiniMax-M2, gpt-oss e altri — filtrabili per classe di
dimensione e capacità, più ricerca libera su Hugging Face. Ogni id di repository
è verificato contro l'API nella suite di test.

I download appartengono al runtime, non alla scheda del browser: sopravvivono
alla navigazione, al ricaricamento della pagina e al riavvio del server,
ritentano da soli in caso di caduta di rete con backoff esponenziale, e
riprendono dal byte esatto. I modelli pubblicati in più file sono trattati come
un modello solo.

### Test

```bash
npm test
```

La suite scrive file GGUF reali con header binario completo, li shardizza, li
rilegge dal fabric e verifica che i byte sopravvivano intatti. Controlla anche
che i valori non misurabili restino `null`, che uno shard mancante sia un errore
e non un dato sintetico, e che una misura influenzata dalla page cache porti
sempre un avviso.

### Privacy

Nessun prompt e nessun modello lascia la macchina. L'API è in ascolto solo su
`127.0.0.1` e non è prevista alcuna telemetria remota.

### Licenza

Doppia licenza: **CC BY-NC-SA 4.0** per uso non commerciale, accordo separato
per uso commerciale. Vedi [LICENSE](LICENSE).

Autore: Riccardo Sparacino — [LinkedIn](https://www.linkedin.com/in/riccardo-sparacino/)
