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

Some clients stop reading without closing the socket, which is not observable as
a disconnect. Two things cover that: a new request supersedes the one in flight,
and `POST /v1/generation/stop` frees the GPU on demand without restarting
anything.

**Choosing a model for agent work.** A *Thinking* variant reasons before every
answer and cannot be told not to — `enable_thinking`, `reasoning_budget: 0` and
`/no_think` are all ignored by a model trained to always reason. For an 80B at
19 tok/s that is minutes of deliberation per step, and the reasoning consumes the
context an agent needs for its own instructions. Prefer an *Instruct* variant for
IDE integration and keep the Thinking one for questions where the wait pays off.

Whatever base URL you configure works: `.../v1`, `.../api`, or the bare host.
Providers append `/api/chat` to whatever you typed, and the resulting duplicated
prefix is normalised rather than answered with an HTML 404 that surfaces inside
the IDE as an unreadable error.

### Runtime tuning

First, why a 45 GB model runs at all on a machine with 31.7 GB of RAM. The
engine memory-maps the weights rather than reading them, so loading costs
seconds regardless of file size and the pages fault in only as they are touched.
On a Mixture-of-Experts model that is the whole game: each token reaches a
handful of experts, so the resident working set is the experts that keep coming
up, not the file. The model does not fit; the part of it being used does.

Two consequences run through everything below. Load time says nothing about
model size, so it cannot be used to tell a real restart from a reused engine.
And the first generation after a load is reading from disk while later ones read
from memory, which is why every measurement here reports more than one run.

Every launch parameter has an automatic value and a user override, shown side
by side with the reason for the automatic choice. The knob that matters on very
large models is **experts routed per token**: a Mixture-of-Experts model sends
each token to a handful of its experts, and that handful — not the total
parameter count — is what has to be read per token. Lowering it cuts bytes per
token proportionally, and output quality with it.

Measured on Qwen3-Next-80B (512 experts, 10 routed by default). Cold is the
first generation after loading; warm is the second, with the hot experts already
resident:

| Experts per token | Cold | Warm |
|---|---|---|
| 10 (model default) | 4.4 tok/s | 10.3 tok/s |
| 2 | 8.7 tok/s | 15.4 tok/s |

The cold/warm gap is worth internalising before drawing conclusions from any
single number: the same configuration more than doubles once the page cache has
the experts it keeps reaching for.

There is a floor, and it is worth understanding before reaching for the slider.
On GLM-4.6, going from 8 experts to 1 moves active parameters from 34.9B to
20.1B — not to 4B — because attention, the shared expert and the embeddings are
read for every token whatever the routing does. Only the expert weights scale.

The second lever is **where** the experts live rather than how many run: pinned
to the CPU across every layer, the GPU holds only the small dense path. Unlike
expert reduction it changes no output at all, so it is the first thing to try —
but the numbers this README used to quote for it were wrong, and the story of
why is worth more than the numbers were.

The engine was being reused across configuration changes. `initialize()` matched
on the model path alone, so a settings change was accepted, reported as loaded,
and silently ignored: the server kept running with the arguments it was born
with. Every "faster" reading was the same engine, warmer. The tell was a 45 GB
model reporting a 6-second load, and a cold run that came out *faster* than the
warm one after it. Reuse now compares the full argument signature.

The lesson generalises past this bug: a warm cache flatters whatever ran last,
so a sweep that never restarts measures its own ordering. Compare within a single
sweep, watch the load time, and distrust any configuration change that was free.

**"Automatic" means the engine's own automatic wherever it already decides
well.** An earlier version of this planner computed a GPU layer count from
total-bytes-over-layer-count and imposed it, alongside forcing every expert onto
the CPU and halving the batch sizes. llama.cpp's `-ngl` already defaults to
`auto` and decides from real tensor sizes against live VRAM — information no
estimate here can match. The planner now imposes a value only where it can
justify one, and offers the rest as choices with their trade-offs written out.

Two theories that sounded right and did not survive measurement, both since
removed from the automatic path:

- *A narrow context should help a disk-bound model, leaving more memory for
  weights.* At 8192 the same model ran slightly **slower** than at 32768. The KV
  cache is small next to the weights, and a wider window reuses more prompt.
- *Smaller batches should stop prompt processing from thrashing the cache.*
  No measured benefit; the engine's defaults held up.

Both remain available as user choices, with what was measured stated next to
them. A guess that costs throughput is not an optimisation, and the honest place
for one is behind a switch the user controls.

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

**Models published across several files are handled as one model.** Anything
past 50 GB arrives as an ordered set, and each part's header describes only the
tensors it holds, with offsets into itself. Reading the first part alone reports
a fraction of the model and would copy bytes from the wrong file for everything
else, so parsing, inspection and sharding all follow each tensor to the part it
actually lives in. Engine startup is given a budget derived from total size
rather than a fixed timeout — a 220 GB set needs minutes just to fault in, and a
fixed limit killed exactly the large models this runtime exists for.

### Active Weight Generation Runtime & Pipeline Measurement

AILOFlow features the **AILOFlow Hierarchical Engine (`ailo-hierarchical`)**, executing the complete **DwarfStar architecture** directly during text generation:

```text
             AILOFlow Runtime
                    │
          Hierarchical Memory
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
    VRAM           RAM           SSD
      │             │             │
      └─────────────┼─────────────┘
                    ↓
              ACTIVE WEIGHTS
                    ↓
                COMPUTE
                    ↓
                  TOKEN
```

- **Active Weight Execution (`ailo-hierarchical`)**: Directly streams `.sflow` container shards or GGUF tensors from the Storage Fabric through the `HierarchicalCache` (VRAM -> RAM -> SSD) and `PrefetchEngine` into active layer compute passes for token generation.
- **llama.cpp Engine Integration (`llama.cpp`)**: External backend option for running pure GGUF models directly via native binaries when full memory offloading is preferred.
- **Pipeline Benchmark (`npm run cli -- benchmark`)**: Sweeps layer tensors across disk shards to measure raw aggregate storage delivery throughput and prefetch hit rate under maximum I/O pressure.

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

Alcuni client smettono di leggere senza chiudere il socket, cosa che non è
osservabile come disconnessione. Due cose lo coprono: una nuova richiesta
sostituisce quella in corso, e `POST /v1/generation/stop` libera la GPU su
richiesta senza riavviare nulla.

**Scegliere il modello per il lavoro agentico.** Una variante *Thinking* ragiona
prima di ogni risposta e non le si può dire di non farlo: `enable_thinking`,
`reasoning_budget: 0` e `/no_think` vengono tutti ignorati da un modello
addestrato a ragionare sempre. Per un 80B a 19 tok/s significa minuti di
riflessione a ogni passo, e quel ragionamento consuma il contesto che serve
all'agente per le proprie istruzioni. Per l'integrazione negli IDE conviene una
variante *Instruct*, tenendo la Thinking per le domande in cui l'attesa ripaga.

Qualunque base URL tu configuri funziona: `.../v1`, `.../api` o l'host nudo. I
provider aggiungono `/api/chat` a ciò che hai scritto, e il prefisso duplicato
che ne risulta viene normalizzato invece di produrre un 404 HTML che dentro
l'IDE appare come un errore illeggibile.

### Regolazione del runtime

Prima di tutto, perché un modello da 45 GB gira su una macchina con 31,7 GB di
RAM. Il motore mappa in memoria i pesi invece di leggerli, quindi il caricamento
costa secondi indipendentemente dalla dimensione del file e le pagine entrano
solo quando vengono toccate. Su un Mixture-of-Experts è tutto lì: ogni token
raggiunge una manciata di esperti, quindi il working set residente sono gli
esperti che continuano a ricorrere, non il file. Il modello non ci sta; ci sta
la parte che viene usata.

Da qui due conseguenze che attraversano tutto il resto. Il tempo di caricamento
non dice nulla sulla dimensione del modello, quindi non si può usare per
distinguere un riavvio vero da un motore riusato. E la prima generazione dopo un
caricamento legge da disco mentre le successive leggono da memoria: per questo
ogni misura qui riporta più di una corsa.

Ogni parametro di avvio ha un valore automatico e un override utente, mostrati
affiancati con il motivo della scelta automatica. La manopola che conta sui
modelli enormi è **esperti attivati per token**: un modello Mixture-of-Experts
instrada ogni token verso una manciata dei suoi esperti, e quella manciata — non
il numero totale di parametri — è ciò che va letto per ogni token. Ridurla taglia
i byte per token in proporzione, e con essi la qualità dell'output.

Misurato su Qwen3-Next-80B (512 esperti, 10 instradati di serie). "Freddo" è la
prima generazione dopo il caricamento, "caldo" la seconda, con gli esperti
richiesti già residenti:

| Esperti per token | Freddo | Caldo |
|---|---|---|
| 10 (valore del modello) | 4,4 tok/s | 10,3 tok/s |
| 2 | 8,7 tok/s | 15,4 tok/s |

Il divario freddo/caldo va tenuto a mente prima di trarre conclusioni da un
singolo numero: la stessa configurazione più che raddoppia quando la page cache
ha gli esperti che il modello continua a richiedere.

Esiste però un pavimento, e conviene conoscerlo prima di toccare il cursore. Su
GLM-4.6, passare da 8 esperti a 1 porta i parametri attivi da 34,9B a 20,1B —
non a 4B — perché attenzione, esperto condiviso ed embedding vengono letti a
ogni token qualunque cosa faccia il routing. Solo i pesi degli esperti scalano.

La seconda leva è **dove** stanno gli esperti, non quanti ne girano: fissati
sulla CPU su tutti i layer, la GPU tiene solo il piccolo percorso denso. A
differenza della riduzione degli esperti non cambia una virgola dell'output, ed
è quindi la prima cosa da provare — ma le cifre che questo README dava per essa
erano sbagliate, e il perché vale più delle cifre.

Il motore veniva riusato tra un cambio di configurazione e l'altro.
`initialize()` confrontava solo il percorso del modello, quindi una modifica
veniva accettata, dichiarata caricata e ignorata in silenzio: il server
continuava con gli argomenti con cui era nato. Ogni lettura "più veloce" era lo
stesso motore, più caldo. L'indizio era un modello da 45 GB che dichiarava 6
secondi di caricamento, e una corsa a freddo risultata *più veloce* di quella a
caldo successiva. Ora il riuso confronta l'intera firma degli argomenti.

La lezione va oltre il bug: una cache calda favorisce ciò che ha girato per
ultimo, quindi una sequenza che non riavvia misura il proprio ordine. Confrontare
dentro una sola sequenza, guardare il tempo di caricamento, e diffidare di ogni
cambio di configurazione che è risultato gratis.

**"Automatico" significa l'automatico del motore ovunque esso decida già bene.**
Una versione precedente di questo pianificatore calcolava il numero di layer su
GPU dividendo i byte totali per i layer e lo imponeva, oltre a forzare tutti gli
esperti su CPU e a dimezzare i batch. `-ngl` di llama.cpp ha già come default
`auto` e decide dalle dimensioni reali dei tensori contro la VRAM libera —
informazione che nessuna stima fatta qui può eguagliare. Ora il pianificatore
impone un valore solo dove sa giustificarlo, e offre il resto come scelte con i
compromessi scritti accanto.

Due teorie che sembravano giuste e non hanno retto alla misura, entrambe rimosse
dal percorso automatico:

- *Un contesto stretto dovrebbe aiutare un modello su disco, lasciando più
  memoria ai pesi.* A 8192 lo stesso modello è andato leggermente **più piano**
  che a 32768. La cache KV è piccola accanto ai pesi, e una finestra più larga
  riusa più prompt.
- *Batch più piccoli dovrebbero evitare che l'elaborazione del prompt sporchi la
  cache.* Nessun beneficio misurato; i default del motore hanno retto.

Restano disponibili come scelte dell'utente, con accanto scritto ciò che è stato
misurato. Una supposizione che costa throughput non è un'ottimizzazione, e il
posto onesto per una supposizione è dietro un interruttore che decide l'utente.

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

**I modelli pubblicati in più file sono trattati come un modello solo.** Tutto
ciò che supera i 50 GB arriva come insieme ordinato, e l'header di ogni parte
descrive soltanto i tensori che quella parte contiene, con offset relativi a se
stessa. Leggere la prima parte da sola riporta una frazione del modello e
copierebbe byte dal file sbagliato per tutto il resto, quindi analisi, scheda e
sharding seguono ogni tensore fino alla parte in cui vive davvero. L'avvio del
motore riceve un budget calcolato sulla dimensione totale invece di un timeout
fisso: un insieme da 220 GB richiede minuti solo per essere mappato in memoria,
e un limite fisso uccideva proprio i modelli grandi per cui questo runtime
esiste.

### Runtime a Pesi Attivi & Misurazione della Pipeline

AILOFlow integra the **Motore Gerarchico AILOFlow (`ailo-hierarchical`)**, che realizza direttamente l'architettura **DwarfStar** durante la generazione del testo:

```text
             AILOFlow Runtime
                    │
          Hierarchical Memory
                    │
      ┌─────────────┼─────────────┐
      ↓             ↓             ↓
    VRAM           RAM           SSD
      │             │             │
      └─────────────┼─────────────┘
                    ↓
              ACTIVE WEIGHTS
                    ↓
                COMPUTE
                    ↓
                  TOKEN
```

- **Esecuzione a Pesi Attivi (`ailo-hierarchical`)**: Esegue lo streaming dei tensori dai container `.sflow` o file GGUF dallo Storage Fabric attraverso la `HierarchicalCache` (VRAM -> RAM -> SSD) e il `PrefetchEngine` verso i passaggi di calcolo dei layer attivi per la generazione dei token.
- **Integrazione Motore llama.cpp (`llama.cpp`)**: Opzione backend esterna per eseguire modelli GGUF nativi tramite binario quando si sceglie l'offload di memoria standard.
- **Benchmark della Pipeline (`npm run cli -- benchmark`)**: Scansiona i tensori dai vari drive per misurare la banda di storage aggregata e l'hit rate di prefetch sotto massimo carico I/O.

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
