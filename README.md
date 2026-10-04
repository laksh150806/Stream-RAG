# Stream-RAG


**Theme 4 — Streaming Live RAG**

**Retrieve while the user is still speaking. Refine when the intent changes. Answer only from evidence.**

Stream-RAG is a streaming retrieval system built for queries that evolve in real time. It can begin retrieval from stable partial transcripts, split compound requests into independently scoped intents, selectively refresh evidence after late corrections, reuse evidence for presentation-only follow-ups, and expose every retrieval decision with verifiable source IDs.

The main workspace also supports **bring-your-own evidence**: upload a PDF, DOCX, TXT, Markdown, or JSON corpus, activate it as the current workspace, and immediately query the indexed evidence. An optional grounded Groq response turns only the retrieved passages into a concise cited answer; the full corpus is never sent to generation.

## 🔗 Submission links

| Resource | Link |
|---|---|
| 🌐 **Live Demo** | [https://stream-rag.onrender.com](https://stream-rag.onrender.com) |
| 💻 **GitHub Repository** | [github.com/laksh150806/Stream-RAG](https://github.com/laksh150806/Stream-RAG) |
| 🖥️ **Presentation Deck** | [View PPT](./SRM_Snipe%20Coders_Presentation.pptx) |
| 🤖 **AI Usage Disclosure** | [View AI Disclosure](./SRM_Snipe%20Coders_AI_Disclosure.docx) |
| 🎥 **Demo Video** | [Watch on Google Drive](https://drive.google.com/file/d/18O8BbbWUUb8T_4ok6-977WpDluhvMvNU/view?usp=sharing) |


## Why this is different from normal end-of-turn RAG

A normal RAG system waits for the final query. Stream-RAG operates on **cumulative transcript fragments** and maintains an evolving answer.

```text
Incoming transcript fragments
        ↓
Retrieval Controller
WAIT / RETRIEVE / REUSE / SUPPRESS
        ↓
Intent Decomposition + Per-Intent Scope
        ↓
BM25 + Concept Normalization + Metadata Constraints
        ↓
Exact Evidence + Citation Validation
        ↓
Versioned Answer + Telemetry
        ↓
Late correction → selectively refresh affected intents only
```

The core workspace is deterministic and corpus-only: it does not need an API key or model download. It accepts pasted evidence plus PDF, DOCX, TXT, Markdown, and structured JSON uploads. A separate hosted live-audio page demonstrates browser recording/upload → Groq Whisper → persistent ChromaDB transcript retrieval with lexical reranking → sourced Groq generation.

## ✨ What judges can see

- **Bring any document** — upload PDF, DOCX, TXT, Markdown, JSON, or paste text; the active source and indexing status are shown explicitly.
- **Streaming retrieval** — retrieval can start from stable partial speech/transcript input before end-of-utterance.
- **Evidence-first answers** — the strongest retrieved passage is shown first with its exact source ID; secondary evidence stays available without cluttering the answer.
- **Grounded response generation** — optional Groq generation sees only retrieved evidence and returned citation IDs are checked against that evidence.
- **Live corrections** — late scope changes selectively refresh affected intents instead of restarting every search.
- **Inspectable behavior** — controller decisions, answer versions, retrieval counts, scopes, timing and the technical trace remain visible to judges.

## ⚙️ Core capabilities

- **Early retrieval** — stable partial clauses can trigger retrieval before end-of-utterance.
- **Multi-intent decomposition** — compound requests become separate retrieval jobs.
- **Multi-sentence independent scopes** — one utterance can ask for Pune capacity, Delhi cancellation, and Bengaluru catering without collapsing everything into one global city.
- **Selective late refinement** — a correction such as `Actually city: Delhi` only re-runs intents that depend on the changed city.
- **Presentation-only suppression** — requests such as `repeat the last answer in two bullets` reuse evidence and execute no new corpus search.
- **Paraphrase-aware lexical retrieval** — concept normalization handles common equivalents such as `delegates → people`, `fit → capacity`, `called off → cancellation`, and `overseas → international`.
- **Grounded evidence** — returned claims are exact corpus excerpts with source IDs.
- **Optional grounded AI answer** — when Groq is configured, generation receives only already-retrieved evidence, cites retrieved source IDs, and is blocked from silently citing unretrieved sources.
- **Citation guard** — source IDs and quotes are mechanically validated before output.
- **Session-only state** — conversational state stays inside the active session.
- **Observability** — the UI exposes controller state, answer version, search count, active intents, early-retrieval lead, intent scopes, a live decision timeline, and the full event trace.
- **Bring-your-own evidence** — upload PDF, DOCX, TXT, Markdown, or JSON; long text is converted into overlapping retrieval-sized evidence sections.
- **Visible indexing status** — the workspace reports indexed evidence sections and searchable chunks immediately after ingestion.
- **Exportable telemetry** — session events and answer versions can be downloaded as JSON.

## 🚀 Recommended judge demo

### A. Prove it is not hardcoded

1. Upload an unrelated PDF/DOCX/TXT file in **Build your evidence workspace**.
2. Confirm the selected filename, then choose **Use uploaded evidence**.
3. Verify the sidebar shows the file as **ACTIVE** and reports indexed sections/chunks.
4. Ask a factual question whose answer exists only in that document.
5. Show **BEST RETRIEVED EVIDENCE**, its source ID, then choose **Generate grounded answer** to show the concise cited response.

This is the strongest first demo because it demonstrates arbitrary evidence ingestion, retrieval provenance and evidence-only generation.

### B. Show the streaming controller

Load the bundled sample dataset, reset the conversation, then run the sequence below.

Open the [live application](https://stream-rag.onrender.com).

#### 1. Early retrieval + decomposition

```text
Workshop capacity in Pune and cancellation policy and catering options and accessibility
```

Watch the controller and decision timeline. Retrieval begins on stable partial input before the final transcript fragment.

#### 2. Selective refinement

```text
Actually city: Delhi
```

Capacity, cancellation, and catering switch to Delhi evidence. Accessibility stays unchanged because it does not depend on the city.

#### 3. Retrieval suppression

```text
Please repeat your last answer in two bullets.
```

The controller shows **SUPPRESS** and the search count does not increase.

#### 4. Independent scopes in one utterance

```text
What is workshop capacity in Pune? What is the cancellation policy in Delhi? What are catering options in Bengaluru?
```

Expected intent scopes:

```text
capacity      → city: Pune
cancellation  → city: Delhi
catering      → city: Bengaluru
```

## 📊 Verified development evaluation

Run:

```bash
python -m unittest discover -s tests -v
python scripts/evaluate.py
```


| Configuration | Exact final source-set checks |
|---|---:|
| **Full Stream-RAG** | **10 / 10** |
| End-of-turn baseline | **10 / 10** |
| No decomposition | **7 / 10** |
| No selective refinement | **9 / 10** |

For the authored development set:

| Early-retrieval behavior | Cases with provisional retrieval |
|---|---:|
| **Stream-RAG** | **10 / 10** |
| End-of-turn baseline | **0 / 10** |

The important result is **not** that streaming improves final accuracy on this small set: both full streaming and end-of-turn reach 10/10 final source accuracy. The measured advantage is that Stream-RAG begins retrieval before the user finishes, while the end-of-turn baseline waits.

These are **10 self-authored synthetic development cases**, not an official or held-out benchmark. See [evaluation details](docs/evaluation.md) and [development results](docs/development-results.json).

## 🔍 Example paraphrases

```text
Could our thirty delegates fit in Cedar Hall?
→ pune-capacity:s1

What would we forfeit if we called off the gathering in Delhi?
→ delhi-cancellation:s1

What paperwork is needed to claim an overseas journey?
→ international-travel:s1
```

Unsupported requests do not invent evidence:

```text
Quantum banana teleportation
→ No supporting evidence found in this corpus.
```

## 🧠 Retrieval and state model

The bundled demo corpus contains **17 synthetic documents** covering workshop policies and travel reimbursement. It is development/demo data only.

The default workspace uses:

- local BM25 retrieval;
- concept normalization;
- metadata-aware constraints;
- rule-based English intent decomposition;
- per-intent scopes;
- exact source excerpts;
- deterministic citation verification.

It is deliberately inspectable. It is **not** presented as a dense semantic retriever or as a generative chatbot.

The sidebar can ingest **PDF, DOCX, TXT, Markdown, or JSON** directly. Text-bearing documents are converted into overlapping retrieval-sized evidence sections; scanned/image-only PDFs require OCR first. Structured JSON can use the format below:

```json
{
  "documents": [
    {
      "id": "doc-1",
      "title": "Example policy",
      "text": "Verbatim evidence.",
      "metadata": {
        "city": "Pune",
        "topic": "capacity"
      }
    }
  ]
}
```

Limits: 2 MB upload, 1500 documents, 4000 chunks. Document IDs must be unique and metadata values must be scalar.

## 🌊 Streaming event contract

Each input event contains the cumulative transcript **for that turn**, not only the latest delta.

```json
{
  "timestamp_s": 0.8,
  "turn_id": "1",
  "text": "Workshop capacity in Pune",
  "final": false
}
```

Timestamps must be monotonic across the session. After a turn is finalized, use a new `turn_id`.

Replay without the Streamlit UI:

```bash
python replay.py --corpus data/demo_corpus.json --input data/demo_stream.jsonl
```

Exports include answer versions, controller decisions, retrieval timings, source IDs, uncertainty, and the event trace.

## 🎙️ Hosted live-audio / news extension

The sidebar **Live audio / news RAG** page provides the stable hosted flow:

```text
Browser recording or audio upload
        ↓
Groq Whisper transcription
        ↓
Persistent ChromaDB transcript store
        ↓
Vector retrieval + optional BM25 reciprocal-rank reranking
        ↓
Groq sourced answer generation
```

It also accepts pasted transcripts, stores timestamped transcript chunks in the ChromaDB collection, supports recency filtering and recent-chunk inspection, and keeps generation grounded in the retrieved transcript evidence.

Set `GROQ_API_KEY` through Render Environment settings or Streamlit secrets. Do **not** commit secrets to the repository.

Direct BBC World Service server capture remains under an **Experimental** expander because some cloud-hosting regions cannot access BBC media CDN routes. It is not required for the main demo.

## 🛠️ Run locally

Python 3.12 on Linux:

```bash
sh run.sh
```

Or:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The local Streamlit URL is normally:

```text
http://localhost:8501
```

Docker:

```bash
docker compose up --build
```


## ☁️ Deployment

The hosted demo runs on Render.

- Branch: `main`
- Entrypoint: `app.py`
- Build command: `pip install -r requirements.txt`
- Start command:

```bash
python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

Additional details: [docs/deployment.md](docs/deployment.md).

## 📁 Important repository files

| Path | Purpose |
|---|---|
| `app.py` | Main Theme 4 Stream-RAG workspace |
| `rag_core.py` | Streaming controller, decomposition, retrieval, refinement, citation guard |
| `pages/2_Live_News.py` | Hosted browser-audio/news extension |
| `hosted_news.py` | Whisper/Groq transcript RAG utilities |
| `data/demo_corpus.json` | 17-document synthetic demo corpus |
| `data/demo_stream.jsonl` | Reproducible streaming demo |
| `scripts/evaluate.py` | Authored development evaluation + ablations |
| `docs/development-results.json` | Latest verified development results |
| `docs/evaluation.md` | Evaluation interpretation and limitations |
| `docs/architecture.md` | Architecture notes |
| `docs/telemetry.schema.json` | Telemetry/event schema |
| `tests/` | Core, UI, hosted-news, and regression tests |

## ⚠️ Current limitations

- The core retriever remains lexical BM25 with concept normalization rather than a dense semantic retriever.
- English decomposition and scope handling are rule-based.
- The 10-case evaluation is authored and synthetic; it does not establish general or official benchmark accuracy.
- Complex negation and truly unfamiliar paraphrases may still fail.
- Direct server-side BBC capture is hosting-region dependent.

## 📚 Project documentation

- [Architecture](docs/architecture.md)
- [Evaluation](docs/evaluation.md)
- [Development results](docs/development-results.json)
- [Dataset card](data/DATASET_CARD.md)
- [Telemetry schema](docs/telemetry.schema.json)
- [Deployment guide](docs/deployment.md)
- [Original live-news experiment notes](docs/original-news-readme.md)

## 🤖 AI disclosure

The project AI-usage disclosure is provided here:

**[AI Usage Disclosure](./SRM_Snipe%20Coders_AI_Disclosure.docx)**

