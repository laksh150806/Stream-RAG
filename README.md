# Stream-RAG

An inspectable prototype for **Theme 4: Streaming Live RAG**. Cumulative transcript fragments trigger early retrieval, compound questions create separate jobs, and late scope changes update dependent claims while retaining unaffected evidence.

The default workspace runs **without an API key or model download**. It uses local BM25 retrieval, English parsing rules, and exact source excerpts. It is not a dense retriever or a generative chatbot. The original BBC/Groq experiment remains available as an optional page.

## Deploy

See [deployment instructions](docs/deployment.md). `render.yaml` configures the key-free workspace for Render. For Streamlit Community Cloud use branch `main` and entrypoint `app.py`. The live-news model requires separate setup and a Groq key.

## Run

Python 3.12 on Linux:

```bash
sh run.sh
```

Or:

```bash
python -m pip install -r requirements.lock.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, normally http://localhost:8501.

```bash
docker compose up --build
```

The Python/Streamlit path is tested. Docker is unavailable in the review environment, so the container command is not verified. The default container includes only the Theme 4 workspace, not BBC dependencies or FFmpeg.

## Try the live flow

1. `Workshop capacity in Pune and cancellation policy and catering options`
2. `Actually city: Delhi`
3. `Please repeat your last answer in two bullets.`

Watch source IDs, answer versions and retrieval counts. The first message is replayed in cumulative fragments; retrieval can start before the final fragment. The second changes city-dependent evidence. The third groups existing evidence without searching. The simulator is labeled and does not record microphone audio. The advanced panel accepts real cumulative transcript fragments and JSONL streams.

## Corpus and event contract

The **17-document synthetic corpus** is for development/demo only. No organizer dataset was supplied. Upload a JSON object with `documents`, or a list of documents:

```json
{"documents":[{"id":"doc-1","title":"Example policy","text":"Verbatim evidence.","metadata":{"city":"Pune","topic":"capacity"}}]}
```

IDs must be unique. Limits: 2 MB upload, 1500 documents, 4000 chunks. Scalar metadata defines filtering scopes. Named corpus values are recognized in text; explicit `key: value` supports unknown values safely. Implicit unseen place names and complex constraints are not automatically understood.

Each event has cumulative text **for that turn**, not a delta. Timestamps must be monotonic across the session; use a new turn ID after `final: true`.

```json
{"timestamp_s":0.8,"turn_id":"1","text":"Workshop capacity in Pune","final":false}
```

Replay without Streamlit or third-party packages:

```bash
python replay.py --corpus data/demo_corpus.json --input data/demo_stream.jsonl
```

Output includes answer versions, query decisions, source IDs, timings and uncertainty. Exact excerpts are checked against session chunks. Inference tokens are zero in extractive mode.

## Validation

```bash
python -m unittest discover -s tests -v
python scripts/evaluate.py
```

UI tests require Streamlit and are skipped without it. The nine-case authored development evaluation passes **6/9 exact source-set checks**. Three paraphrase challenges expose lexical limitations. This is not an official or held-out accuracy score. See [evaluation](docs/evaluation.md) and [results](docs/development-results.json) for the baseline and two ablations.

## Optional BBC live-news mode

The sidebar links to the original news experiment. It needs FFmpeg, optional packages, network access and a Groq key set through `GROQ_API_KEY` or Streamlit secrets. Copy the secrets example and keep the real file untracked.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r Live-Streaming-Data-RAG/requirements.txt
```

Actual components: Whisper `whisper-large-v3-turbo`, MiniLM `all-MiniLM-L6-v2`, Chroma and Groq `llama-3.3-70b-versatile`. No reranker is configured. The misleading toggle/model label were corrected. A queue/Event worker replaces Streamlit access from a background thread. Time filters now implement their stated lookback interval. Transcript indexing failures are surfaced.

BBC mode uses external data and persistent transcripts and is separate from the corpus-isolated evaluation workspace. Full Groq inference remains unverified without credentials.

## Provenance and readiness

This repository began as a file-identical snapshot of `laksh150806/Streaming-Live-RAG`, with a single initial commit. The earlier update ZIP separately retains a backup bundle of the original history. The source repository was not modified.

See [architecture](docs/architecture.md), [dataset card](data/DATASET_CARD.md), [telemetry schema](docs/telemetry.schema.json), and [original news README](docs/original-news-readme.md).

No hackathon gate is certified. Remaining work includes semantic intent/constraint handling, unseen-data evaluation, full Groq testing, Docker validation, a demo video and hosting.
