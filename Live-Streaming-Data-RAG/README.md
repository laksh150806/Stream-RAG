# BBC live-news experiment

For the Theme 4 workspace, run `streamlit run app.py` from the repository root.

Direct launch, after installing this directory's dependencies and FFmpeg:

```bash
streamlit run Live-Streaming-Data-RAG/streaming_rag_app_.py
```

Configure `GROQ_API_KEY` in the environment or Streamlit secrets. Do not put real keys in code.

Models: Whisper `whisper-large-v3-turbo`, local `all-MiniLM-L6-v2` embeddings and `llama-3.3-70b-versatile` through Groq. Retrieval is standard Chroma similarity search; no reranker is configured.

The worker captures 60-second chunks, then transcribes and indexes them, followed by a delay. Recording is not gapless. Stop lets an in-flight capture complete; the worker exits when its UI heartbeat expires. Transcript data persists on disk and must not be mixed into the isolated evaluation corpus.

The earlier project description and upstream attribution link are retained in `docs/original-news-readme.md`. They describe an older NVIDIA-based implementation, not the current setup.
