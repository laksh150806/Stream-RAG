# Deploy Stream-RAG

The repository contains a Render Blueprint at `render.yaml` for the key-free
Streamlit workspace. Connect this repository to Render and deploy that Blueprint.
It installs `requirements.txt`, binds to the assigned `$PORT`, and checks
`/_stcore/health`. The configuration requests a free service; no database or paid
service is provisioned by this configuration.

Alternatively, in Streamlit Community Cloud choose `laksh150806/Stream-RAG`,
branch `main`, entrypoint `app.py`, and Python 3.12.

The default app needs no secrets. It retrieves exact excerpts from the uploaded
or bundled corpus using BM25; it is not the original generative live-news model.
The optional hosted BBC page uses the bundled `imageio-ffmpeg` binary, network
access, and `GROQ_API_KEY`. Never place that key in source control. The heavier
original MiniLM/Chroma experiment still uses its separate requirements file.

Conversation state and uploaded corpora live in the current Streamlit session.
Export traces before restarting. Hosting setup is not proof of a successful
deployment: confirm provider build logs and the health check before sharing a URL.

## Hosted news update

The deployed BBC page includes bundled FFmpeg and the Groq SDK via `requirements.txt`. Set `GROQ_API_KEY` in Render and redeploy. `GROQ_CHAT_MODEL` is optional. `BBC_STREAM_URL` is an optional emergency override when a provider region cannot reach BBC's media CDN. The UI exposes separate Groq and BBC-source checks so failures can be isolated before transcription. This hosted adapter uses BM25 instead of MiniLM/Chroma to keep memory usage small. The original model implementation remains in `Live-Streaming-Data-RAG`. The key-free workspace is unchanged.
