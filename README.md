# 📡 Streaming Live RAG

A real-time **Retrieval-Augmented Generation (RAG)** application that listens to the BBC World Service live radio stream, transcribes audio using Groq's Whisper API, stores transcripts in a ChromaDB vector database, and answers natural language questions about the live broadcast.

---

## 🧠 How It Works

```
Live Radio Stream → FFmpeg Records Audio → Groq Whisper Transcribes
      → HuggingFace Embeds Text → ChromaDB Stores Vectors
              → User Asks Question → LLM Answers from Transcripts
```

1. **Background Capture** — FFmpeg records 60-second chunks from the BBC World Service internet radio stream
2. **Transcription** — Each audio chunk is sent to Groq's Whisper API (`whisper-large-v3-turbo`) for speech-to-text
3. **Embedding & Storage** — Transcripts are embedded using a local HuggingFace model and stored in ChromaDB with timestamps
4. **Querying** — When you ask a question, it finds the most relevant transcript chunks via vector similarity search, then passes them to a powerful LLM (via Groq) to generate a grounded answer

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| 🎙️ Speech-to-Text | Groq Whisper (`whisper-large-v3-turbo`) |
| 🧠 LLM | `openai/gpt-oss-120b` via Groq |
| 📐 Embeddings | `all-MiniLM-L6-v2` (local, HuggingFace) |
| 🗄️ Vector Store | ChromaDB (persistent) |
| 🎬 Audio Capture | FFmpeg |
| 🖥️ UI | Streamlit |

---

## 🚀 Quick Start (Local)

### 1. Prerequisites
- Python 3.9+
- FFmpeg installed and on PATH
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Install dependencies
```bash
pip install -r Live-Streaming-Data-RAG/requirements.txt
```

### 3. Set your API key
Either set it as an environment variable:
```bash
export GROQ_API_KEY="your_key_here"
```
Or paste it directly into `streaming_rag_app_.py` (line 21).

### 4. Run the app
```bash
streamlit run Live-Streaming-Data-RAG/streaming_rag_app_.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Fork or clone this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **"New app"**
3. Select your repo and set the **main file path** to:
   ```
   Live-Streaming-Data-RAG/streaming_rag_app_.py
   ```
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy!** — the `packages.txt` file will automatically install FFmpeg on the server.

---

## 🖥️ App Features

### 🔍 Query Tab
- Ask any natural language question about what was discussed on the radio
- Filter by time window (Last 10 min, 30 min, 60 min, or All Time)
- Toggle reranking for improved result accuracy
- See source citations with timestamps

### 📡 Manual Capture Tab
- Manually trigger audio capture without running the background loop
- Capture 1–5 chunks at once and see transcriptions in real time

### 📊 Database Tab
- See total number of transcripts and words indexed
- Browse the 10 most recent transcript chunks

---

## 📁 Project Structure

```
Streaming-Live-RAG/
├── Live-Streaming-Data-RAG/
│   ├── streaming_rag_app_.py   # Main Streamlit application
│   ├── requirements.txt        # Python dependencies
│   ├── packages.txt            # System dependencies (ffmpeg)
│   └── .streamlit/
│       └── secrets.toml        # API key template (do not commit real key!)
└── README.md
```

---

## ⚠️ Notes

- The BBC World Service stream URL (`http://stream.live.vc.bbcmedia.co.uk/bbc_world_service`) is a public internet radio stream — no login or subscription required.
- Transcripts and the ChromaDB database are stored locally in `./streaming_rag/` and persist across sessions.
- The free Groq tier has generous rate limits but may throttle if you capture very frequently.

---

## 📄 License

MIT
