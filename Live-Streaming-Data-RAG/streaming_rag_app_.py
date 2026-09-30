try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
import subprocess
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import time
import threading
from uuid import uuid4
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from news_capture import CaptureWorker
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ============================================================================
# CONFIGURATION
# ============================================================================
st.set_page_config(page_title="Live News Intelligence", page_icon="📡", layout="wide")

def get_secret_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "")

GROQ_API_KEY = get_secret_key()
BBC_STREAM_URL = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"
CHUNK_DURATION = 60
CHANNEL_ID = 0

BASE_DIR = Path("./streaming_rag")
AUDIO_DIR = BASE_DIR / "audio_chunks"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
METADATA_DIR = BASE_DIR / "metadata"
CHROMA_DIR = BASE_DIR / "chroma_langchain"

for d in [AUDIO_DIR, TRANSCRIPT_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if not GROQ_API_KEY:
    st.error("⚠️ **GROQ_API_KEY is missing!**\n\nPlease configure your Groq API Key in **Streamlit Cloud Settings → Secrets**:\n```toml\nGROQ_API_KEY = \"gsk_...\"\n```")
    st.stop()

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================
@st.cache_resource
def init_components(api_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",        
        temperature=0.2, 
        max_tokens=200,
        api_key=api_key
    )
    
    vectorstore = Chroma(
        collection_name="bbc_streaming_rag", 
        embedding_function=embeddings, 
        persist_directory=str(CHROMA_DIR)
    )
    
    reranker = None # Reranking removed for simplicity in local setup
    
    return embeddings, llm, vectorstore, reranker

try:
    embeddings, llm, vectorstore, reranker = init_components(GROQ_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to initialize app components: {e}")
    st.stop()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def index_transcript_langchain(metadata: dict) -> str:
    chunk_id = metadata["chunk_id"]
    transcript = metadata["transcript"]
    doc = Document(
        page_content=transcript,
        metadata={k: v for k, v in metadata.items() if k not in ['transcript']}
    )
    vectorstore.add_documents([doc], ids=[chunk_id])
    return chunk_id

def build_time_filter(minutes_ago: int = None, time_window_minutes: int = 5, channel_id: int = None):
    conditions = []
    if channel_id is not None:
        conditions.append({"channel_id": {"$eq": channel_id}})
    if minutes_ago is not None:
        now_unix = int(datetime.now(timezone.utc).timestamp())
        window_start = now_unix - (minutes_ago * 60)
        window_end = now_unix
        conditions.append({"unix_start": {"$lte": window_end}})
        conditions.append({"unix_end": {"$gte": window_start}})
    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# ============================================================================
# NEW: ENHANCED QUERY WITH RERANKING
# ============================================================================
def query_streaming_rag_with_rerank(
    query_text: str, 
    channel_id: int = None, 
    minutes_ago: int = None, 
    time_window_minutes: int = 5, 
    initial_k: int = 20,  # Retrieve more initially
    top_k: int = 5        # Keep top 5 after reranking
):
    """
    Standard retrieval (Reranking disabled in Groq setup).
    """
    where_filter = build_time_filter(minutes_ago, time_window_minutes, channel_id)
    
    docs = vectorstore.similarity_search(
        query_text, 
        k=top_k, 
        filter=where_filter
    )
    
    return docs

# ============================================================================
# CORE CLASS (Same as before)
# ============================================================================
class StreamingDataRAG:
    def __init__(self):
        self.counter_file = BASE_DIR / "chunk_counter.txt"
        self.chunk_counter = int(self.counter_file.read_text()) if self.counter_file.exists() else 0
        self.session_start = datetime.now(timezone.utc)
    
    def _save_counter(self):
        self.counter_file.write_text(str(self.chunk_counter))
    
    def capture_audio_chunk(self) -> Path:
        self.chunk_counter += 1
        self._save_counter()
        timestamp_utc = datetime.now(timezone.utc)
        ts_str = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
        chunk_file = AUDIO_DIR / f"chunk_{uuid4().hex}_{ts_str}.wav"
        cmd = ["ffmpeg", "-y", "-i", BBC_STREAM_URL, "-t", str(CHUNK_DURATION), 
               "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(chunk_file)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=CHUNK_DURATION + 30, check=True)
            return chunk_file
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError("FFmpeg capture failed. Check the stream URL and FFmpeg installation.") from exc
    
    def transcribe_with_groq(self, audio_file: Path) -> str:
        client = Groq(api_key=GROQ_API_KEY)
        try:
            with open(audio_file, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(audio_file.name, file.read()),
                    model="whisper-large-v3-turbo",
                )
                return transcription.text
        except Exception as e:
            raise RuntimeError("Groq transcription failed; check API configuration and quota.") from e
    
    def save_and_index(self, audio_file: Path, transcript: str) -> dict:
        filename_parts = audio_file.stem.split('_')
        ts_str = filename_parts[-1]
        start_dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(seconds=CHUNK_DURATION)
        chunk_id = audio_file.stem
        metadata = {
            "chunk_id": chunk_id, "channel_id": CHANNEL_ID, "source": "BBC_World_Service",
            "start_time_utc": start_dt.isoformat(), "end_time_utc": end_dt.isoformat(),
            "unix_start": int(start_dt.timestamp()), "unix_end": int(end_dt.timestamp()),
            "duration_seconds": CHUNK_DURATION, "transcript": transcript,
            "char_count": len(transcript), "word_count": len(transcript.split()),
            "audio_file": str(audio_file), "language": "en-US", "asr_model": "whisper-large-v3-turbo"
        }
        (TRANSCRIPT_DIR / f"{chunk_id}.txt").write_text(transcript, encoding='utf-8')
        (METADATA_DIR / f"{chunk_id}.json").write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        index_transcript_langchain(metadata)
        return metadata

# ============================================================================
# UPDATED ask_streaming_rag WITH RERANKING
# ============================================================================
def ask_streaming_rag(
    question: str, 
    channel_id: int = None, 
    minutes_ago: int = None, 
    time_window_minutes: int = 5, 
    use_reranking: bool = True,
    top_k: int = 3
) -> dict:
    """
    Enhanced RAG with optional reranking for better accuracy
    """
    # Use reranking if enabled
    if use_reranking:
        docs = query_streaming_rag_with_rerank(
            query_text=question,
            channel_id=channel_id,
            minutes_ago=minutes_ago,
            time_window_minutes=time_window_minutes,
            initial_k=20,  # Retrieve 20 candidates
            top_k=top_k    # Rerank to top K
        )
    else:
        # Fallback to direct vector search
        where_filter = build_time_filter(minutes_ago, time_window_minutes, channel_id)
        docs = vectorstore.similarity_search(question, k=top_k, filter=where_filter)
    
    if not docs:
        return {
            "question": question, 
            "answer": "No relevant transcripts found for the specified time window.", 
            "sources": [],
            "reranked": False
        }
    
    context_parts = []
    sources = []
    for doc in docs:
        meta = doc.metadata
        time_str = meta['start_time_utc'][11:19]
        context_parts.append(f"[{time_str} UTC - {meta['chunk_id']}]\n{doc.page_content}")
        sources.append({
            "chunk_id": meta['chunk_id'], 
            "time": meta['start_time_utc'], 
            "preview": doc.page_content[:100]
        })
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Original prompt (unchanged)
    prompt = f"""You are an AI assistant analyzing BBC World Service news transcripts. Answer the user's question using only the information in the transcripts. If the information is not present, explicitly say "I don't know."

TRANSCRIPTS:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
Be accurate, concise, and directly answer the question.
Do not add extra background or explanations unless they are needed to answer.
If you use information from a specific segment, mention its time (for example, "In the 09:12 UTC segment…").
If the transcripts do not contain the answer, respond with: "I don't know based on the available transcripts." """
    
    response = llm.invoke(prompt)
    return {
        "question": question, 
        "answer": response.content, 
        "sources": sources, 
        "num_sources": len(sources),
        "reranked": False
    }

# ============================================================================
# BACKGROUND CAPTURE (Same as before)
# ============================================================================
if 'capture_worker' not in st.session_state:
    st.session_state.capture_worker = None
    st.session_state.capture_log = []

def capture_once():
    rag = StreamingDataRAG()
    audio = rag.capture_audio_chunk()
    transcript = rag.transcribe_with_groq(audio)
    if not transcript or not transcript.strip():
        raise RuntimeError("No speech was transcribed from this chunk.")
    return rag.save_and_index(audio, transcript)

def get_db_stats():
    all_docs = vectorstore._collection.get(include=["metadatas"])
    if not all_docs['ids']:
        return None
    unix_times = [(m['unix_start'], m['unix_end']) for m in all_docs['metadatas']]
    return {
        "total_docs": len(all_docs['ids']),
        "total_words": sum(m.get('word_count', 0) for m in all_docs['metadatas']),
        "oldest_time": datetime.fromtimestamp(min(t[0] for t in unix_times), tz=timezone.utc),
        "newest_time": datetime.fromtimestamp(max(t[1] for t in unix_times), tz=timezone.utc),
        "span_minutes": (max(t[1] for t in unix_times) - min(t[0] for t in unix_times)) / 60
    }

# ============================================================================
# STREAMLIT UI (Enhanced with Reranking Toggle)
# ============================================================================
st.markdown("""<style>
.main-header {font-size: 2.5rem; font-weight: 700; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
              -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.stButton>button {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border: none; 
                   padding: 0.75rem; font-weight: 600; border-radius: 8px;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📡 Live News Intelligence</div>', unsafe_allow_html=True)
st.caption("Real-time Streaming Data to RAG")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 System Status")
    stats = get_db_stats()
    if stats:
        st.metric("📚 Transcripts", stats['total_docs'])
        st.metric("💬 Words", f"{stats['total_words']:,}")
        st.metric("⏱️ Span", f"{stats['span_minutes']:.1f} min")
        age = (datetime.now(timezone.utc) - stats['newest_time']).total_seconds() / 60
        st.metric("🕐 Latest", f"{age:.1f} min ago")
    else:
        st.info("No data yet")
    
    st.markdown("---")
    @st.fragment(run_every=2)
    def capture_controls():
        st.markdown("### 🔄 Background Capture")
        worker = st.session_state.capture_worker
        if worker:
            st.session_state.capture_log.extend(worker.drain())
            st.session_state.capture_log = st.session_state.capture_log[-10:]
        running = bool(worker and worker.running)
        # Do not start a second capture while a stopped worker finishes its current chunk.
        alive = bool(worker and worker.thread.is_alive())
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start", disabled=alive, type="primary"):
                worker = CaptureWorker(capture_once)
                st.session_state.capture_worker = worker
                worker.start()
                st.rerun()
        with col2:
            if st.button("⏹️ Stop", disabled=not running):
                worker.stop()
                st.rerun()
        if running:
            st.success("Live capture active")
        elif alive:
            st.info("Stopping after the current chunk…")
        for log in reversed(st.session_state.capture_log[-5:]):
            st.text(log)
    capture_controls()
    st.markdown("---")
    st.markdown("### 🧠 AI Models")
    st.write("**Embeddings:** all-MiniLM-L6-v2 (Local)")
    st.write("**Reranker:** None (Standard Vector Search)")
    st.write("**LLM:** llama-3.3-70b-versatile (Groq)")

tab1, tab2, tab3 = st.tabs(["🔍 Query", "📡 Manual Capture", "📊 Database"])

with tab1:
    st.markdown("### Ask Questions")
    
    # NEW: Reranking toggle
    use_rerank = False
    st.caption("Retrieval: MiniLM vector search. No reranker is configured.")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Question", placeholder="What were the main topics?", label_visibility="collapsed")
    with col2:
        time_filter = st.selectbox("Time", ["All Time", "Last 60m", "Last 30m", "Last 10m"])
    
    time_map = {"All Time": None, "Last 60m": 60, "Last 30m": 30, "Last 10m": 10}
    
    if st.button("🔍 Search", type="primary"):
        if question:
            with st.spinner("Analyzing with reranking..." if use_rerank else "Analyzing..."):
                result = ask_streaming_rag(
                    question, 
                    channel_id=0, 
                    minutes_ago=time_map[time_filter], 
                    time_window_minutes=30, 
                    use_reranking=use_rerank,
                    top_k=5
                )
                
                st.markdown("### 💡 Answer")
                if result.get('reranked'):
                    st.info("✨")
                st.success(result['answer'])
                
                if result['sources']:
                    st.markdown("### 📚 Sources")
                    for i, src in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i} - {src['time'][11:19]} UTC - {src['chunk_id']}"):
                            st.write(src['preview'])

with tab2:
    st.markdown("### 🎙️ Manual Capture")
    col1, col2 = st.columns([1, 1])
    with col1:
        n_chunks = st.number_input("Chunks", 1, 5, 1)
    with col2:
        st.write("")
        worker = st.session_state.capture_worker
        start = st.button("▶️ Capture Now", type="primary", width='stretch', disabled=bool(worker and worker.thread.is_alive()))
    
    if start:
        rag = StreamingDataRAG()
        progress = st.progress(0)
        status = st.empty()
        for i in range(n_chunks):
            status.info(f"📡 Capturing {i+1}/{n_chunks}...")
            progress.progress(i / n_chunks)
            try:
                meta = capture_once()
                with st.expander(f"✅ {meta['chunk_id']} ({meta['word_count']} words)", expanded=True):
                    st.write(meta['transcript'][:300])
            except Exception as exc:
                status.error(str(exc))
                st.stop()
        progress.progress(1.0)
        status.success(f"✅ Done!")
        time.sleep(2)
        st.rerun()

with tab3:
    st.markdown("### 📊 Database")
    stats = get_db_stats()
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📚 Docs", stats['total_docs'])
        col2.metric("💬 Words", f"{stats['total_words']:,}")
        col3.metric("⏱️ Span", f"{stats['span_minutes']:.0f}m")
        col4.metric("📝 Avg", stats['total_words'] // stats['total_docs'])
        st.markdown("---")
        all_docs = vectorstore._collection.get(include=["metadatas", "documents"])
        sorted_docs = sorted(zip(all_docs['ids'], all_docs['metadatas'], all_docs['documents']),
                           key=lambda x: x[1]['unix_start'], reverse=True)[:10]
        for doc_id, meta, text in sorted_docs:
            with st.expander(f"{meta['chunk_id']} - {meta['start_time_utc'][11:19]} UTC ({meta['word_count']} words)"):
                st.write(text)
    else:
        st.info("Database empty")
