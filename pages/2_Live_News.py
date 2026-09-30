from pathlib import Path
import runpy
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
st.set_page_config(page_title='BBC Live News · Stream-RAG', page_icon='📻', layout='wide')
st.warning('Optional live-news experiment. This mode downloads BBC audio and stores transcripts locally. Use the main workspace for corpus-isolated Theme 4 evaluation.')
try:
    runpy.run_path(str(ROOT / 'Live-Streaming-Data-RAG/streaming_rag_app_.py'), run_name='__main__')
except ModuleNotFoundError as exc:
    st.info(f'BBC mode needs the optional dependencies ({exc.name} is missing). The main workspace works without these.')
    st.code('pip install -r Live-Streaming-Data-RAG/requirements.txt')
