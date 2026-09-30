import json
import os
import streamlit as st
from hosted_news import connection_check, capture_audio, transcribe, make_document, answer, safe_error

st.set_page_config(page_title='Live News · Stream-RAG', page_icon='📻', layout='wide')
st.title('📻 Live News Intelligence')
st.caption('Capture speech. Search its evidence. Get a sourced answer.')
st.info('Hosted mode: Groq speech recognition and generation with BM25 text retrieval. The original MiniLM/Chroma version remains in the repository.')
st.caption('Audio and retrieved excerpts are sent to Groq. Transcripts stay in this session; export them to keep a copy.')
if 'news_docs' not in st.session_state:
    st.session_state.news_docs = []
if not os.environ.get('GROQ_API_KEY', '').strip():
    st.error('GROQ_API_KEY is missing. Add it in Render → Environment, then redeploy.')
    st.stop()
st.success('Groq key is configured. Check connection to verify it.')
if st.button('Check connection', key='news_check'):
    try:
        st.session_state.news_connection = connection_check()
        st.success('Groq authentication passed.')
    except Exception as exc:
        st.error(safe_error(exc))
if 'news_connection' in st.session_state:
    st.caption('Answer model: ' + st.session_state.news_connection['chat_model'] + ' · Speech: whisper-large-v3-turbo')

def add_transcript(text, source):
    st.session_state.news_docs.append(make_document(text, source))
    st.session_state.news_docs = st.session_state.news_docs[-100:]
    st.session_state.pop('news_answer', None)

capture_tab, ask_tab, evidence_tab = st.tabs(['Capture audio', 'Ask questions', 'Transcripts'])
with capture_tab:
    duration = st.selectbox('BBC clip duration', [20, 30, 60])
    if st.button('Capture BBC audio', key='news_capture'):
        try:
            with st.spinner('Capturing and transcribing…'):
                text = transcribe(capture_audio(duration))
                add_transcript(text, 'BBC World Service')
            st.success('Transcript added. Open Ask questions.')
            st.write(text)
        except Exception as exc:
            st.error(safe_error(exc))
    upload = st.file_uploader('Or upload speech (maximum 20 MB)', type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'])
    if st.button('Transcribe uploaded audio', disabled=upload is None):
        try:
            with st.spinner('Transcribing…'):
                text = transcribe(upload.getvalue(), upload.name)
                add_transcript(text, 'Uploaded audio')
            st.success('Transcript added.')
            st.write(text)
        except Exception as exc:
            st.error(safe_error(exc))
    with st.expander('Test with a pasted transcript'):
        pasted = st.text_area('Transcript', max_chars=100000)
        if st.button('Add transcript', disabled=not pasted.strip()):
            add_transcript(pasted, 'User-provided transcript')
            st.success('Transcript added.')
with ask_tab:
    question = st.text_input('Question', placeholder='What were the main topics?', max_chars=2000)
    window = st.selectbox('Received within', ['This session', 'Last 10 minutes', 'Last 30 minutes', 'Last 60 minutes'])
    if st.button('Generate answer', key='news_ask', disabled=not question.strip()):
        try:
            with st.spinner('Searching and generating…'):
                config = st.session_state.get('news_connection') or connection_check()
                st.session_state.news_connection = config
                minutes = {'This session': None, 'Last 10 minutes': 10, 'Last 30 minutes': 30, 'Last 60 minutes': 60}[window]
                st.session_state.news_answer = answer(question, st.session_state.news_docs, config['chat_model'], minutes)
        except Exception as exc:
            st.error(safe_error(exc))
    if 'news_answer' in st.session_state:
        result = st.session_state.news_answer
        st.write(result['answer'])
        st.caption('Model: ' + result['model'] + '. Check claims against these source excerpts.')
        for source in result['sources']:
            with st.expander(source['id'] + ' · ' + source['title']):
                st.write(source['text'])
with evidence_tab:
    st.metric('Session transcripts', len(st.session_state.news_docs))
    st.download_button('Export transcripts', json.dumps(st.session_state.news_docs, indent=2), 'news-transcripts.json', 'application/json')
    if st.button('Clear session transcripts'):
        st.session_state.news_docs = []
        st.session_state.pop('news_answer', None)
        st.rerun()
    for doc in reversed(st.session_state.news_docs):
        with st.expander(doc['title']):
            st.write(doc['text'])
