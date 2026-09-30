import json

import streamlit as st

from hosted_news import answer, capture_audio, connection_check, key_configured, make_document, safe_error, transcribe

st.set_page_config(page_title='Live Audio · Stream-RAG', page_icon='🎙️', layout='wide')
st.title('🎙️ Live Audio & News Intelligence')
st.caption('Record or upload speech. Transcribe it with Whisper. Ask grounded questions from the transcript.')
st.info(
    'Stable hosted path: browser recording or uploaded audio → Groq Whisper → BM25 retrieval → sourced answer. '
    'Direct BBC server capture is kept only as an experimental option because some cloud regions block BBC media CDNs.'
)
st.caption(
    'Audio and retrieved excerpts are sent to Groq. '
    'Transcripts stay in this browser session; export them to keep a copy.'
)

if 'news_docs' not in st.session_state:
    st.session_state.news_docs = []

if not key_configured():
    st.error('GROQ_API_KEY is missing. Add it in Render → Environment or Streamlit secrets, then redeploy.')
    st.stop()

st.success('Groq key is configured.')

if st.button('Check AI connection', key='news_check'):
    try:
        st.session_state.news_connection = connection_check()
        st.success('Groq authentication passed.')
    except Exception as exc:
        st.error(safe_error(exc))

if 'news_connection' in st.session_state:
    config = st.session_state.news_connection
    st.caption(
        'Answer model: '
        + config['chat_model']
        + ' · Speech: '
        + config.get('speech_model', 'whisper-large-v3-turbo')
    )


def add_transcript(text, source):
    st.session_state.news_docs.append(make_document(text, source))
    st.session_state.news_docs = st.session_state.news_docs[-100:]
    st.session_state.pop('news_answer', None)


capture_tab, ask_tab, evidence_tab = st.tabs(['Transcribe audio', 'Ask questions', 'Transcripts'])

with capture_tab:
    st.subheader('1. Add speech')
    record_col, upload_col = st.columns(2)

    with record_col:
        st.markdown('**Record in the browser**')
        recorded = st.audio_input('Record a short voice or news clip', key='news_recording')
        if st.button('Transcribe recording', disabled=recorded is None, type='primary'):
            try:
                with st.spinner('Transcribing recording…'):
                    text = transcribe(recorded.getvalue(), 'browser-recording.wav')
                    add_transcript(text, 'Browser recording')
                st.success('Transcript added.')
                st.write(text)
            except Exception as exc:
                st.error(safe_error(exc))

    with upload_col:
        st.markdown('**Upload an audio file**')
        upload = st.file_uploader(
            'Maximum 20 MB',
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'],
            key='news_upload',
        )
        if st.button('Transcribe uploaded audio', disabled=upload is None):
            try:
                with st.spinner('Transcribing upload…'):
                    text = transcribe(upload.getvalue(), upload.name)
                    add_transcript(text, 'Uploaded audio')
                st.success('Transcript added.')
                st.write(text)
            except Exception as exc:
                st.error(safe_error(exc))

    with st.expander('Add a pasted transcript instead'):
        pasted = st.text_area('Transcript', max_chars=100000)
        if st.button('Add transcript', disabled=not pasted.strip()):
            add_transcript(pasted, 'User-provided transcript')
            st.success('Transcript added.')

    st.divider()
    with st.expander('🧪 Experimental: capture BBC World Service from the server'):
        st.warning(
            'Optional experiment only. BBC media CDNs can reject or block cloud-hosted servers. '
            'Use browser recording or upload for the reliable demo path.'
        )
        duration = st.selectbox('BBC clip duration', [20, 30, 60], key='bbc_duration')

        check_col, capture_col = st.columns(2)
        with check_col:
            if st.button('Test BBC source', key='news_source_check'):
                try:
                    with st.spinner('Testing BBC source…'):
                        sample = capture_audio(5)
                    st.success(f'BBC source reachable ({max(1, len(sample) // 1024)} KB captured).')
                except Exception as exc:
                    st.warning(safe_error(exc))

        with capture_col:
            if st.button('Capture + transcribe BBC', key='news_capture'):
                try:
                    with st.spinner('Capturing and transcribing…'):
                        audio = capture_audio(duration)
                        text = transcribe(audio)
                        add_transcript(text, 'BBC World Service')
                    st.success('Transcript added.')
                    st.write(text)
                except Exception as exc:
                    st.warning(safe_error(exc))

with ask_tab:
    st.subheader('2. Ask from the transcript evidence')
    if not st.session_state.news_docs:
        st.info('Record, upload, or paste a transcript first.')

    question = st.text_input('Question', placeholder='What were the main topics?', max_chars=2000)
    window = st.selectbox(
        'Received within',
        ['This session', 'Last 10 minutes', 'Last 30 minutes', 'Last 60 minutes'],
    )
    if st.button('Generate sourced answer', key='news_ask', disabled=not question.strip()):
        try:
            with st.spinner('Searching transcript evidence…'):
                config = st.session_state.get('news_connection') or connection_check()
                st.session_state.news_connection = config
                minutes = {
                    'This session': None,
                    'Last 10 minutes': 10,
                    'Last 30 minutes': 30,
                    'Last 60 minutes': 60,
                }[window]
                st.session_state.news_answer = answer(
                    question,
                    st.session_state.news_docs,
                    config['chat_model'],
                    minutes,
                )
        except Exception as exc:
            st.error(safe_error(exc))

    if 'news_answer' in st.session_state:
        result = st.session_state.news_answer
        st.markdown('### Answer')
        st.write(result['answer'])
        st.caption('Model: ' + result['model'] + '. Verify claims against the source excerpts below.')
        for source in result['sources']:
            with st.expander(source['id'] + ' · ' + source['title']):
                st.write(source['text'])

with evidence_tab:
    st.metric('Session transcripts', len(st.session_state.news_docs))
    st.download_button(
        'Export transcripts',
        json.dumps(st.session_state.news_docs, indent=2),
        'news-transcripts.json',
        'application/json',
    )
    if st.button('Clear session transcripts'):
        st.session_state.news_docs = []
        st.session_state.pop('news_answer', None)
        st.rerun()

    if not st.session_state.news_docs:
        st.info('No transcripts in this session yet.')

    for doc in reversed(st.session_state.news_docs):
        with st.expander(doc['title']):
            st.write(doc['text'])
