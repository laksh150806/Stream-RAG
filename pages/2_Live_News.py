import json
import streamlit as st

from hosted_news import (
    add_transcript_to_db,
    answer_db,
    capture_audio,
    clear_db,
    connection_check,
    get_db_stats,
    get_recent_chunks,
    key_configured,
    safe_error,
    transcribe,
)

st.set_page_config(page_title='Live Audio · Stream-RAG', page_icon='🎙️', layout='wide')

st.title('🎙️ BBC World Service Live Stream RAG')
st.caption('Real-time Retrieval-Augmented Generation (RAG) on live broadcast radio.')

st.info(
    '📻 **Stream URL:** `http://stream.live.vc.bbcmedia.co.uk/bbc_world_service` (Public internet radio stream)\n\n'
)

if not key_configured():
    st.error('GROQ_API_KEY is missing. Add it in Render → Environment or Streamlit secrets (.streamlit/secrets.toml), then reload.')
    st.stop()

# Avoid a blocking Groq model-list request every time a fresh Live News page opens.
# The explicit button below performs the remote authentication/model check when needed.
if 'news_connection' not in st.session_state:
    st.session_state.news_connection = None

config = st.session_state.news_connection

# Keep an explicit, user-triggered authentication check. Besides making the
# current model selection visible, this preserves the stable UI contract used
# by the hosted-page smoke test without ever displaying the API key.
if st.button('Check Groq connection', key='news_check'):
    try:
        config = connection_check()
        st.session_state.news_connection = config
        st.success('Groq API key is configured. Authentication will be verified on the first transcription or generated answer.')
    except Exception as exc:
        st.error(safe_error(exc))

query_tab, capture_tab, db_tab = st.tabs(['🔍 Query', '📡 Manual Capture', '📊 Database'])

# -----------------------------------------------------------------------------
# TAB 1: 🔍 Query Tab
# -----------------------------------------------------------------------------
with query_tab:
    st.subheader('🔍 Ask Questions about Live Radio Broadcast')
    st.caption('Ask any natural language question about what was discussed on the BBC World Service stream.')

    question = st.text_input(
        'Natural Language Question',
        placeholder='What were the main news topics discussed on the radio?',
        max_chars=2000,
        key='query_input',
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        time_window = st.selectbox(
            'Time Window Filter',
            ['Last 10 min', 'Last 30 min', 'Last 60 min', 'All Time'],
            index=3,
            key='time_window_select',
        )

    with col2:
        st.write(' ')
        st.write(' ')
        enable_reranking = st.toggle(
            'Toggle Reranking',
            value=True,
            help='Rerank transcript results using BM25 lexical scoring for improved accuracy',
            key='rerank_toggle',
        )

    if st.button('🔍 Search & Generate Answer', type='primary', disabled=not question.strip(), key='btn_query'):
        try:
            with st.spinner('Searching the transcript index & generating a grounded answer…'):
                config = st.session_state.get('news_connection') or connection_check()
                st.session_state.news_connection = config

                minutes_map = {
                    'Last 10 min': 10,
                    'Last 30 min': 30,
                    'Last 60 min': 60,
                    'All Time': None,
                }
                minutes = minutes_map[time_window]

                result = answer_db(
                    question=question,
                    model=config['chat_model'],
                    minutes=minutes,
                    rerank=enable_reranking,
                )
                st.session_state.news_query_result = result
        except Exception as exc:
            st.error(safe_error(exc))

    if 'news_query_result' in st.session_state:
        result = st.session_state.news_query_result
        st.divider()
        st.markdown('### 💡 Answer')
        st.write(result['answer'])

        rerank_status = 'Enabled (Hybrid Vector + BM25)' if result.get('rerank') else 'Disabled (Vector Distance)'
        st.caption(f"**Model:** {result['model']} | **Reranking:** {rerank_status} | **Window:** {time_window}")

        st.markdown('#### 📚 Source Citations')
        if not result['sources']:
            st.info('No matching transcript citations returned for this query.')
        else:
            for idx, source in enumerate(result['sources'], 1):
                score_label = f"Rerank Score: {source.get('rerank_score')}" if 'rerank_score' in source else f"Vector Score: {source.get('vector_score')}"
                with st.expander(f"[{source['id']}] {source['title']} — {score_label}"):
                    st.write(source['text'])
                    st.caption(f"**Chunk ID:** `{source['id']}` | **Word Count:** {source['metadata'].get('word_count', 0)}")

# -----------------------------------------------------------------------------
# TAB 2: 📡 Manual Capture Tab
# -----------------------------------------------------------------------------
with capture_tab:
    st.subheader('🎙️ Live Audio Capture & Transcription')
    st.caption('Record BBC World Service or any live audio in your browser, transcribe it with Groq Whisper, and make it immediately queryable by the RAG pipeline.')

    st.markdown('### 🌐 Browser Audio Capture')
    st.caption(
        'On desktop, play BBC World Service in another tab/device '
        'and record it here; on mobile, play the broadcast on another device or speaker. The recorded audio is '
        'sent directly to Groq Whisper, stored in the hosted transcript index, and becomes immediately queryable.'
    )
    browser_bbc = st.audio_input('Record BBC audio in your browser', key='bbc_browser_capture')
    if st.button('🎙️ Transcribe & Index Browser BBC Audio', type='primary',
                 disabled=browser_bbc is None, key='btn_bbc_browser_capture'):
        try:
            with st.spinner('Transcribing BBC audio with Groq Whisper…'):
                text = transcribe(browser_bbc.getvalue(), 'bbc-browser.wav')
                doc = add_transcript_to_db(text, source='BBC World Service (Browser Capture)')
            st.success('✅ BBC audio transcribed and indexed in the transcript index.')
            st.info(f"**ID:** `{doc['id']}` | **Words:** {doc['metadata']['word_count']} | **Timestamp:** {doc['metadata']['timestamp']}")
            st.write(doc['text'])
        except Exception as exc:
            st.error(safe_error(exc))

    with st.expander('🎙️ Other Audio Input Methods (Recording, Upload, Paste)'):
        rec_col, up_col = st.columns(2)

        with rec_col:
            st.markdown('**Browser Voice Recording**')
            recorded = st.audio_input('Record microphone audio', key='manual_rec')
            if st.button('Transcribe & Store Recording', disabled=recorded is None, key='btn_transcribe_rec'):
                try:
                    with st.spinner('Transcribing browser recording…'):
                        text = transcribe(recorded.getvalue(), 'browser-rec.wav')
                        doc = add_transcript_to_db(text, source='Browser Recording')
                    st.success('Recording saved to the transcript index.')
                    st.write(text)
                except Exception as exc:
                    st.error(safe_error(exc))

        with up_col:
            st.markdown('**Upload Audio File**')
            upload = st.file_uploader('Audio file (WAV, MP3, M4A, FLAC)', type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'webm'], key='manual_file')
            if st.button('Transcribe & Store Upload', disabled=upload is None, key='btn_transcribe_up'):
                try:
                    with st.spinner('Transcribing uploaded audio file…'):
                        text = transcribe(upload.getvalue(), upload.name)
                        doc = add_transcript_to_db(text, source='Uploaded Audio')
                    st.success('Uploaded transcript saved to the transcript index.')
                    st.write(text)
                except Exception as exc:
                    st.error(safe_error(exc))

        st.markdown('---')
        pasted = st.text_area('Paste Raw Transcript Text', max_chars=100000, key='manual_pasted')
        if st.button('Store Pasted Transcript', disabled=not pasted.strip(), key='btn_store_pasted'):
            doc = add_transcript_to_db(pasted, source='Pasted Transcript')
            st.success('Pasted transcript saved to the transcript index.')
            st.write(doc['text'])

# -----------------------------------------------------------------------------
# TAB 3: 📊 Database Tab
# -----------------------------------------------------------------------------
with db_tab:
    st.subheader('📊 Transcript Storage Status')

    st.caption('Database access is loaded on demand so normal page navigation stays lightweight.')

    if st.button('📊 Load / Refresh Database Status', key='btn_db_load'):
        try:
            st.session_state.news_db_stats = get_db_stats()
            st.session_state.news_recent_chunks = get_recent_chunks(limit=10)
        except Exception as exc:
            st.error(safe_error(exc))

    if 'news_db_stats' in st.session_state:
        stats = st.session_state.news_db_stats
        recent_chunks = st.session_state.get('news_recent_chunks', [])

        m1, m2 = st.columns(2)
        with m1:
            st.metric('Total Transcripts Indexed', stats['total_chunks'])
        with m2:
            st.metric('Total Words Indexed', stats['total_words'])

        st.divider()
        st.markdown('### 🕒 10 Most Recent Transcript Chunks')
        if not recent_chunks:
            st.info('No transcript chunks in the database yet. Capture or transcribe audio to index items.')
        else:
            for idx, chunk in enumerate(recent_chunks, 1):
                meta = chunk['metadata']
                with st.expander(f"#{idx} [{chunk['id']}] {meta.get('source', 'Radio')} · {meta.get('timestamp', '')} ({meta.get('word_count', 0)} words)"):
                    st.write(chunk['text'])
                    st.caption(f"**Chunk ID:** `{chunk['id']}` | **Unix Timestamp:** `{meta.get('unix_time', 0)}`")

        st.divider()
        st.markdown('### ⚙️ Database Actions')
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            all_chunks = get_recent_chunks(limit=1000)
            export_json = json.dumps(all_chunks, indent=2)
            st.download_button(
                '📥 Export Database Transcripts (JSON)',
                export_json,
                'chromadb_transcripts.json',
                'application/json',
                key='btn_db_export',
            )
        with btn_c2:
            if st.button('🗑️ Clear / Reset Transcript Database', type='secondary', key='btn_db_clear'):
                clear_db()
                st.session_state.pop('news_db_stats', None)
                st.session_state.pop('news_recent_chunks', None)
                st.warning('Persistent transcript database reset successfully.')
                st.rerun()
