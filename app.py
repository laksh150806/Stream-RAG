"""Theme 4 workspace. Run: streamlit run app.py."""
import json
import time
from pathlib import Path

import streamlit as st

from rag_core import StreamingSession, load_corpus, simulate_stream

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Stream-RAG · Live evidence', page_icon='◉', layout='wide')
st.markdown('''<style>
.block-container{max-width:1280px;padding-top:2rem} h1{letter-spacing:-.055em}
[data-testid="stMetric"]{background:rgba(34,211,238,.06);border:1px solid rgba(34,211,238,.18);border-radius:12px;padding:14px}
.eyebrow{color:#22d3ee;font-size:12px;letter-spacing:2px;font-weight:700}
</style>''', unsafe_allow_html=True)


def fresh_session(payload=None):
    payload = payload or json.loads((ROOT / 'data/demo_corpus.json').read_text())
    st.session_state.corpus_payload = payload
    st.session_state.engine = StreamingSession(load_corpus(payload))
    st.session_state.turn_number = 0
    st.session_state.last_transcript = ''


if 'engine' not in st.session_state:
    fresh_session()

with st.sidebar:
    st.title('◉ Stream-RAG')
    st.caption('THEME 4 WORKSPACE')
    st.success('Corpus-only mode · No API key needed')
    st.caption('Answers are source excerpts. This mode uses local BM25 retrieval and rule-based intent parsing; it does not call a generative LLM.')
    upload = st.file_uploader('Replace evidence corpus', type=['json'])
    if st.button('Load uploaded corpus', disabled=upload is None):
        try:
            if upload.size > 2 * 1024 * 1024:
                raise ValueError('Upload limit is 2 MB.')
            payload = json.loads(upload.getvalue())
            load_corpus(payload)
            fresh_session(payload)
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    if st.button('Reset conversation', width='stretch'):
        fresh_session(st.session_state.corpus_payload)
        st.rerun()
    st.divider()
    st.caption('The bundled corpus is synthetic demo data, not real company or travel policy. Uploaded evidence stays in this session.')
    st.page_link('pages/2_Live_News.py', label='BBC live-news mode', icon='📻')

engine = st.session_state.engine
st.markdown('<div class="eyebrow">LISTEN EARLY · KEEP CONTEXT · SHOW THE EVIDENCE</div>', unsafe_allow_html=True)
st.title('Answers that evolve with you.')
st.caption('Watch retrieval begin before a transcript ends, then refine the same answer as details change.')
session_tab, corpus_tab, trace_tab, about_tab = st.tabs(['Live session', 'Evidence library', 'Trace & export', 'How it works'])


def draw_answer():
    snapshot = engine.snapshot()
    a, b, c = st.columns(3)
    a.metric('Answer version', snapshot['answer_version'])
    b.metric('Searches executed', snapshot['retrieval_count'])
    c.metric('Active intents', len(snapshot['claims']))
    if snapshot['constraints']:
        st.caption('Current scope: ' + ' · '.join(f'{k}: {v}' for k, v in snapshot['constraints'].items()))
    if not snapshot['claims']:
        st.info('Send a question or a transcript fragment to begin.')
        return
    if snapshot['style'] == 'two_bullets':
        # Group existing evidence without inventing a new summary or dropping citations.
        groups = [[], []]
        for i, item in enumerate(snapshot['claims']):
            pieces = [f"{e['quote']} [{e['source_id']}]" for e in item['evidence']]
            groups[i % 2].append(' '.join(pieces) or item['uncertainty'])
        for group in groups:
            if group:
                st.markdown('- ' + ' '.join(group))
    else:
        for item in snapshot['claims']:
            with st.container(border=True):
                st.markdown('**' + item['query'].replace('*', '') + '**')
                if item['uncertainty']:
                    st.warning(item['uncertainty'])
                for evidence in item['evidence']:
                    st.write(evidence['quote'])
                    st.caption(f"[{evidence['source_id']}] · {evidence['title']}")


with session_tab:
    message = st.text_input('Question or follow-up', placeholder='Workshop capacity in Pune and cancellation policy and catering options', key='message')
    stream = st.button('▶ Stream message', type='primary')
    st.caption('Demo input is replayed in small cumulative text chunks, simulating incoming speech. It does not record your microphone.')
    with st.expander('Send actual transcript fragments / replay JSONL'):
        fragment = st.text_input('Cumulative transcript for this turn', key='fragment')
        final_fragment = st.checkbox('End of utterance', value=False)
        manual = st.button('Send fragment')
        replay_file = st.file_uploader('Timestamped events (.jsonl)', type=['jsonl'])
        run_replay = st.button('Run uploaded replay', disabled=replay_file is None)
    output = st.empty()
    if stream and message.strip():
        st.session_state.turn_number += 1
        start = max(0, engine.last_timestamp + .4)
        for event in simulate_stream(message, st.session_state.turn_number, start=start):
            engine.ingest(**event)
            st.session_state.last_transcript = event['text']
            with output.container():
                st.caption(('Final transcript: ' if event['final'] else 'Listening: ') + event['text'])
                draw_answer()
            time.sleep(.15)
    elif manual and fragment.strip():
        if engine.turn is None or engine.turn in engine._closed_turns:
            st.session_state.turn_number += 1
        engine.ingest(fragment, timestamp_s=max(0, engine.last_timestamp + .4), final=final_fragment, turn_id=str(st.session_state.turn_number))
        with output.container():
            draw_answer()
    elif run_replay:
        try:
            if replay_file.size > 2 * 1024 * 1024:
                raise ValueError('Replay limit is 2 MB.')
            events = [json.loads(line) for line in replay_file.getvalue().decode().splitlines() if line.strip()]
            if len(events) > 1000:
                raise ValueError('Replay limit is 1000 events.')
            candidate = StreamingSession(load_corpus(st.session_state.corpus_payload))
            for event in events:
                candidate.ingest(**event)
            st.session_state.engine = candidate
            st.session_state.turn_number = len(candidate._closed_turns) + 1
            st.rerun()
        except (ValueError, TypeError, UnicodeError) as exc:
            st.error(f'Replay rejected: {exc}')
    else:
        with output.container():
            draw_answer()
    st.divider()
    st.markdown('**Try this sequence**')
    st.code('Workshop capacity in Pune and cancellation policy and catering options\nActually city: Delhi\nPlease repeat your last answer in two bullets.', language=None)

with corpus_tab:
    st.subheader('Every answer starts here')
    st.caption(f'{len(engine.retriever.chunks)} source chunks. References point to exact excerpts shown below.')
    st.download_button('Download corpus', json.dumps(st.session_state.corpus_payload, indent=2), 'corpus.json', 'application/json')
    for chunk in engine.retriever.chunks:
        with st.expander(f'{chunk.id} · {chunk.title}'):
            st.write(chunk.text)
            st.json(chunk.metadata)

with trace_tab:
    st.subheader('A trace for every input event')
    st.download_button('Export session + answer versions', json.dumps(engine.export(), indent=2), 'stream-rag-session.json', 'application/json')
    st.download_button('Download sample input replay', (ROOT / 'data/demo_stream.jsonl').read_text(), 'demo_stream.jsonl', 'application/x-ndjson')
    st.dataframe(engine.events, width='stretch')
    with st.expander('Answer version history'):
        st.json(engine.versions)

with about_tab:
    st.markdown('''### What this version does
1. Accepts cumulative, timestamped transcript fragments and decides to wait, retrieve, or reuse evidence.
2. Splits compound requests into parallel retrieval jobs.
3. Applies named constraints such as `city: Delhi` to relevant claims while retaining unaffected evidence.
4. Returns exact source excerpts, validates citation IDs, and flags missing evidence.
5. Keeps conversation state in this browser session and exports a complete event trace.

### Current limits
The controller and decomposition are English rules, and retrieval is lexical BM25 with a small synonym map. Implicit constraints, complex negation and unfamiliar paraphrases may fail. Excerpts prove source provenance, not that every retrieved passage answers the question. This is an engineering prototype; official hackathon gates have not been measured.

The hosted BBC page uses Groq speech recognition and generation with BM25 retrieval and session-only transcripts. Its displayed model is selected from the models available to the configured account. The original MiniLM/Chroma experiment is retained in the repository.
''')
