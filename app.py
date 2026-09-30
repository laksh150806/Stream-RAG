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
.controller-card{border:1px solid rgba(148,163,184,.22);border-radius:14px;padding:14px 16px;margin:4px 0 14px;background:rgba(15,23,42,.28)}
.controller-label{font-size:11px;letter-spacing:1.6px;font-weight:700;color:#94a3b8;margin-bottom:6px}
.controller-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.controller-state{font-size:18px;font-weight:800;letter-spacing:.04em}
.controller-detail{font-size:13px;color:#cbd5e1}
.state-retrieve{color:#22c55e}.state-wait{color:#f59e0b}.state-suppress{color:#38bdf8}.state-reuse{color:#a78bfa}.state-idle{color:#94a3b8}
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
    st.caption('Answers are source excerpts. This mode uses local BM25 retrieval with concept normalization, metadata-aware constraints, and rule-based intent parsing; it does not call a generative LLM.')
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
    st.page_link('pages/2_Live_News.py', label='Live audio / news RAG', icon='🎙️')

engine = st.session_state.engine
st.markdown('<div class="eyebrow">LISTEN EARLY · KEEP CONTEXT · SHOW THE EVIDENCE</div>', unsafe_allow_html=True)
st.title('Answers that evolve with you.')
st.caption('Watch retrieval begin before a transcript ends, then refine the same answer as details change.')
session_tab, corpus_tab, trace_tab, eval_tab, about_tab = st.tabs(['Live session', 'Evidence library', 'Trace & export', 'Evaluation', 'How it works'])


def controller_status():
    """Return the latest controller decision for judge-visible observability."""
    for event in reversed(engine.events):
        kind = event.get('event')
        if kind == 'retrieval_suppressed':
            return 'SUPPRESS', 'Presentation-only request · no corpus search', 'suppress'
        if kind == 'retrieval_started':
            count = len(event.get('sub_queries', []))
            trigger = str(event.get('trigger', 'retrieval')).replace('_', ' ').title()
            return 'RETRIEVE', f'{trigger} · {count} search job' + ('s' if count != 1 else ''), 'retrieve'
        if kind == 'retrieval_wait':
            reason = str(event.get('reason', 'waiting for a stable intent')).replace('_', ' ')
            return 'WAIT', reason.capitalize(), 'wait'
        if kind == 'retrieval_reused':
            return 'REUSE', 'Existing evidence reused · no new search', 'reuse'
    return 'IDLE', 'Waiting for transcript input', 'idle'


def early_lead():
    """Seconds retrieval started before the final transcript for the active turn."""
    if engine.turn is None:
        return None
    rows = [e for e in engine.events if e.get('turn_id') == engine.turn]
    starts = [e['timestamp_s'] for e in rows if e.get('event') == 'retrieval_started' and e.get('trigger') == 'provisional']
    finals = [e['timestamp_s'] for e in rows if e.get('event') == 'transcript_received' and e.get('final')]
    if not starts:
        return None
    if not finals:
        return 'active'
    return max(0.0, min(finals) - min(starts))


def decision_timeline():
    if engine.turn is None:
        return []
    rows = []
    labels = {
        'transcript_received': 'TRANSCRIPT',
        'retrieval_wait': 'WAIT',
        'retrieval_started': 'RETRIEVE',
        'retrieval_completed': 'EVIDENCE',
        'refinement_planned': 'REFINE',
        'retrieval_suppressed': 'SUPPRESS',
        'retrieval_reused': 'REUSE',
        'answer_updated': 'ANSWER',
    }
    for event in engine.events:
        kind = event.get('event')
        if event.get('turn_id') != engine.turn or kind not in labels:
            continue
        if kind == 'transcript_received':
            detail = ('final · ' if event.get('final') else 'partial · ') + str(event.get('text', ''))[:90]
        elif kind == 'retrieval_started':
            detail = f"{event.get('trigger', '')} · {len(event.get('sub_queries', []))} job(s)"
        elif kind == 'retrieval_completed':
            detail = ', '.join(event.get('source_ids', [])) or 'no evidence'
        elif kind == 'refinement_planned':
            detail = 'changed: ' + ', '.join(event.get('changed_constraints', []))
        else:
            detail = str(event.get('reason', '')).replace('_', ' ')
        rows.append({'t (s)': event.get('timestamp_s'), 'decision': labels[kind], 'detail': detail})
    return rows[-12:]


def draw_answer():
    snapshot = engine.snapshot()
    state, detail, css_state = controller_status()
    st.markdown(
        f'<div class="controller-card"><div class="controller-label">RETRIEVAL CONTROLLER</div>'
        f'<div class="controller-row"><span class="controller-state state-{css_state}">{state}</span>'
        f'<span class="controller-detail">{detail}</span></div></div>',
        unsafe_allow_html=True,
    )
    lead = early_lead()
    a, b, c1, d = st.columns(4)
    a.metric('Answer version', snapshot['answer_version'])
    b.metric('Searches executed', snapshot['retrieval_count'])
    c1.metric('Active intents', len(snapshot['claims']))
    d.metric('Early retrieval lead', '—' if lead is None else ('before final' if lead == 'active' else f'{lead:.1f} s'))
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

    timeline = decision_timeline()
    if timeline:
        st.markdown('**Live decision timeline**')
        st.dataframe(timeline, width='stretch', hide_index=True)

    st.divider()
    st.markdown('**Try this sequence**')
    st.code('Workshop capacity in Pune and cancellation policy and catering options and accessibility\nActually city: Delhi\nPlease repeat your last answer in two bullets.', language=None)

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

with eval_tab:
    st.subheader('Development evaluation')
    report = json.loads((ROOT / 'docs/development-results.json').read_text())
    streaming = report['modes']['streaming']
    baseline = report['modes']['end_of_turn']
    no_decomp = report['modes']['no_decomposition']
    no_refine = report['modes']['no_refinement']

    a, b, c, d = st.columns(4)
    a.metric('Streaming source checks', f"{streaming['passed']}/{streaming['total']}")
    b.metric('End-of-turn source checks', f"{baseline['passed']}/{baseline['total']}")
    c.metric('No decomposition', f"{no_decomp['passed']}/{no_decomp['total']}")
    d.metric('No refinement', f"{no_refine['passed']}/{no_refine['total']}")

    stream_early = sum(r['provisional_search_batches'] > 0 for r in streaming['results'])
    baseline_early = sum(r['provisional_search_batches'] > 0 for r in baseline['results'])
    st.success(f'Early retrieval: streaming {stream_early}/{streaming["total"]} cases · end-of-turn baseline {baseline_early}/{baseline["total"]}.')
    st.caption(report['scope'])
    st.dataframe(
        [{
            'case': r['case'],
            'pass': r['pass'],
            'searches': r['searches'],
            'provisional batches': r['provisional_search_batches'],
        } for r in streaming['results']],
        width='stretch',
        hide_index=True,
    )
    st.caption('These are authored synthetic development checks, not official or held-out benchmark results. CI reruns unit/UI tests and requires all streaming development checks to pass.')

with about_tab:
    st.markdown('''### What this version does
1. Accepts cumulative, timestamped transcript fragments and decides to wait, retrieve, or reuse evidence.
2. Splits compound requests into parallel retrieval jobs.
3. Applies named constraints such as `city: Delhi` to relevant claims while retaining unaffected evidence.
4. Returns exact source excerpts, validates citation IDs, and flags missing evidence.
5. Keeps conversation state in this browser session and exports a complete event trace.

### Current limits
The controller and decomposition are English rules. Retrieval combines BM25 with concept normalization, metadata-value inference, and confidence filtering. This improves common paraphrases while keeping the core deterministic and inspectable; complex negation and truly unfamiliar language can still fail. Excerpts prove source provenance, not that every retrieved passage answers the question. This is an engineering prototype; official hackathon gates have not been measured.

The hosted live-audio page uses browser recording or uploaded audio, Groq Whisper transcription, BM25 retrieval, Groq answer generation, and session-only transcripts. Direct BBC server capture is retained only as an experimental fallback because some cloud regions block BBC media CDNs.
''')
