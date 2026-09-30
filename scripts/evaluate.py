"""Small authored development suite; never an official/held-out score."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_core import StreamingSession, load_corpus

ROOT = Path(__file__).resolve().parents[1]
cases = [
    ('compound', ['Workshop capacity in Pune and cancellation policy and catering options'], ['pune-capacity:s1','pune-cancellation:s1','pune-catering:s1']),
    ('late_city', ['Workshop capacity in Pune and accessibility policy','Actually city: Delhi'], ['delhi-capacity:s1','workshop-accessibility:s1']),
    ('format_only', ['Workshop capacity in Pune','Please repeat your last answer in two bullets.'], ['pune-capacity:s1']),
    ('unknown_city', ['Workshop capacity in Pune','Actually city: Atlantis'], []),
    ('unsupported_question', ['Quantum banana teleportation'], []),
    ('lexical_alias', ['Food options in Bengaluru'], ['bengaluru-catering:s1']),
    ('paraphrase_capacity', ['Could our thirty delegates fit in Cedar Hall?'], ['pune-capacity:s1']),
    ('paraphrase_cancellation', ['What would we forfeit if we called off the gathering in Delhi?'], ['delhi-cancellation:s1']),
    ('paraphrase_travel', ['What paperwork is needed to claim an overseas journey?'], ['international-travel:s1']),
]
report = {'scope':'Nine self-authored development cases against a synthetic corpus. No Groq calls or official benchmark. Exact source-set matching is strict and small-sample.', 'modes':{}}
for name, options in [('streaming',{}),('end_of_turn',{'early':False}),('no_decomposition',{'decompose':False}),('no_refinement',{'refine':False})]:
    results = []
    for case_id, turns, expected in cases:
        s = StreamingSession(load_corpus((ROOT/'data/demo_corpus.json').read_text()), **options)
        for i, text in enumerate(turns):
            # One nonfinal complete-text fragment isolates early vs end-of-turn behavior.
            s.ingest(text,timestamp_s=i*2,turn_id=str(i),final=False)
            s.ingest(text,timestamp_s=i*2+1,turn_id=str(i),final=True)
        actual = sorted({e['source_id'] for c in s.snapshot()['claims'] for e in c['evidence']})
        results.append({'case':case_id,'pass':actual==sorted(expected),'expected':sorted(expected),'actual':actual,'searches':s.snapshot()['retrieval_count'],'provisional_search_batches':sum(e['event']=='retrieval_started' and e['trigger']=='provisional' for e in s.events)})
    report['modes'][name]={'passed':sum(r['pass'] for r in results),'total':len(results),'results':results}
print(json.dumps(report,indent=2))
