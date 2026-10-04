import json
import unittest
from pathlib import Path
from rag_core import StreamingSession, load_corpus

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.chunks = load_corpus((ROOT / 'data/demo_corpus.json').read_text())
        self.s = StreamingSession(self.chunks)

    def send(self, text, turn='1', timestamp=0, final=True):
        return self.s.ingest(text, timestamp_s=timestamp, turn_id=turn, final=final)

    def ids(self):
        return {e['source_id'] for c in self.s.snapshot()['claims'] for e in c['evidence']}

    def test_early_retrieval_waits_for_incomplete_clause(self):
        self.send('Workshop capacity in', final=False)
        self.assertEqual(self.s.snapshot()['retrieval_count'], 0)
        self.send('Workshop capacity in Pune', timestamp=.8, final=False)
        self.assertEqual(self.ids(), {'pune-capacity:s1'})
        self.assertEqual(self.s.events[-3]['event'], 'retrieval_completed')

    def test_multi_intent_and_no_duplicate_final_search(self):
        text = 'Workshop capacity in Pune and cancellation policy and catering options'
        self.send(text, final=False)
        self.assertEqual(len(self.s.intents), 3)
        before = self.s.snapshot()['retrieval_count']
        self.send(text, timestamp=1, final=True)
        self.assertEqual(self.s.snapshot()['retrieval_count'], before)
        self.assertEqual(self.ids(), {'pune-capacity:s1', 'pune-cancellation:s1', 'pune-catering:s1'})

    def test_multi_sentence_preserves_independent_scopes(self):
        text = (
            'What is workshop capacity in Pune? '
            'What is the cancellation policy in Delhi? '
            'What are catering options in Bengaluru?'
        )
        result = self.send(text)
        self.assertEqual(
            self.ids(),
            {'pune-capacity:s1', 'delhi-cancellation:s1', 'bengaluru-catering:s1'},
        )
        scopes = [claim['scope'] for claim in result['claims']]
        self.assertEqual(scopes[0].get('city'), 'Pune')
        self.assertEqual(scopes[1].get('city'), 'Delhi')
        self.assertEqual(scopes[2].get('city'), 'Bengaluru')
        self.assertNotIn('city', result['constraints'])

    def test_late_scope_changes_only_dependent_claims(self):
        self.send('Workshop capacity in Pune and accessibility policy')
        unaffected = self.s.snapshot()['claims'][1]
        before = self.s.snapshot()['retrieval_count']
        result = self.send('Actually city: Delhi', turn='2', timestamp=1)
        self.assertEqual(result['retrieval_count'] - before, 1)
        self.assertEqual(result['claims'][1], unaffected)
        self.assertEqual(self.ids(), {'delhi-capacity:s1', 'workshop-accessibility:s1'})

    def test_format_only_keeps_evidence_and_suppresses_search(self):
        self.send('Workshop capacity in Pune and catering options')
        before = self.s.snapshot()
        self.send('Please repeat your last', turn='2', timestamp=1, final=False)
        after = self.send('Please repeat your last answer in two bullets.', turn='2', timestamp=2)
        self.assertEqual(after['retrieval_count'], before['retrieval_count'])
        self.assertEqual(after['claims'], before['claims'])
        self.assertEqual(after['style'], 'two_bullets')

    def test_format_request_with_new_fact_is_not_suppressed(self):
        self.send('Workshop capacity in Pune')
        before = self.s.snapshot()['retrieval_count']
        self.send('Explain catering options in two bullets', turn='2', timestamp=1)
        self.assertGreater(self.s.snapshot()['retrieval_count'], before)

    def test_unknown_scope_does_not_keep_old_city_evidence(self):
        self.send('Workshop capacity in Pune')
        self.send('Actually city: Atlantis', turn='2', timestamp=1)
        self.assertEqual(self.ids(), set())
        self.assertTrue(self.s.snapshot()['claims'][0]['uncertainty'])

    def test_unknown_question_returns_uncertainty(self):
        result = self.send('Quantum banana teleportation')
        self.assertFalse(self.ids())
        self.assertTrue(result['claims'][0]['uncertainty'])

    def test_paraphrase_concepts_and_inferred_scope(self):
        cases = [
            ('Could our thirty delegates fit in Cedar Hall?', {'pune-capacity:s1'}),
            ('What would we forfeit if we called off the gathering in Delhi?', {'delhi-cancellation:s1'}),
            ('What paperwork is needed to claim an overseas journey?', {'international-travel:s1'}),
            ('Do you have wheelchair access?', {'workshop-accessibility:s1'}),
        ]
        for query, expected in cases:
            session = StreamingSession(self.chunks)
            session.ingest(query, timestamp_s=0, turn_id='1', final=True)
            actual = {e['source_id'] for c in session.snapshot()['claims'] for e in c['evidence']}
            self.assertEqual(actual, expected, query)

    def test_revised_partial_removes_retracted_intent(self):
        self.send('Workshop capacity in Pune and catering options', final=False)
        self.send('Workshop capacity in Pune', timestamp=1)
        self.assertEqual(len(self.s.intents), 1)
        self.assertEqual(self.ids(), {'pune-capacity:s1'})

    def test_evidence_and_version_history_are_isolated(self):
        snap = self.send('Workshop capacity in Pune')
        snap['claims'][0]['evidence'][0]['quote'] = 'tampered'
        self.assertNotIn('tampered', str(self.s.snapshot()))
        other = StreamingSession(self.chunks)
        self.assertEqual(other.snapshot()['claims'], [])

    def test_citation_guard_rejects_tampering(self):
        self.send('Workshop capacity in Pune')
        self.s.intents['1:0']['claims'][0]['quote'] = 'unverified claim'
        with self.assertRaises(RuntimeError):
            self.s.snapshot()

    def test_end_of_turn_baseline(self):
        self.s = StreamingSession(self.chunks, early=False)
        self.send('Workshop capacity in Pune', final=False)
        self.assertEqual(self.s.snapshot()['retrieval_count'], 0)
        self.send('Workshop capacity in Pune', timestamp=1)
        self.assertEqual(self.ids(), {'pune-capacity:s1'})

    def test_invalid_timestamps_and_closed_turns(self):
        self.send('Workshop capacity in Pune', timestamp=2)
        for value in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.send('Catering options', turn='2', timestamp=value)
        with self.assertRaises(ValueError):
            self.send('Catering options', timestamp=3)

    def test_trace_has_one_completion_for_each_input(self):
        events = [json.loads(line) for line in (ROOT / 'data/demo_stream.jsonl').read_text().splitlines()]
        for event in events:
            self.s.ingest(**event)
        self.assertEqual(sum(e['event'] == 'event_completed' for e in self.s.events), len(events))
        self.assertTrue(self.s.versions)

    def test_upload_schema_and_exact_source_chunks(self):
        with self.assertRaises(ValueError):
            load_corpus([{'id':'same', 'text':'one'}, {'id':'same', 'text':'two'}])
        with self.assertRaises(ValueError):
            load_corpus([{'id':'a', 'text':'one', 'metadata':{'bad':[]}}])
        text = 'First paragraph.\n\n' + ('The second paragraph contains words. ' * 50)
        chunks = load_corpus([{'id':'doc', 'text':text}])
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(c.text in text for c in chunks))


    def test_unrelated_custom_corpus_is_not_sample_hardcoded(self):
        payload = {
            'documents': [
                {'id': 'robot-payload', 'title': 'XR-7 payload', 'text': 'XR-7 inspection drone supports a maximum payload of 2.4 kilograms.', 'metadata': {'topic': 'payload'}},
                {'id': 'robot-maintenance', 'title': 'XR-7 maintenance', 'text': 'Preventive maintenance is required every 120 flight hours.', 'metadata': {'topic': 'maintenance'}},
            ]
        }
        session = StreamingSession(load_corpus(payload))
        session.ingest('What is the XR-7 payload?', timestamp_s=0.0, final=True, turn_id='1')
        snapshot = session.snapshot()
        source_ids = [e['source_id'] for item in snapshot['claims'] for e in item['evidence']]
        self.assertIn('robot-payload:s1', source_ids)
        self.assertTrue(all('pune' not in source_id.lower() for source_id in source_ids))


if __name__ == '__main__':
    unittest.main()
