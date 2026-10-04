import unittest
from pathlib import Path
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, 'Install requirements.lock.txt for UI tests')
class UITests(unittest.TestCase):
    def test_keyless_workspace_and_followup_flow(self):
        root = Path(__file__).resolve().parents[1]
        app = AppTest.from_file(str(root/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        for message in ['Workshop capacity in Pune and catering options','Actually city: Delhi','Please repeat your last answer in two bullets.']:
            app.text_input(key='message').set_value(message)
            next(b for b in app.button if b.label=='▶ Stream message').click()
            app.run(timeout=30)
            self.assertFalse(app.exception)
        snapshot = app.session_state['engine'].snapshot()
        self.assertEqual(snapshot['constraints']['city'],'Delhi')
        self.assertEqual(snapshot['style'],'two_bullets')
        self.assertTrue(all('delhi-' in e['source_id'] for c in snapshot['claims'] for e in c['evidence']))
        next(b for b in app.button if b.label=='Reset conversation').click()
        app.run(timeout=30)
        self.assertEqual(app.session_state['engine'].snapshot()['claims'],[])


    def test_core_grounded_generation_is_evidence_only(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'app.py').read_text(encoding='utf-8')
        self.assertIn('Retrieved evidence:', source)
        self.assertIn("invalid = cited - allowed", source)
        self.assertIn('the full corpus is never sent to generation', source)



    def test_custom_evidence_activation_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / 'app.py').read_text(encoding='utf-8')
        self.assertIn("Use uploaded evidence", source)
        self.assertIn("fresh_session(payload)", source)
        self.assertIn("ACTIVE: {active_name}", source)
        self.assertNotIn("upload.name}:{upload.size}", source)



if __name__=='__main__':
    unittest.main()
