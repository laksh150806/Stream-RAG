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


if __name__=='__main__':
    unittest.main()
