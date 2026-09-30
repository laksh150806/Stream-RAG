import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import hosted_news as news
from streamlit.testing.v1 import AppTest

class HostedNewsTests(unittest.TestCase):
    def test_empty_or_expired_evidence_never_calls_groq(self):
        old = news.make_document('The workshop has 100 seats.', 'Test')
        old['metadata']['unix_time'] = 0
        with patch.object(news, 'client') as client:
            self.assertEqual(news.answer('seats', [old], 'test', 10)['sources'], [])
            client.assert_not_called()

    def test_generation_receives_only_retrieved_evidence(self):
        rows = [news.make_document('Pune workshop capacity is 100 seats.', 'Test'), news.make_document('Rain expected tomorrow.', 'Other')]
        with patch.object(news, 'client') as c:
            c.return_value.chat.completions.create.return_value = NS(choices=[NS(message=NS(content='100 seats.'))])
            result = news.answer('workshop capacity', rows, 'test')
            prompt = c.return_value.chat.completions.create.call_args.kwargs['messages'][1]['content']
            self.assertIn('100 seats', prompt)
            self.assertNotIn('Rain expected', prompt)
            self.assertTrue(result['sources'])

    def test_page_checks_auth_without_showing_key(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-not-real'}), patch.object(news, 'connection_check', return_value={'chat_model':'test-model'}):
            app = AppTest.from_file('../pages/2_Live_News.py').run()
            self.assertFalse(app.exception)
            app.button(key='news_check').click().run()
            self.assertFalse(app.exception)
            self.assertIn('Groq authentication passed.', [s.value for s in app.success])

    def test_api_errors_do_not_expose_response(self):
        exc = Exception('secret response body')
        exc.status_code = 401
        self.assertNotIn('secret response', news.safe_error(exc))

if __name__ == '__main__': unittest.main()
