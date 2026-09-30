import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import hosted_news as news
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, _):
        return self.body


class HostedNewsTests(unittest.TestCase):
    def test_empty_or_expired_evidence_never_calls_groq(self):
        old = news.make_document('The workshop has 100 seats.', 'Test')
        old['metadata']['unix_time'] = 0
        with patch.object(news, 'client') as client:
            self.assertEqual(news.answer('seats', [old], 'test', 10)['sources'], [])
            client.assert_not_called()

    def test_generation_receives_only_retrieved_evidence(self):
        rows = [
            news.make_document('Pune workshop capacity is 100 seats.', 'Test'),
            news.make_document('Rain expected tomorrow.', 'Other'),
        ]
        with patch.object(news, 'client') as c:
            c.return_value.chat.completions.create.return_value = NS(
                choices=[NS(message=NS(content='100 seats.'))]
            )
            result = news.answer('workshop capacity', rows, 'test')
            prompt = c.return_value.chat.completions.create.call_args.kwargs['messages'][1]['content']
            self.assertIn('100 seats', prompt)
            self.assertNotIn('Rain expected', prompt)
            self.assertTrue(result['sources'])

    def test_playlist_parser_extracts_current_streams(self):
        body = (
            b'[playlist]\n'
            b'File1=http://example.test/live.mp3\n'
            b'File2=https://example.test/live2.mp3\n'
        )
        with patch('urllib.request.urlopen', return_value=_Response(body)):
            self.assertEqual(
                news._playlist_streams('http://example.test/list.pls'),
                ['http://example.test/live.mp3', 'https://example.test/live2.mp3'],
            )

    def test_candidate_streams_prefer_override_and_deduplicate(self):
        with patch.dict(os.environ, {'BBC_STREAM_URL': 'http://override.test/live'}), \
                patch.object(
                    news,
                    '_playlist_streams',
                    return_value=['http://override.test/live', 'http://dynamic.test/live'],
                ):
            streams = news._candidate_streams()
        self.assertEqual(streams[0], 'http://override.test/live')
        self.assertEqual(streams.count('http://override.test/live'), 1)
        self.assertIn('http://dynamic.test/live', streams)

    def test_capture_falls_back_after_ffmpeg_failure(self):
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            self.skipTest('imageio-ffmpeg not installed')

        calls = []

        def fake_run(cmd, **_):
            calls.append(cmd)
            if len(calls) == 1:
                raise subprocess.CalledProcessError(
                    1,
                    cmd,
                    stderr=b'Server returned 403 Forbidden',
                )
            Path(cmd[-1]).write_bytes(b'RIFF' + b'x' * 5000)
            return NS(stderr=b'')

        with patch.object(news, '_candidate_streams', return_value=['http://one', 'http://two']), \
                patch.object(news.subprocess, 'run', side_effect=fake_run):
            data = news.capture_audio(5)

        self.assertGreater(len(data), 4096)
        self.assertEqual(len(calls), 2)

    def test_page_checks_auth_without_showing_key(self):
        with patch.dict(os.environ, {'GROQ_API_KEY': 'test-not-real'}), \
                patch.object(
                    news,
                    'connection_check',
                    return_value={
                        'chat_model': 'test-model',
                        'speech_model': 'whisper-large-v3-turbo',
                    },
                ):
            app = AppTest.from_file(str(ROOT / 'pages/2_Live_News.py')).run()
            self.assertFalse(app.exception)
            app.button(key='news_check').click().run()
            self.assertFalse(app.exception)
            self.assertIn('Groq authentication passed.', [s.value for s in app.success])

    def test_api_errors_do_not_expose_response(self):
        exc = Exception('secret response body')
        exc.status_code = 401
        self.assertNotIn('secret response', news.safe_error(exc))

    def test_runtime_error_message_is_preserved(self):
        self.assertEqual(
            news.safe_error(RuntimeError('BBC capture unavailable')),
            'BBC capture unavailable',
        )


if __name__ == '__main__':
    unittest.main()
