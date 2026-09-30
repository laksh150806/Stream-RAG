import ast
import json
import tempfile
import threading
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from news_capture import CaptureWorker

ROOT = Path(__file__).resolve().parents[1]


def original_functions():
    source = ROOT / 'Live-Streaming-Data-RAG/streaming_rag_app_.py'
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name not in {'get_secret_key','init_components'}]
    ns = dict(datetime=datetime, timezone=timezone, timedelta=timedelta, Path=Path, json=json)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
    return ns


class NewsTests(unittest.TestCase):
    def test_last_minutes_window_includes_now(self):
        ns = original_functions()
        for minutes in [10,30,60]:
            before = int(datetime.now(timezone.utc).timestamp())
            result = ns['build_time_filter'](minutes,30,0)['$and']
            upper = result[1]['unix_start']['$lte']
            lower = result[2]['unix_end']['$gte']
            self.assertLessEqual(abs(upper-before),1)
            self.assertEqual(upper-lower,minutes*60)

    def test_transcript_reaches_index_and_reports_actual_asr(self):
        ns = original_functions()
        indexed = []
        with tempfile.TemporaryDirectory() as root:
            ns.update(BASE_DIR=Path(root),TRANSCRIPT_DIR=Path(root),METADATA_DIR=Path(root),CHUNK_DURATION=60,CHANNEL_ID=0,index_transcript_langchain=indexed.append)
            rag = ns['StreamingDataRAG']()
            result = rag.save_and_index(Path(root)/'chunk_uuid_20260930T120000Z.wav','Synthetic transcript.')
            self.assertEqual(len(indexed),1)
            self.assertEqual(result['asr_model'],'whisper-large-v3-turbo')
            self.assertEqual(result['chunk_id'],'chunk_uuid_20260930T120000Z')

    def test_indexing_failure_is_not_silenced(self):
        ns = original_functions()
        def fail(_):
            raise RuntimeError('Index unavailable')
        with tempfile.TemporaryDirectory() as root:
            ns.update(BASE_DIR=Path(root),TRANSCRIPT_DIR=Path(root),METADATA_DIR=Path(root),CHUNK_DURATION=60,CHANNEL_ID=0,index_transcript_langchain=fail)
            with self.assertRaisesRegex(RuntimeError,'Index unavailable'):
                ns['StreamingDataRAG']().save_and_index(Path(root)/'chunk_id_20260930T120000Z.wav','Test')

    def test_worker_does_not_need_streamlit_context(self):
        reached = threading.Event()
        def capture():
            reached.set()
            return {'chunk_id':'test','word_count':3}
        worker = CaptureWorker(capture,interval=30)
        worker.start()
        self.assertTrue(reached.wait(2))
        worker.stop()
        worker.thread.join(timeout=2)
        self.assertFalse(worker.thread.is_alive())
        self.assertIn('Captured test',worker.drain()[0])

    def test_worker_reports_failures(self):
        reached = threading.Event()
        def capture():
            reached.set()
            raise RuntimeError('audio unavailable')
        worker = CaptureWorker(capture,interval=30)
        worker.start()
        self.assertTrue(reached.wait(2))
        worker.stop()
        worker.thread.join(timeout=2)
        self.assertIn('audio unavailable',worker.drain()[0])


if __name__ == '__main__':
    unittest.main()
