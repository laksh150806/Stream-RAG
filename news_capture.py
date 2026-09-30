"""Background capture owns no Streamlit objects; the UI drains its queue."""
import queue
import threading
import time


class CaptureWorker:
    def __init__(self, capture_once, interval=5, idle_timeout=180):
        self.capture_once = capture_once
        self.interval = interval
        self.idle_timeout = idle_timeout
        self.messages = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        self.last_heartbeat = time.monotonic()
        self.thread = threading.Thread(target=self._run, daemon=True, name='bbc-capture')

    @property
    def running(self):
        return self.thread.is_alive() and not self.stop_event.is_set()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def drain(self):
        self.last_heartbeat = time.monotonic()
        messages = []
        while True:
            try:
                messages.append(self.messages.get_nowait())
            except queue.Empty:
                return messages

    def _run(self):
        while not self.stop_event.is_set() and time.monotonic() - self.last_heartbeat < self.idle_timeout:
            try:
                metadata = self.capture_once()
                message = f"Captured {metadata['chunk_id']}: {metadata['word_count']} words"
            except Exception as exc:
                message = f'Capture failed: {type(exc).__name__}: {str(exc)[:160]}'
            if self.messages.full():
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    pass
            self.messages.put_nowait(message)
            if self.stop_event.wait(self.interval):
                break
