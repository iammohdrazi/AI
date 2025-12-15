import sys
import threading
import itertools
import time


class Spinner:
    def __init__(self, message="Searching"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        spinner = itertools.cycle(["|", "/", "-", "\\"])
        while not self._stop_event.is_set():
            sys.stdout.write(f"\r{self.message}... {next(spinner)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
