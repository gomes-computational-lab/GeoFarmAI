from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, TextIO


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def run_log_context(out_dir: str, prefix: str) -> Iterator[Path]:
    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{prefix}_{timestamp}.log"

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with log_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, log_file)  # type: ignore[assignment]
        try:
            print(f"[log] Recording run output to {log_path}")
            yield log_path
        finally:
            print(f"[log] Finished recording run output to {log_path}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr
