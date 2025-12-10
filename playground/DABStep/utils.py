import threading
import json
from pathlib import Path


append_answer_lock = threading.Lock()

def append_answer(entry: dict, jsonl_file: Path) -> None:
    jsonl_file.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")