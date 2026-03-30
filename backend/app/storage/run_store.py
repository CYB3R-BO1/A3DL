from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunStore:
    def __init__(self, root: str = "../artifacts/reports") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, payload: dict[str, Any]) -> str:
        out = self.root / f"{run_id}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(out)

    def load(self, run_id: str) -> dict[str, Any]:
        path = self.root / f"{run_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))
