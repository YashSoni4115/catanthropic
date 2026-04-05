from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _predict_probabilities(model: Any, vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    X = np.asarray(vectors, dtype=np.float32)
    proba = model.predict_proba(X)
    if proba.ndim != 2 or proba.shape[1] < 2:
        raise ValueError("model predict_proba must return [n,2]")
    return [float(v) for v in proba[:, 1]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Streaming inference worker for win model v1")
    parser.add_argument("--model", required=True, help="Path to pickled sklearn model")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            vectors = payload.get("vectors") if isinstance(payload, dict) else None
            if not isinstance(vectors, list):
                raise ValueError("payload must include list field 'vectors'")
            probabilities = _predict_probabilities(model, vectors)
            response: Dict[str, Any] = {"probs": probabilities}
        except Exception as exc:
            response = {"probs": [], "error": str(exc)}

        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
