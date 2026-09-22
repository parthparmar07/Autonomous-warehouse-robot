from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_digest(payload: Any) -> str:
    """Compute a deterministic SHA-256 digest for JSON-serializable objects."""
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
