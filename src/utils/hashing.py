from __future__ import annotations

import hashlib
import json


def policy_fingerprint(payload):
    """Stable identity for execution weights; resaving a zip cannot reopen a test set."""
    digest=hashlib.sha256()
    for key in ('environment_version','observation_version','num_agents','obs_dim','action_dim'):
        digest.update(str(payload[key]).encode())
    # Equivalent GRUCell -> one-layer GRU migration must not reopen sealed tests.
    recurrent={'gru.weight_ih_l0','gru.weight_hh_l0','gru.bias_ih_l0','gru.bias_hh_l0'}
    actor={key.removesuffix('_l0') if key in recurrent else key:value for key,value in payload['actor'].items()}
    for key,tensor in sorted(actor.items()):
        array=tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode());digest.update(str(array.shape).encode());digest.update(str(array.dtype).encode());digest.update(array.tobytes())
    return digest.hexdigest()
from typing import Any


def sha256_digest(payload: Any) -> str:
    """Compute a deterministic SHA-256 digest for JSON-serializable objects."""
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
