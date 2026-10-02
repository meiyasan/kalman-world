"""Every test session gets its own registry: the shared store (kalman.api.store)
is a file, and tests must neither read a developer's ~/.kalman/state nor leave
models in it."""
import os
import tempfile

_STATE = tempfile.mkdtemp(prefix="kalman-test-state-")
os.environ["KALMAN_STATE_DIR"] = _STATE
