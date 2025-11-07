# io/__init__.py
from . import record, collection
import sys

# Make sure submodule reference is consistent
sys.modules[__name__ + ".record"] = record
sys.modules[__name__ + ".collection"] = collection

# Shortcut alias (points to same class)
KalmanStateRecord = record.KalmanStateRecord
KalmanStateCollection = collection.KalmanStateCollection

__all__ = ["KalmanStateRecord", "KalmanStateCollection", "record", "collection"]