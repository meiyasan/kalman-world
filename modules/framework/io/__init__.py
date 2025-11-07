# io/__init__.py
from . import record, collection
import sys

# Make sure submodule reference is consistent
sys.modules[__name__ + ".record"] = record
sys.modules[__name__ + ".collection"] = collection

# Shortcut alias (points to same class)
KalmanRecord = record.KalmanRecord
KalmanRecordCollection = collection.KalmanRecordCollection

__all__ = ["KalmanRecord", "KalmanRecordCollection", "record", "collection"]