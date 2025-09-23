import numpy as np
import matplotlib.pyplot as plt

from . import OscillatorCompositeModel
class SuspensionModel(OscillatorCompositeModel):

    NAME = "Suspension Model"
    DESCRIPTION = "Composite model for simulating the behavior of a suspension system."
    VERSION = "1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat