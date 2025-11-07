import numpy as np
import matplotlib.pyplot as plt

from . import OscillatorCompositeModel
class PowerGridStateModel(OscillatorCompositeModel):

    NAME = "Power Grid State Model"
    DESCRIPTION = "Composite model for simulating the power lines of a grid system."
    VERSION = "1.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def preprocess(self, zk, uk=None, **kwargs):
        return zk, uk

    def inprocess(self, zk_hat, **kwargs):
        return zk_hat

    def postprocess(self, zk_hat, **kwargs):
        return zk_hat