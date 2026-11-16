"""compose: several models side by side, each with its own state.

``compose(a, b)`` (``a.compose(b)``, ``Composite``) is ``Kalman::Models::Composite``
seen from Python. Pinned here: the three ways in, the two readouts, the parts'
sensors and names, parameters as ``"<part>.<name>"`` and their estimation with
``joint``, a part that is code, and nesting with the other operations. The
passes of the oscillator composites are in ``test_native_composite``; the same
scenarios in C++ are ``tests/cxx/27_composite.cc``.
"""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.framework import (Composite, KalmanInstrument, KalmanModel, KalmanSensor, compose)
from modules.framework.constants import KalmanEngineKind, KalmanReadout, KalmanStructureKind
from modules.framework.native import core
from tests.test_joint import Oscillator

needs_core = unittest.skipUnless(core.HAS_CXX_CORE, "libKalman is not built")

FS = 10.0


def cart(name="cart", v0=1.0):
    dt = 1 / FS
    return KalmanModel(name, fs=FS, F=np.array([[1, dt], [0, 1]]), H=np.array([[1.0, 0.0]]),
                       Q=np.diag([1e-4, 1e-2]), R=np.array([[0.25]]), x0=np.array([[0.0], [v0]]), P0=np.eye(2))


@needs_core
class TestCompose(unittest.TestCase):

    def test_three_ways(self):
        for model in (compose(cart("a"), cart("b")), cart("a").compose(cart("b")), Composite(cart("a"), cart("b")),
                      compose([cart("a"), cart("b")])):
            self.assertIsInstance(model, Composite)
            self.assertEqual((model.n, model.m, model.l), (4, 2, 0))
            self.assertEqual(model.part_names, ["a", "b"])

    def test_stacked(self):
        a, b = cart("a", 1.0), cart("b", -2.0)
        model = compose(a, b)
        self.assertIs(model.structure, KalmanStructureKind.MATRICES)
        self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        np.testing.assert_array_equal(model.x0.ravel(), [0, 1, 0, -2])
        x = np.array([[1.0], [2.0], [10.0], [-4.0]])
        np.testing.assert_allclose(model.f(x).ravel(), [1.2, 2.0, 9.6, -4.0])
        np.testing.assert_array_equal(model.h(x).ravel(), [1.0, 10.0])
        F, H = model.F(), model.H()
        self.assertEqual((F[0, 1], F[2, 3], F[0, 2], H[0, 0], H[1, 2], H[0, 2]), (0.1, 0.1, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(model.R()[1, 1], 0.25)
        self.assertEqual([s.name for s in model.sensors], ["a", "b"])       # each part brings its sensor
        self.assertEqual(float(model.sensors[1].measure(x)[0, 0]), 10.0)
        self.assertEqual(model.part("b"), slice(2, 4))
        self.assertEqual(model.part(b), slice(2, 4))
        with self.assertRaises(KeyError):
            model.part("c")

    def test_summed(self):
        model = compose(cart("a"), cart("b"), readout=KalmanReadout.SUMMED)
        self.assertEqual((model.n, model.m), (4, 1))
        x = np.array([[1.0], [2.0], [10.0], [-4.0]])
        self.assertEqual(float(model.h(x)[0, 0]), 11.0)
        np.testing.assert_array_equal(model.H(), [[1.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_equal(model.R(), [[0.25]])                  # the noise of the one channel
        self.assertEqual(len(model.sensors), 1)

    def test_the_first_part_is_filtered_as_it_would_be_alone(self):
        a = cart("a")
        model = compose(a, cart("b", -2.0))
        rng = np.random.default_rng(3)
        zk = rng.standard_normal((200, 2)).cumsum(axis=0) * 0.1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            both = np.asarray(model.filter(zk, smooth=False, reset=True).xk).reshape(200, -1)
            alone = np.asarray(cart("a").filter(zk[:, :1], smooth=False, reset=True).xk).reshape(200, -1)
        np.testing.assert_allclose(both[:, :2], alone, atol=1e-10)

    def test_a_part_of_code(self):
        swing = Oscillator(damping=0.3)
        other = KalmanModel("cart", fs=swing.fs, F=np.eye(2), H=np.array([[1.0, 0.0]]), Q=1e-3 * np.eye(2),
                            R=np.array([[0.1]]), x0=np.zeros((2, 1)), P0=np.eye(2))
        model = compose(other, swing)
        self.assertIs(model.structure, KalmanStructureKind.MAPS)
        self.assertIs(model.DEFAULT_ENGINE, KalmanEngineKind.EXTENDED)
        x = np.array([[1.0], [2.0], [0.4], [-0.7]])
        np.testing.assert_allclose(model.f(x)[2:].ravel(), swing.f(x[2:]).ravel())

    def test_parameters_of_a_part(self):
        thermo = KalmanInstrument("thermo", [KalmanSensor("probe", G=[[1.0, 0.0]], R=0.04)], offset=0.5)
        left = cart("left").add_instrument(thermo)
        model = compose(left, cart("right"), names=["left", "right"])
        self.assertEqual(model.parameters()["left.thermo.offset"], 0.5)
        model.set_parameter("left.thermo.offset", 2.0)
        self.assertEqual(left.parameters()["thermo.offset"], 2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            joint = model.joint({"left.thermo.offset": (1.0, 0.0)})
        self.assertEqual(joint.n, 5)
        self.assertEqual(list(joint.states.__members__)[-1], "left.thermo.offset")

    def test_they_nest(self):
        pair = compose(cart("a"), cart("b"))
        self.assertEqual(compose(pair, cart("c")).n, 6)
        self.assertEqual(pair.lag(2).n, 12)
        self.assertEqual(pair.add_sensor(KalmanSensor("gap", G=[[1.0, 0.0, -1.0, 0.0]], R=0.01)).m, 3)

    def test_refusals(self):
        with self.assertRaises(TypeError):
            compose()
        with self.assertRaises(ValueError):
            Composite()
        with self.assertRaises(TypeError):
            compose(cart(), KalmanSensor("s", G=[[1.0, 0.0]], R=1.0))
        with self.assertRaises(TypeError):
            compose(cart("a"), cart("b"), name="pair")                      # name= is an instrument's
        with self.assertRaises(ValueError):
            compose(cart("a"), cart("b"), names=["same", "same"])
        with self.assertRaises(ValueError):
            compose(cart("a"), KalmanModel("slow", fs=20.0, F=np.eye(2), H=np.array([[1.0, 0.0]]), Q=np.eye(2),
                                           R=np.array([[1.0]]), x0=np.zeros((2, 1)), P0=np.eye(2)))
        wide = KalmanModel("wide", fs=FS, F=np.eye(2), H=np.eye(2), Q=np.eye(2), R=np.eye(2),
                           x0=np.zeros((2, 1)), P0=np.eye(2))
        with self.assertRaises(ValueError):
            compose(cart("a"), wide, readout=KalmanReadout.SUMMED)


if __name__ == "__main__":
    unittest.main()
