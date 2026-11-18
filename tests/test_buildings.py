"""The CO2 mass balance of a room (kalman.apps.buildings)."""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.apps import RoomCO2Model, l_shaped_room
from modules.framework import compose
from modules.framework.native import core

T = 240
MINUTE = np.arange(T)
PEOPLE = np.where(MINUTE < 30, 0, np.where(MINUTE < 120, 6, 2)).astype(float)


def sensors(room):
    return (room.sensor("B", (2, 9), rate=1 / 60), room.sensor("C", (9, 2), rate=1 / 300),
            room.sensor("D", (5, 5), rate=1 / 600))


class TestTheBalance(unittest.TestCase):

    def test_the_room(self):
        room = l_shaped_room()
        self.assertEqual(room.sum(), 108)
        self.assertFalse(room[6:, 6:].any())
        with self.assertRaises(ValueError):
            RoomCO2Model("A", at=(10, 10))                     # the missing quarter is not in the room

    def test_an_empty_room_stays_at_the_outdoor_level(self):
        room = RoomCO2Model("A")
        x = room.x0
        for _ in range(60):
            x = room.f(x, u=np.array([[0.0], [1.0]]))
        np.testing.assert_allclose(x, 420.0, atol=1e-9)

    def test_steady_state_is_the_balance_of_breath_and_ventilation(self):
        """N people breathing q into a room of volume V ventilated at lambda: c - c_out = N q / (lambda V)."""
        room = RoomCO2Model("A", mixing=1.0)                   # strong mixing: the room is well mixed
        x = room.x0
        for _ in range(2000):
            x = room.f(x, u=np.array([[4.0], [1.0]]))
        volume = 108 * 0.5 * 0.5 * 2.5
        expected = 420.0 + 4 * 0.005e-3 / (2.0 / 3600 * volume) * 1e6
        self.assertAlmostEqual(float(np.mean(x)), expected, delta=0.5)

    def test_a_model_of_matrices(self):
        from modules.framework.constants import KalmanEngineKind
        self.assertEqual(RoomCO2Model("A").DEFAULT_ENGINE, KalmanEngineKind.LINEAR)
        self.assertEqual(RoomCO2Model("A", count_people=True).n, 109)


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestWhatTheSensorsTell(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")
        self.truth_room = RoomCO2Model("A", at=(2, 2))
        B, C, D = sensors(self.truth_room)
        self.truth = self.truth_room.add_sensor(B).add_sensor(C).add_sensor(D).sample(
            T * 60, uk=np.column_stack([PEOPLE, np.ones(T)]), seed=2)

    def test_the_people_are_counted_from_the_co2(self):
        room = RoomCO2Model("A", at=(2, 2), count_people=True)
        B, C, D = sensors(room)
        record = room.add_sensor(B).add_sensor(C).add_sensor(D).filter(self.truth.zk, uk=np.ones((T, 1)), reset=True)
        count = record.xk[:, -1]
        self.assertAlmostEqual(float(np.mean(count[60:115])), 6.0, delta=0.5)
        self.assertAlmostEqual(float(np.mean(count[160:])), 2.0, delta=0.5)
        self.assertAlmostEqual(float(np.mean(count[5:28])), 0.0, delta=0.5)

    def test_a_drifted_sensor_is_calibrated_with_the_field(self):
        B, C, D = sensors(self.truth_room)
        real = self.truth_room.add_instrument(compose(B, name="B").calibrate(1.0, 100.0)).add_sensor(C).add_sensor(D)
        data = real.sample(T * 60, uk=np.column_stack([PEOPLE, np.ones(T)]), seed=2)
        room = RoomCO2Model("A", at=(2, 2), count_people=True)
        B, C, D = sensors(room)
        believed = room.add_instrument(compose(B, name="B").calibrate(1.0, 0.0)).add_sensor(C).add_sensor(D)
        plain = believed.filter(data.zk, uk=np.ones((T, 1)), reset=True)
        record = believed.joint({"B.offset": (150.0, 0.0)}).filter(data.zk, uk=np.ones((T, 1)), reset=True)
        n = room.n
        offset, sigma = record.xk[-1, n], np.sqrt(record.Pk[-1, n, n])
        self.assertLess(abs(offset - 100.0), 4 * sigma)
        self.assertLess(sigma, 5.0)
        error = lambda r: np.sqrt(np.mean((r.xk[60:, :n - 1] - data.xk[60:]) ** 2))
        self.assertLess(error(record), 0.5 * error(plain))


if __name__ == "__main__":
    unittest.main()
