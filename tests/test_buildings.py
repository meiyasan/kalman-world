"""The CO2 mass balance of a room (kalman.apps.buildings)."""
import sys
import unittest
import warnings

import numpy as np

sys.path.insert(0, ".")

from modules.apps import RoomCO2Model, air_conditioner, door, l_shaped_room, ventilation_from_decay, window
from modules.framework import compose, route
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

def fall(room, steps=180, start=1200.0):
    """The room's own fall from ``start`` ppm with nobody in: times [s] and the mean level."""
    x = np.full((room.n, 1), start)
    levels = []
    for k in range(steps):
        levels.append(float(np.mean(x)))
        x = room.f(x, u=np.array([[0.0], [1.0]]), t=k * 60.0)
    return 60.0 * np.arange(steps), np.asarray(levels)


class TestOpenings(unittest.TestCase):
    """A window, a door, an air conditioner: what each does to the balance."""

    def test_shut_openings_change_nothing(self):
        plain = RoomCO2Model("A")
        fitted = RoomCO2Model("A", openings=[window("window", [(4, 0)]), door("door", [(0, 5)]), air_conditioner()])
        np.testing.assert_array_equal(fitted.F(), plain.F())
        np.testing.assert_array_equal(fitted.B(), plain.B())

    def test_fresh_air_adds_to_the_ventilation(self):
        """An air conditioner with an intake of 3 air changes per hour, in a room at 2: the fall is at 5."""
        ach, _ = ventilation_from_decay(*fall(RoomCO2Model("A", openings=[air_conditioner(ach=3.0, on=True)])))
        self.assertAlmostEqual(ach, 5.0, delta=0.01)

    def test_a_recirculating_unit_stirs_and_renews_nothing(self):
        people = np.array([[4.0], [1.0]])
        spread = {}
        for on in (False, True):
            room = RoomCO2Model("A", openings=[air_conditioner(ach=0.0, mixing=0.1, on=on)])
            ach, _ = ventilation_from_decay(*fall(room))
            self.assertAlmostEqual(ach, 2.0, delta=0.01)        # the fall is the room's own, on or off
            x = room.x0
            for _ in range(600):
                x = room.f(x, u=people)
            spread[on] = float(np.ptp(x))
        self.assertLess(spread[True], 0.5 * spread[False])       # and the room is more even

    def test_a_window_airs_the_room_from_its_side(self):
        opened = RoomCO2Model("A", openings=[window("window", [(4, 0), (5, 0)], flow=0.1, open=True)])
        shut = RoomCO2Model("A")
        people = np.array([[4.0], [1.0]])
        xo, xs = opened.x0, shut.x0
        for _ in range(600):
            xo, xs = opened.f(xo, u=people), shut.f(xs, u=people)
        self.assertLess(float(np.mean(xo)), float(np.mean(xs)) - 100.0)
        field = opened.field(xo)
        self.assertLess(field[4, 0], field[4, 5])                 # lowest at the window

    def test_a_door_lets_in_the_air_beyond(self):
        """An empty room whose door gives on a corridor at 900 ppm settles between the two levels."""
        room = RoomCO2Model("A", openings=[door("door", [(0, 5)], flow=0.05, beyond=900.0, open=True)])
        x = room.x0
        for _ in range(600):
            x = room.f(x, u=np.array([[0.0], [1.0]]))
        field = room.field(x)
        self.assertGreater(np.nanmin(field), 420.0)
        self.assertLess(np.nanmax(field), 900.0)
        self.assertEqual(np.unravel_index(np.nanargmax(field), field.shape), (0, 5))

    def test_a_schedule_follows_the_steps(self):
        steps = np.zeros(10)
        steps[4:7] = 1.0
        room = RoomCO2Model("A", openings=[window("window", [(4, 0)], open=steps)])
        opened = RoomCO2Model("A", openings=[window("window", [(4, 0)], open=True)])
        np.testing.assert_array_equal(room.F(t=3 * 60.0), RoomCO2Model("A").F())
        np.testing.assert_array_equal(room.F(t=5 * 60.0), opened.F())
        np.testing.assert_array_equal(room.F(t=8 * 60.0), RoomCO2Model("A").F())

    def test_variant_and_what_is_refused(self):
        room = RoomCO2Model("A", count_people=True, openings=[window("window", [(4, 0)])])
        opened = room.variant(window=True)
        self.assertTrue(opened.count_people)
        self.assertFalse(np.array_equal(opened.F(), room.F()))
        with self.assertRaises(KeyError):
            room.variant(skylight=True)
        with self.assertRaises(ValueError):
            RoomCO2Model("A", openings=[window("window", [(10, 10)])])       # not in the room
        with self.assertRaises(ValueError):
            RoomCO2Model("A", openings=[window("w", [(4, 0)]), door("w", [(0, 5)])])


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestCountingWithAWindow(unittest.TestCase):
    """Four people; the window is open for an hour. The count holds when the window
    is reported, and when it is not but the filter weighs both regimes; it falls
    when the window is left out."""

    SPOTS = {"B": (2, 9), "C": (9, 2), "D": (5, 5)}

    def build(self, count, opened):
        room = RoomCO2Model("A", count_people=count,
                            openings=[window("window", [(4, 0), (5, 0)], flow=0.1, open=opened)])
        model = room
        for name, at in self.SPOTS.items():
            model = model.add_sensor(room.sensor(name, at))
        return room, model

    def test_known_ignored_and_routed(self):
        warnings.simplefilter("ignore")
        people = np.where(MINUTE < 30, 0, 4).astype(float)
        opened = ((MINUTE >= 120) & (MINUTE < 180)).astype(float)
        _, truth = self.build(False, opened)
        data = truth.sample(T * 60, uk=np.column_stack([people, np.ones(T)]), seed=2)
        ones = np.ones((T, 1))
        while_open = slice(150, 180)

        room, known = self.build(True, opened)
        count = known.filter(data.zk, uk=ones, reset=True).xk[:, room.n - 1]
        self.assertAlmostEqual(count[while_open].mean(), 4.0, delta=0.3)

        _, ignored = self.build(True, False)
        count = ignored.filter(data.zk, uk=ones, reset=True).xk[:, room.n - 1]
        self.assertLess(count[while_open].mean(), 3.0)

        _, shut = self.build(True, False)
        _, wide = self.build(True, True)
        modes = route([shut, wide], np.array([[0.98, 0.02], [0.02, 0.98]]))
        count = modes.filter(data.zk, uk=ones, reset=True).xk[:, room.n - 1]
        self.assertAlmostEqual(count[while_open].mean(), 4.0, delta=0.3)
        is_open = modes.mode_probabilities[:, 1]
        self.assertGreater(is_open[while_open].mean(), 0.9)       # the filter says when it was open
        self.assertLess(is_open[60:115].mean(), 0.1)
        self.assertLess(is_open[200:].mean(), 0.1)


if __name__ == "__main__":
    unittest.main()
