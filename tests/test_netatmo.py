"""The Netatmo adapter: readings on the model's steps, the web API and files,
and five modules whose offsets are estimated while the people are counted."""
import io
import json
import os
import tempfile
import unittest
import warnings
from unittest import mock
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from modules.adapters.netatmo import NetatmoCloud, NetatmoCSV, NetatmoModule, on_steps
from modules.apps.buildings import RoomCO2Model, empty_at_night, ventilation_from_decay
from modules.framework import compose
from modules.framework.native import core

TZ = "Europe/Paris"
T0 = datetime(2026, 11, 16, 0, 0, tzinfo=ZoneInfo(TZ)).timestamp()


class TestOnSteps(unittest.TestCase):

    def test_readings_go_to_the_nearest_step(self):
        z = on_steps([10.0, 290.0, 310.0, 1210.0], [400.0, 500.0, 600.0, 700.0], start=0.0, end=1500.0)
        self.assertEqual(z.shape, (5, 1))
        np.testing.assert_allclose(z[:, 0], [400.0, 550.0, np.nan, np.nan, 700.0])

    def test_outside_the_window_is_dropped(self):
        z = on_steps([-400.0, 2000.0], [1.0, 2.0], start=0.0, end=600.0)
        self.assertTrue(np.isnan(z).all())


class TestEmptyAtNight(unittest.TestCase):

    def test_zero_people_at_night_nothing_known_by_day(self):
        room = RoomCO2Model("A", count_people=True, fs=1 / 300)
        night = empty_at_night(room, tz=TZ)
        z = night.read(start=T0, end=T0 + 86400)["schedule"][:, 0]
        self.assertEqual(len(z), 288)
        np.testing.assert_array_equal(z[:72], 0.0)              # midnight to 6 am
        self.assertTrue(np.isnan(z[72:]).all())

    def test_needs_the_count(self):
        with self.assertRaises(ValueError):
            RoomCO2Model("A").schedule()


class TestFiles(unittest.TestCase):

    def test_the_web_export(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.csv")
            with open(path, "w") as f:
                f.write('"Room";\n"Timestamp";"Timezone : Europe/Paris";"Temperature";"Humidity";"CO2";"Noise"\n'
                        "1000;2026/11/16 00:00:00;20.1;40;512;35\n1300;x;20.1;40;;35\n1600;x;20.0;41;530;35\n")
            times, ppm = NetatmoCSV.parse(path)
            np.testing.assert_array_equal(times, [1000.0, 1600.0])
            np.testing.assert_array_equal(ppm, [512.0, 530.0])

    def test_a_logger_table(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.csv")
            with open(path, "w") as f:
                f.write("time,co2\n0,450\n300,460\n")
            source = NetatmoCSV({"m1": path})
            times, ppm = source.readings("station", "m1", 0, 300)
            np.testing.assert_array_equal(ppm, [450.0])


class FakeNetatmo:
    """The web API as far as the adapter uses it: a token, then pages of readings."""

    def __init__(self, n=2500):
        self.calls, self.n = [], n

    def __call__(self, request):
        url = request.full_url
        self.calls.append(request)
        if url.endswith("/oauth2/token"):
            return io.BytesIO(json.dumps({"access_token": "a", "expires_in": 10800,
                                          "refresh_token": "rotated"}).encode())
        from urllib.parse import parse_qs, urlparse
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        assert request.get_header("Authorization") == "Bearer a"
        times = [t for t in range(0, self.n * 300, 300) if int(q["date_begin"]) <= t <= int(q["date_end"])]
        times = times[:int(q["limit"])]
        return io.BytesIO(json.dumps({"status": "ok", "body": {str(t): [400 + t // 300 % 7] for t in times}}).encode())


class TestTheWebAPI(unittest.TestCase):

    SECRETS = {"NETATMO_CLIENT_ID": "id", "NETATMO_CLIENT_SECRET": "secret", "NETATMO_REFRESH_TOKEN": "first"}

    def test_pages_and_a_replaced_token(self):
        fake = FakeNetatmo()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, self.SECRETS):
            token_file = os.path.join(d, "kept", "netatmo.json")
            cloud = NetatmoCloud(token_file=token_file, opener=fake)
            times, ppm = cloud.readings("70:ee:50:00:00:01", "03:00:00:00:00:02", 0, 2500 * 300)
            self.assertEqual(len(times), 2500)                  # three pages of at most 1024
            self.assertEqual(sum("getmeasure" in c.full_url for c in fake.calls), 3)
            self.assertEqual(sum("oauth2" in c.full_url for c in fake.calls), 1)
            self.assertIn(b"refresh_token=first", fake.calls[0].data)
            self.assertEqual(json.load(open(token_file)), {"refresh_token": "rotated"})
            self.assertEqual(os.stat(token_file).st_mode & 0o777, 0o600)

            again = FakeNetatmo()                               # the next run uses the token that was kept
            NetatmoCloud(token_file=token_file, opener=again).readings("s", None, 0, 300)
            self.assertIn(b"refresh_token=rotated", again.calls[0].data)

    def test_a_refused_kept_token_falls_back_on_the_given_one(self):
        import urllib.error
        fake = FakeNetatmo()

        def opener(request):
            if request.data and b"refresh_token=stale" in request.data:
                raise urllib.error.HTTPError(request.full_url, 400, "invalid_grant", None, None)
            return fake(request)

        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, self.SECRETS):
            token_file = os.path.join(d, "netatmo.json")
            json.dump({"refresh_token": "stale"}, open(token_file, "w"))
            NetatmoCloud(token_file=token_file, opener=opener).readings("s", None, 0, 300)
            self.assertEqual(json.load(open(token_file)), {"refresh_token": "rotated"})

    def test_no_credentials_fails_closed(self):
        from modules.providers import credentials
        empty = {k: v for k, v in os.environ.items() if not k.startswith("NETATMO_")}
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, empty, clear=True), \
                mock.patch.object(credentials, "_STORE", os.path.join(d, "none.json")):
            with self.assertRaisesRegex(PermissionError, "NETATMO_CLIENT_ID"):
                NetatmoCloud(token_file=os.path.join(d, "t.json"), opener=FakeNetatmo()).readings("s", None, 0, 300)

    def test_a_file_without_the_columns_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.csv")
            open(path, "w").write("date,co2 level,timezone\n1,2,3\n")
            with self.assertRaisesRegex(ValueError, "time column"):
                NetatmoCSV.parse(path)


@unittest.skipUnless(core.HAS_CXX_CORE, "needs the C++ core")
class TestFiveModules(unittest.TestCase):
    """Two days, five modules, four of them off by an unknown offset; the room is
    empty at night. The offsets are found and the people counted."""

    def test_offsets_and_count(self):
        warnings.simplefilter("ignore")
        step, T = 300.0, 2 * 288
        t = T0 + step * np.arange(T)
        hour = np.array([datetime.fromtimestamp(x, ZoneInfo(TZ)).hour for x in t])
        people = np.where((hour >= 9) & (hour < 12), 5, np.where((hour >= 14) & (hour < 18), 3, 0)).astype(float)
        spots = {"main": (2, 2), "desk": (2, 9), "door": (9, 2), "centre": (5, 5), "window": (4, 1)}
        offsets = {"main": 0.0, "desk": 60.0, "door": -40.0, "centre": 25.0, "window": 90.0}

        room = RoomCO2Model("main", at=spots["main"], fs=1 / step)
        truth = room
        for name in list(spots)[1:]:
            truth = truth.add_instrument(compose(room.sensor(name, spots[name], sigma=20.0), name=name)
                                         .calibrate(1.0, offsets[name]))
        sim = truth.sample(T * step, uk=np.column_stack([people, np.ones(T)]), seed=3)

        with tempfile.TemporaryDirectory() as d:
            files = {}
            for j, name in enumerate(spots):
                files[name] = os.path.join(d, f"{name}.csv")
                lost = np.random.default_rng(j).random(T) < 0.05
                with open(files[name], "w") as f:
                    f.write("Timestamp;Timezone : Europe/Paris;CO2\n")
                    for k in np.flatnonzero(~lost):
                        f.write(f"{int(t[k]) + (k * 37) % 81 - 40};x;{sim.zk[k, j]:.0f}\n")
            source = NetatmoCSV(files)
            counting = RoomCO2Model("unread", count_people=True, fs=1 / step)
            modules = [NetatmoModule(counting, n, spots[n], station="main", module=None if n == "main" else n,
                                     source=source) for n in spots]
            watched = counting
            for module in modules:
                watched = watched.add_instrument(module)
            watched = watched.add_instrument(empty_at_night(counting, tz=TZ))
            zk = watched.measurements(start=T0, end=T0 + T * step)

        self.assertEqual(zk.shape, (T, 7))

        # the ventilation, from the fall after the last person left at 6 pm of the first day (truth: 2 per hour)
        left = int(np.flatnonzero(hour == 18)[0])
        ach, sigma = ventilation_from_decay(t[left:left + 48], zk[left:left + 48, 1:6])
        self.assertLess(abs(ach - 2.0), max(3 * sigma, 0.1))
        self.assertLess(sigma, 0.2)
        estimated = [n for n in spots if n != "main"]
        run = watched.joint({f"{n}.offset": (60.0, 0.5) for n in estimated}).filter(zk, uk=np.ones((T, 1)), reset=True)
        n = counting.n
        for i, name in enumerate(estimated):
            sigma = np.sqrt(run.Pk[-1, n + i, n + i])
            self.assertLess(abs(run.xk[-1, n + i] - offsets[name]), max(3 * sigma, 8.0), name)
        error = run.xk[:, n - 1] - people
        self.assertLess(np.sqrt(np.mean(error[people > 0] ** 2)), 0.6)
        self.assertLess(np.sqrt(np.mean(error[people == 0] ** 2)), 0.4)


if __name__ == "__main__":
    unittest.main()
