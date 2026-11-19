"""CO2 in a room: a mass balance on a grid of cells.

The room is a set of cells of a rectangular grid (an L-shaped room is the grid
without one quarter). In each cell the CO2 concentration ``c`` [ppm] follows

    dc/dt = D lap(c) - lambda (c - c_out) + S / V

- ``D lap(c)``: mixing between neighbouring cells of the room (no flow through
  the walls: a cell exchanges only with its neighbours inside the room);
- ``lambda (c - c_out)``: ventilation, ``lambda`` the air changes per second,
  bringing outdoor air at ``c_out``;
- ``S / V``: the CO2 the occupants breathe out, spread over the seats.

The equation is linear, so the model is a model of matrices: ``F`` and ``B``
are its exact discretization over one step (matrix exponential), and it runs on
the linear engine. The known inputs ``u = [occupants, 1]`` are the number of
people in the room and a constant 1 that carries the outdoor air.

With ``count_people=True`` the number of occupants is not given but estimated:
it becomes the last component of the state, a number that drifts slowly
(people come and go), and ``u = [1]``. The model stays linear.

Openings change the balance while they are open (:class:`Opening`): a window
exchanges the air of its cells with the outdoors, a door with the space beyond
it, and an air conditioner brings fresh air to the whole room, or only stirs
it. An opening that is always open or always shut leaves the model a model of
matrices; one that follows a known schedule (a contact sensor, the unit's own
signal) makes ``F`` and ``B`` functions of time, each the exact discretization
of the balance in that configuration. An opening nobody reports is two models,
shut and open, under ``route``.

The model's own sensor reads one cell. Other sensors are added with the
framework's operations; a sensor that drifts is an instrument with a
calibration, whose offset ``joint`` estimates with the field.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ...framework.models.linear import LinearKalmanModel
from ...framework.sensor import KalmanSensor


class Opening:
    """Something that changes the air of the room while it is open: a window, a
    door, an air conditioner.

    Parameters
    ----------
    name : what it is called.
    cells : the cells it acts on, ``(row, column)`` each; ``None`` for the whole room.
    flow : the air it exchanges while open [m^3/s], shared by its cells.
    ach : that exchange as air changes per hour of the whole room, in place of ``flow``.
    level : the CO2 of the air that comes in [ppm]; ``None`` for the outdoor level.
    mixing : what it adds to the room's mixing coefficient while open [m^2/s].
    open : ``True`` or ``False`` for always; or how open it is at each step, an
        array (0 shut, 1 open, one value per step of the model) or a function of
        the time [s].
    """

    def __init__(self, name, cells=None, *, flow=0.0, ach=None, level=None, mixing=0.0, open=False):
        if not isinstance(name, str) or not name:
            raise ValueError("Opening: an opening needs a name.")
        if float(flow) < 0.0 or float(mixing) < 0.0 or (ach is not None and float(ach) < 0.0):
            raise ValueError(f"Opening {name!r}: flow, ach and mixing cannot be negative.")
        self.name = name
        self.cells = None if cells is None else [tuple(c) for c in cells]
        self.flow, self.ach = float(flow), None if ach is None else float(ach)
        self.level = None if level is None else float(level)
        self.mixing = float(mixing)
        self.open = open

    @property
    def scheduled(self):
        """``True`` when how open it is changes with time."""
        return not isinstance(self.open, (bool, int, float, np.bool_))

    def state(self, t, fs):
        """How open it is at time ``t`` [s], from 0 to 1."""
        if not self.scheduled:
            value = float(self.open)
        elif callable(self.open):
            value = float(self.open(0.0 if t is None else float(t)))
        else:
            steps = np.asarray(self.open, dtype=float).reshape(-1)
            k = 0 if t is None else int(round(float(t) * fs))
            value = float(steps[min(max(k, 0), len(steps) - 1)])
        return min(max(round(value, 3), 0.0), 1.0)

    def __repr__(self):
        return f"Opening({self.name!r}, open={'scheduled' if self.scheduled else bool(self.open)})"


def window(name, cells, flow=0.1, *, open=False):
    """A window on the outdoors: ``flow`` [m^3/s] through its ``cells`` while open."""
    return Opening(name, cells, flow=flow, open=open)


def door(name, cells, flow=0.05, *, beyond=600.0, open=False):
    """A door on another space whose CO2 is ``beyond`` [ppm]: ``flow`` [m^3/s]
    through its ``cells`` while open."""
    return Opening(name, cells, flow=flow, level=beyond, open=open)


def air_conditioner(name="ac", *, ach=0.0, mixing=0.03, on=False):
    """An air conditioner: while on it stirs the room (``mixing`` [m^2/s]) and,
    when it has a fresh-air intake, renews it (``ach`` more air changes per hour;
    0 for a unit that only recirculates)."""
    return Opening(name, None, ach=ach, mixing=mixing, open=on)


def l_shaped_room(rows: int = 12, cols: int = 12) -> np.ndarray:
    """The cells of an L-shaped room: the grid without its upper-right quarter."""
    room = np.ones((rows, cols), dtype=bool)
    room[rows // 2:, cols // 2:] = False
    return room


class RoomCO2Model(LinearKalmanModel):
    """CO2 concentration in every cell of a room.

    Parameters
    ----------
    sensor : the name of the model's own CO2 sensor.
    at : ``(row, column)`` of the cell that sensor reads.
    room : boolean grid, True for the cells of the room; an L by default.
    cell : the side of a cell [m]; height : the height of the room [m].
    mixing : the mixing coefficient D [m^2/s].
    ach : ventilation, in air changes per hour.
    outdoor : the outdoor concentration [ppm].
    seats : the cells where the occupants sit; their breath is spread over them.
    emission : the CO2 one person breathes out [L/s].
    sigma : the noise of the model's own sensor [ppm].
    process_uncertainty : what the balance leaves out (a door opened, a person
        moving), as a standard deviation per cell and per step [ppm].
    count_people : estimate the number of occupants as the last state component
        instead of taking it as an input.
    people_drift : how much that number may change in a step, a standard
        deviation [people]; used with ``count_people``.
    openings : the windows, doors and air conditioners of the room
        (:class:`Opening`, or :func:`window`, :func:`door`, :func:`air_conditioner`).
    fs : steps per second; one step a minute by default.
    """

    NAME = "Room CO2 Model"
    DESCRIPTION = "CO2 mass balance on the cells of a room: mixing, ventilation and occupants."
    VERSION = "1.0"

    def __init__(self, sensor: str = "co2", at=(2, 2), *, room=None, cell: float = 0.5, height: float = 2.5,
                 mixing: float = 0.02, ach: float = 2.0, outdoor: float = 420.0, seats=None,
                 emission: float = 0.005, sigma: float = 15.0, process_uncertainty: float = 2.0,
                 count_people: bool = False, people_drift: float = 0.3, openings=(), fs: float = 1 / 60, **kwargs):
        self._made_with = dict(sensor=sensor, at=at, room=room, cell=cell, height=height, mixing=mixing, ach=ach,
                               outdoor=outdoor, seats=seats, emission=emission, sigma=sigma,
                               process_uncertainty=process_uncertainty, count_people=count_people,
                               people_drift=people_drift, openings=tuple(openings), fs=fs, **kwargs)
        self.room = l_shaped_room() if room is None else np.asarray(room, dtype=bool)
        rows, cols = self.room.shape
        self.cells = [(r, c) for r in range(rows) for c in range(cols) if self.room[r, c]]
        index = {rc: i for i, rc in enumerate(self.cells)}
        self.index = index
        cells_n = len(self.cells)
        self.count_people = bool(count_people)
        n = cells_n + (1 if self.count_people else 0)
        self.cell, self.height, self.outdoor = float(cell), float(height), float(outdoor)
        seats = [(1, 1), (1, 2), (2, 1), (2, 2)] if seats is None else [tuple(s) for s in seats]
        for s in [tuple(at), *seats]:
            if s not in index:
                raise ValueError(f"RoomCO2Model: the cell {s} is not in the room.")
        self.seats = seats
        self.openings = list(openings)
        names = [o.name for o in self.openings]
        if len(set(names)) != len(names):
            raise ValueError(f"RoomCO2Model: two openings of one name ({names}).")

        # The continuous balance dc/dt = A c + Bc u, u = [occupants, 1].
        lam = float(ach) / 3600.0
        volume = self.cell * self.cell * self.height                     # of one cell [m^3]
        neighbours = np.zeros((cells_n, cells_n))                        # the exchange between cells, per unit k
        for (r, c), i in index.items():
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                j = index.get(nb)
                if j is not None:                       # a wall exchanges nothing
                    neighbours[i, i] -= 1.0
                    neighbours[i, j] += 1.0
        per_person = float(emission) * 1e-3 / volume * 1e6                # ppm/s in one cell
        breath = np.zeros(n)
        for s in seats:
            breath[index[s]] += per_person / len(seats)
        rates = []                                      # per opening: its exchange in each cell [1/s], fully open
        for o in self.openings:
            rate = np.zeros(cells_n)
            if o.cells is None:
                rate[:] = o.ach / 3600.0 if o.ach is not None else o.flow / (volume * cells_n)
            else:
                for c in o.cells:
                    if c not in index:
                        raise ValueError(f"RoomCO2Model: the cell {c} of {o.name!r} is not in the room.")
                per_cell = o.ach / 3600.0 * cells_n if o.ach is not None else o.flow / volume
                rate[[index[c] for c in o.cells]] = per_cell / len(o.cells)
            rates.append(rate)
        dt = 1.0 / float(fs)
        made = {}

        def matrices(states):
            """``F`` and ``B`` over one step with each opening as open as ``states`` says: the
            exact discretization, the inputs held, exp([[A, Bc], [0, 0]] dt)."""
            if states not in made:
                exchange = np.full(cells_n, lam)
                incoming = np.full(cells_n, lam * self.outdoor)
                k = float(mixing)
                for o, rate, s in zip(self.openings, rates, states):
                    exchange = exchange + s * rate
                    incoming = incoming + s * rate * (self.outdoor if o.level is None else o.level)
                    k += s * o.mixing
                A = np.zeros((n, n))
                A[:cells_n, :cells_n] = k / self.cell ** 2 * neighbours - np.diag(exchange)
                outdoor_air = np.zeros(n)
                outdoor_air[:cells_n] = incoming
                if self.count_people:
                    A[:, cells_n] = breath                  # the occupants are the last state
                    Bc = outdoor_air.reshape(n, 1)
                else:
                    Bc = np.column_stack([breath, outdoor_air])
                width = Bc.shape[1]
                block = np.zeros((n + width, n + width))
                block[:n, :n], block[:n, n:] = A, Bc
                E = expm(block * dt)
                made[states] = (E[:n, :n], E[:n, n:])
            return made[states]

        fs_ = float(fs)
        states_at = lambda t: tuple(o.state(t, fs_) for o in self.openings)
        if any(o.scheduled for o in self.openings):     # a known schedule: F and B follow it
            F, B = (lambda t=None: matrices(states_at(t))[0]), (lambda t=None: matrices(states_at(t))[1])
        else:
            F, B = matrices(states_at(None))
        H = np.zeros((1, n))
        H[0, index[tuple(at)]] = 1.0
        q = np.full(n, float(process_uncertainty) ** 2)
        x0 = np.full((n, 1), self.outdoor)
        P0 = np.full(n, 50.0 ** 2)
        if self.count_people:
            q[-1] = float(people_drift) ** 2
            x0[-1], P0[-1] = 0.0, 10.0 ** 2
        kwargs.setdefault("x0", x0)
        kwargs.setdefault("P0", np.diag(P0))
        super().__init__(sensor, F=F, B=B, H=H, Q=np.diag(q), R=np.array([[float(sigma) ** 2]]), fs=fs, **kwargs)
        self.at = tuple(at)

    def variant(self, **open):
        """The same room with some openings set otherwise: ``room.variant(window=True)``.
        The two are the modes of ``route`` when nobody reports the window."""
        unknown = set(open) - {o.name for o in self.openings}
        if unknown:
            raise KeyError(f"RoomCO2Model has no opening {sorted(unknown)}; it has {[o.name for o in self.openings]}.")
        openings = [Opening(o.name, o.cells, flow=o.flow, ach=o.ach, level=o.level, mixing=o.mixing,
                            open=open.get(o.name, o.open)) for o in self.openings]
        return type(self)(**{**self._made_with, "openings": openings})

    def sensor(self, name, at, sigma=15.0, *, rate=None):
        """A CO2 sensor that reads the cell ``at`` with noise ``sigma`` [ppm],
        ``rate`` readings per second (``None``: at every step of the model)."""
        if tuple(at) not in self.index:
            raise ValueError(f"RoomCO2Model.sensor: the cell {at} is not in the room.")
        G = np.zeros((1, int(self.n)))
        G[0, self.index[tuple(at)]] = 1.0
        return KalmanSensor(name, G=G, R=float(sigma) ** 2, rate=rate, labels={name: "ppm"})

    def schedule(self, name="schedule", sigma=0.1, *, rate=None):
        """What is known of the number of occupants, as a sensor that reads it.

        A room known to be empty (at night, say) is a reading of 0 people with a
        small ``sigma``; at every other step the reading is NaN, and the filter
        counts from the CO2 alone. Needs ``count_people=True``.
        """
        if not self.count_people:
            raise ValueError("RoomCO2Model.schedule: the number of occupants is a state only with count_people=True.")
        G = np.zeros((1, int(self.n)))
        G[0, -1] = 1.0
        return KalmanSensor(name, G=G, R=float(sigma) ** 2, rate=rate, labels={name: "people"})

    def field(self, x) -> np.ndarray:
        """The concentration as a grid, NaN outside the room."""
        grid = np.full(self.room.shape, np.nan)
        x = np.asarray(x, dtype=float).reshape(-1)
        for (r, c), i in self.index.items():
            grid[r, c] = x[i]
        return grid


def empty_at_night(room, hours=(0, 6), *, tz=None, sigma=0.1, name="schedule"):
    """The room is empty between two hours of the night: an instrument made of the
    room's ``schedule`` sensor, whose readings are 0 people at those steps and NaN
    at the others.

    ``empty_at_night(room)`` is added to the room model; its readings
    come with the others from ``model.measurements(start=..., end=...)`` [unix s].
    ``hours=(0, 6)``: from midnight to 6 am, in the time zone ``tz`` (the
    computer's when ``None``).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from ...framework.instrument import KalmanInstrument
    first, last = (float(h) for h in hours)
    step = 1.0 / float(room.fs)

    def readings(*, start, end, **query):
        zone = ZoneInfo(tz) if tz else None
        n = max(int(np.floor((float(end) - float(start)) / step)), 0)
        out = np.full((n, 1), np.nan)
        for k in range(n):
            moment = datetime.fromtimestamp(float(start) + k * step, zone)
            hour = moment.hour + moment.minute / 60.0
            if (first <= hour < last) if first <= last else (hour >= first or hour < last):
                out[k, 0] = 0.0
        return out

    return KalmanInstrument(name, [room.schedule(name, sigma=sigma)], provider=readings)


def ventilation_from_decay(times, ppm):
    """The ventilation of a room [air changes per hour] from the fall of its CO2
    once everybody has left, and its standard deviation.

    With nobody breathing, the balance leaves ``c(t) = a + b exp(-lambda t)``:
    the rate of the fall is the ventilation. The level ``a`` it falls to is
    fitted too, so a sensor's offset and the outdoor level do not enter.

    ``times`` [s] and ``ppm`` are the readings of the fall only (from the moment
    the room empties until the level is flat, or the next person comes in);
    ``ppm`` holds one column per sensor, each fitted alone and the rates
    averaged by their uncertainties. NaN readings are left out.
    """
    from scipy.optimize import curve_fit
    t = np.asarray(times, dtype=float).reshape(-1)
    c = np.asarray(ppm, dtype=float).reshape(len(t), -1)
    t = t - t[0]
    fall = lambda t, a, b, lam: a + b * np.exp(-lam * t)
    rates, variances = [], []
    for column in c.T:
        read = np.isfinite(column)
        if read.sum() < 6:
            continue
        tr, cr = t[read], column[read]
        start = (cr[-3:].mean(), max(cr[:3].mean() - cr[-3:].mean(), 1.0), 3.0 / max(tr[-1], 1.0))
        try:
            best, cov = curve_fit(fall, tr, cr, p0=start, bounds=([-np.inf, 0.0, 0.0], np.inf))
        except RuntimeError:
            continue
        if np.isfinite(cov[2, 2]) and cov[2, 2] > 0.0 and best[2] > 0.0:
            rates.append(best[2])
            variances.append(cov[2, 2])
    if not rates:
        raise ValueError("ventilation_from_decay: no fall of the CO2 could be fitted; give the readings from the "
                         "moment the room empties, at least six per sensor.")
    weights = 1.0 / np.asarray(variances)
    rate = float(np.sum(weights * np.asarray(rates)) / np.sum(weights))
    return rate * 3600.0, float(np.sqrt(1.0 / np.sum(weights))) * 3600.0
