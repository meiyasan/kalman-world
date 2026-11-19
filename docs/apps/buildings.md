# CO2 in a room

People in a room breathe out CO2 and ventilation brings outdoor air in. A few
CO2 sensors, each reporting at its own rate, read the air at a few points; the
filter estimates the CO2 in every part of the room, how many people are in it,
and how far a sensor has drifted.

## The model

```
state        x = the CO2 concentration in each cell of the room [ppm]
                 (+ the number of people, with count_people=True)
dynamics     dc/dt = D lap(c) - lambda (c - c_out) + S / V
measurement  one CO2 sensor reads one cell
```

The room is a set of cells of a grid; an L-shaped room is the grid without one
quarter (`l_shaped_room()`). `D lap(c)` mixes neighbouring cells, and no air
crosses a wall. `lambda` is the ventilation in air changes per second, `c_out`
the outdoor concentration, `S / V` the breath of the occupants spread over the
seats.

## How it is built

- **Linear, so matrices.** The balance is linear: its step over a minute is a
  matrix exponential computed once, and the model runs on the linear engine.
- **The people as an input or as a state.** By default the number of people is
  a known input, `u = [occupants, 1]`. With `count_people=True` it is the last
  state component, a slowly drifting number the CO2 reveals.
- **Sensors at their own rates.** `model.sensor(name, (row, column), rate=...)`
  gives a CO2 sensor; one that has nothing new gives no number, and the filter
  skips it.
- **A drifted sensor is an instrument.** `compose(sensor, name="B").calibrate(1, 0)`
  gives the model `B.gain` and `B.offset`; `joint` estimates the offset with the
  field, as long as other sensors read the same air without it.

## Using it

```python
from kalman.apps import RoomCO2Model
from kalman.framework import compose

room = RoomCO2Model("A", at=(2, 2), count_people=True)
B = room.sensor("B", (2, 9), rate=1 / 60)
C = room.sensor("C", (9, 2), rate=1 / 300)
model = room.add_instrument(compose(B, name="B").calibrate(1.0, 0.0)).add_sensor(C)
record = model.joint({"B.offset": (150.0, 0.0)}).filter(zk, uk=np.ones((len(zk), 1)))
```

## Windows, doors and air conditioning

An opening changes the balance while it is open:

| | what it does | made with |
|---|---|---|
| window | exchanges the air of its cells with the outdoors | `window(name, cells, flow=0.1)` |
| door | exchanges it with the space beyond, at its own CO2 level | `door(name, cells, flow=0.05, beyond=600)` |
| air conditioner | stirs the room; with a fresh-air intake, also renews it | `air_conditioner(ach=0, mixing=0.03)` |

`flow` is in m^3/s. An air conditioner that only recirculates has `ach=0`: it
makes the room more even and removes nothing. Whether a unit brings fresh air
is read from the room itself: `ventilation_from_decay` on an evening with the
unit on, and on one with it off; the difference is its `ach`.

How open each one is, is said in one of three ways:

```python
from kalman.apps import RoomCO2Model, window, air_conditioner
from kalman.framework import route

# always (a model of matrices)
room = RoomCO2Model("A", openings=[window("window", [(4, 0), (5, 0)], open=True)])

# reported: one value per step of the model (a contact sensor, the unit's own signal)
room = RoomCO2Model("A", count_people=True,
                    openings=[window("window", [(4, 0), (5, 0)], open=contact), air_conditioner(ach=1.5, on=running)])

# not reported: the two regimes under route; the filter says when it was open
shut = RoomCO2Model("A", count_people=True, openings=[window("window", [(4, 0), (5, 0)])])
modes = route([shut, shut.variant(window=True)], [[0.98, 0.02], [0.02, 0.98]])
record = modes.filter(zk, uk=np.ones((len(zk), 1)))
modes.mode_probabilities[:, 1]                             # the probability that the window is open
```

With four people and the window open for an hour, the count stays at 4.0 when
the window is reported, and at 4.0 when it is routed (which finds the hour);
with the window left out of the model it falls to 2.3. A reported opening
makes `F` and `B` functions of time, each the exact step of the balance in
that configuration.

## A known empty room

What is known of the people is a sensor too: `room.schedule()` reads the count,
and `empty_at_night(room, hours=(0, 6), tz=...)` is an instrument that reads 0
people at those hours and nothing at the others. Every night then pins the
offsets of the CO2 sensors against the outdoor level the empty room settles to.

## Real sensors: Netatmo

`kalman.adapters.NetatmoModule` is one Netatmo indoor module: a CO2 sensor of
the room with its data sheet (one reading every 5 minutes, so `fs=1/300`) and
a calibration. Its readings come from the Netatmo web API (`NetatmoCloud`,
credentials from the environment) or from files (`NetatmoCSV`: the web page's
export or a local logger), and are laid on the model's steps, NaN where a
reading is missing.

```python
from kalman.apps import RoomCO2Model, empty_at_night
from kalman.adapters import NetatmoModule, NetatmoCloud

room = RoomCO2Model("unread", count_people=True, fs=1 / 300)
cloud = NetatmoCloud()
modules = [NetatmoModule(room, "main", (2, 2), station=STATION, source=cloud),
           NetatmoModule(room, "desk", (2, 9), station=STATION, module=DESK, source=cloud)]
watched = room.add_instrument(modules[0]).add_instrument(modules[1]).add_instrument(empty_at_night(room))
zk = watched.measurements(start=t0, end=t1)
record = watched.joint({"desk.offset": (50.0, 0.5)}).filter(zk, uk=np.ones((len(zk), 1)))
```

An offset is a slow random walk (`(sigma, drift)`): it does not return to
where it started, so it takes no `tau`.

**The ventilation sets the scale of the count.** The CO2 a person adds at
steady state is `q / (ach V)`: a room believed twice as ventilated as it is
counts twice the people (with `ach=3` given for a room at 2, five people are
counted as 6.9). `ach` has to be measured for the room: from the decay of the
CO2 after everybody has left (its rate is `ach`), or from an hour with a known
number of people. `ventilation_from_decay(times, ppm)` does the first: it fits
`a + b exp(-lambda t)` to each sensor's readings of the fall and returns `ach`
and its standard deviation. The level `a` is fitted, so the sensors' offsets do
not enter.

```python
from kalman.apps import ventilation_from_decay

evening = slice(k_left, k_left + 48)                       # four hours from the moment the room empties
ach, sigma = ventilation_from_decay(t[evening], zk[evening, :5])
room = RoomCO2Model("unread", count_people=True, fs=1 / 300, ach=ach)
```

Notebook: [15_CO2Room](../../share/notebooks/15_CO2Room.ipynb), whose last step opens a window.
