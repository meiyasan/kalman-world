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

Notebook: [15_CO2Room](../../share/notebooks/15_CO2Room.ipynb).
