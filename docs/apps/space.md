# Space navigation

A spacecraft coasting between worlds, tracked from the ground, and the tools
that plan such a trip.

## The models

| Class | Trip | Forces |
|---|---|---|
| `TransLunarModel` | Earth to the Moon | Earth, Moon and Sun |
| `InterplanetaryModel` | one planet to another (Mars is the worked case) | the Sun, plus the pull of the planets |

```
state        x = [position, velocity]           (megametres and km/s, six numbers)
dynamics     velocity' = the sum of the pulls of each body
measurement  z = position                       (ground-station tracking)
```

## Planning

`kalman.apps` also has the planning tools, which need no filter:

| Function | What it gives |
|---|---|
| `hohmann_transfer` | the cheapest two-burn transfer between two circular orbits |
| `lambert` | the orbit that joins two points in a given time |
| `porkchop` | the cost of every pair of departure and arrival dates |
| `chain_lambert` | a trip through several flybys, with what each flyby gives for free and what must be burnt |

A flyby changes the direction of the spacecraft at no cost in fuel.
`chain_lambert` reports that free change apart from the burn the trip still
needs. With a deep-space manoeuvre placed between the two Venus flybys, the
Cassini trip comes out close to the real mission: the notebook compares the
two, quantity by quantity.

## How it is built

- **Time stepping.** One RK4 step per sample, or `n_sub` of them when the
  spacecraft passes close to a body and the pull changes quickly.
- **Where the planets are.** The Moon and Sun follow circles by default.
  With `use_ephemeris=True`, or an `epoch` for the interplanetary model, their
  positions come from the real ephemeris.
- **Course corrections.** With `control=True` the interplanetary model takes a
  small velocity change as its input. The notebook computes these corrections
  from the filter's estimate and shows what each one removes.
- **C++.** `TransLunarModel` and `InterplanetaryModel` have twins in
  `libKalmanApps` (`Kalman::Apps::Astro::TransLunar`, `::Interplanetary`), so
  `model.filter(z)` runs wholly in C++, about 10 times faster. The planets'
  positions the interplanetary model tabulates from the ephemeris are handed
  to the twin when it runs. A course correction (`control=True`) and an
  integrator of your own keep the Python step.

## Using it

```python
from kalman.apps import hohmann_transfer, chain_lambert

print(hohmann_transfer("earth", "mars"))     # cost, flight time, time between launch windows
```

Notebook: [08_SpaceScience](../../share/notebooks/08_SpaceScience.ipynb).
