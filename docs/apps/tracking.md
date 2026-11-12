# Particle tracking

A charged particle crosses a detector made of concentric layers, in a magnetic
field that bends its path into a circle. Each layer records where particles
crossed it, along with false hits. The task is to find which hits belong
together and to fit a track through them.

## The model

```
state        x = [position x, position y, momentum x, momentum y]
dynamics     the momentum turns at the rate  0.3 q B / p      (field B in tesla, p in GeV/c)
measurement  z = [position x, position y] on a layer
```

The defaults are those of a real drift chamber: a field of 1.5 T, eight layers
from 20 to 104 cm, and hits measured to 130 µm.

## How it is built

- **The filter steps over layers, not over time.** One step carries the state
  from one layer to the next. The model's "time" argument is the radius of the
  layer to reach.
- **Time stepping.** The path is integrated along its length in small RK4
  steps, written in the model, and the last step is cut so that it ends exactly
  on the layer.
- **Finding the hits.** The combinatorial engine starts a candidate from each
  inner hit, predicts where it crosses the next layer, and keeps the hits that
  fall close enough to the prediction. A candidate with several plausible hits
  branches. Candidates are scored, the weak ones are dropped, and two tracks may
  not share a hit.
- **Refit.** Each track found is smoothed from the outside in, which sharpens
  the momentum at the inner layers.
- **C++.** The combinatorial engine exists in C++ too (`Engines::Combinatorial`).

## Using it

```python
from kalman.apps import DriftChamberModel
from kalman.framework.filters import CombinatorialKalmanFilter

chamber = DriftChamberModel(n_layers=8, Bz=1.5)
event = chamber.generate_event(n_tracks=3, n_background=4, seed=0)
tracks = CombinatorialKalmanFilter(chamber).forward(event.hits, layer_radii=event.radii)
```

Notebook: [07_CombinatorialTracking](../../share/notebooks/07_CombinatorialTracking.ipynb).
