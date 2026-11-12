# Computer vision

A detector finds an object in each frame of a video, with some jitter, and
sometimes misses it. The filter turns those detections into a smooth track and
carries it through the frames where nothing was found.

## The models

| Class | State | Used for |
|---|---|---|
| `DetectionModel` | the centre of the object and its velocity | one object |
| `BoxTrackModel` | the centre, area and shape of a box, and how the first three change | one box; the building block of SORT |
| `SortTracker` | one `BoxTrackModel` per object | many objects at once |

## How it is built

- **Constant velocity.** Between frames the object is assumed to keep moving as
  it was. The step is linear and exact.
- **Coasting.** A missed detection is passed as an empty measurement. The filter
  then predicts without correcting, and its uncertainty grows until the object
  is found again.
- **Many objects.** `SortTracker` matches each new detection to the track whose
  predicted box overlaps it most, starts a track for a detection that matches
  none, and ends a track that has gone unmatched for `max_age` frames.

## Using it

```python
from kalman.apps.vision import DetectionModel

ball = DetectionModel("ball", fs=25, frame=(320, 240),
                      measurement_uncertainty=4.0, process_uncertainty=60.0)
track = ball.filter(centroids)               # one (x, y) per frame, NaN where missed
```

Notebook: [10_ComputerVision](../../share/notebooks/10_ComputerVision.ipynb).
