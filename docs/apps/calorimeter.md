# Calorimeter

A particle enters a block of dense material and deposits its energy in a
shower, layer after layer. The filter reconstructs where the particle went and
how much energy it had.

## The model

```
state        x = [x, y, slope in x, slope in y, energy]
dynamics     the position advances by the slope from one layer to the next
measurement  z = the centre of the energy deposit in a layer, and its amount
```

Three classes share this domain: a straight-line tracker with four states, and
an electromagnetic and a hadronic model that add the energy as a fifth.

## How it is built

- **The filter steps over depth, not over time.** One step is one layer of the
  calorimeter.
- **Shower shape.** The energy expected in a layer follows the known profile of
  a shower, which rises to a maximum and then dies away. The profile is measured
  from the front face of the calorimeter, not from the origin of the detector.
- **Materials.** `CalorimeterGeometry` builds a stack from real materials and
  gives its radiation length and Molière radius, the width of a shower.
- **Reconstruction.** `model.reconstruct(...)` runs the filter forward through
  the layers and back, and returns the smoothed track and energy.

## Using it

A calorimeter needs a geometry and simulated showers, so the shortest complete
example is the notebook.

Notebook: [06_Calorimeter](../../share/notebooks/06_Calorimeter.ipynb).
