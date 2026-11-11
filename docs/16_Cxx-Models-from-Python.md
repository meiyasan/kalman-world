# C++ models from Python

This page is for someone who wants their filter to run fast and wants to know
what to write, and where. It answers three questions:

1. What runs in C++ today, and what runs in Python?
2. How do I write a model once in C++ and use it from both languages?
3. How do I check that the C++ path is really being used?

The worked example is a toy: a cup of coffee cooling in a room, read by a
thermometer. The same compiled model is run by a C++ program and by a Python
script, on the same sensor readings.

## What runs where

Python is the interface. You build models, run filters, read records and draw
plots in Python. The arithmetic underneath runs in C++ whenever the library is
installed, and in numpy when it is not. Nothing in your code changes between
the two.

| Piece | Where it runs | When |
|---|---|---|
| Filter workflow, records, plots, notebooks, the JSON API | Python | always |
| The Kalman update, the smoother, sigma-point and particle sums | C++ (`libKalmanFramework`) | when the library is found; numpy otherwise |
| A linear model with fixed matrices | C++, the whole pass in one call | when the library is found |
| A nonlinear model written in Python | Python computes `f` and `h` at every step | always |
| A nonlinear model written in C++ | C++, the whole pass in one call | when its library is loaded |
| The four application models with a C++ twin | C++, the whole pass in one call | when `libKalmanApps` is found |

The rule behind the table: crossing from Python into C++ is cheap, and crossing
back from C++ into Python at every step is not. So a model whose physics is
Python keeps the Python loop, and a model whose physics is C++ runs entirely in
C++.

## Install and check

```bash
make build                      # compiles libKalmanFramework and libKalmanApps
cmake --install var/build       # copies them to var/install/lib
python -c "import kalman.framework.native as n; print(n.HAS_C_ACCEL)"
```

`True` means Python found the library. It looks, in this order, at the file
named by `KALMAN_FRAMEWORK_LIBRARY`, the `lib` folder under `KALMANFRAMEWORK_ROOT`
or `KALMAN_ROOT`, this checkout (`var/install/lib`, then `var/build/lib`), and
finally the system loader.

`False` is not an error. Everything still works, on numpy.

## Three ways to write a model

| You want | Write | Speed of a nonlinear model |
|---|---|---|
| To try an idea quickly | a Python class (see [Models](4_Models.md)) | the Python loop |
| One model for C++ programs, ROOT+ and Python | a C++ class, registered by name | the whole pass in C++ |
| An existing Python class to run faster, unchanged for its users | a C++ twin behind the Python class | the whole pass in C++ |

Measured on this repository's models (2000 steps, `tools/benchmark_twins.py`):
the C++ pass is 9 to 11 times faster for the storm and tornado models, 19 times
for the car, and 33 to 42 times for the ballistic model. The results agree with
the Python ones to about one part in a hundred million.

## The toy: one model, two languages

The files:

| File | What it is |
|---|---|
| `examples/cxx/toy_plugin.cc` | the model, written once, built as `libKalmanPlugin_toy` |
| `examples/cxx/toy_sensor.cc` | a C++ program: simulates the sensor, filters, writes CSV |
| `examples/python/toy_sensor.py` | a Python script: the same model on the same readings |

Run it:

```bash
make build && make examples
./var/build/bin/toy_sensor                  # writes toy_sensor.csv and toy_estimate.csv
python examples/python/toy_sensor.py --plot # reads them, compares, saves toy_sensor.png
```

### The physics

Coffee cools toward the room temperature at a rate nobody measured. The filter
has to learn that rate from the thermometer alone, so the rate is part of the
state:

```
state        x = [T, k]      temperature [degC], cooling rate [1/s]
dynamics     T' = -k (T - T_room),   k' = 0
measurement  z = T           the thermometer
```

The product of `k` and `T` makes the model nonlinear, so it uses the extended
Kalman filter.

### The model, in C++

A model is a class with a step function `f` and a measurement function `h`:

```cpp
class CoolingCoffee : public Kalman::Models::Extended
{
    double fRoom = 21.;

public:
    CoolingCoffee() : Kalman::Models::Extended("toy-coffee", 2, 1)   // 2 states, 1 measurement
    {
        DeclareParameter("room", 21.);     // a setting others can change by name
        ParametersChanged();
        SetSampleRate(1.);                 // one reading a second
        // ... Q, R, the initial state and its covariance
    }

    Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &, double) const override
    {
        Kalman::Vector out(2);
        out(0) = fRoom + (x(0) - fRoom) * std::exp(-x(1) * DeltaT());   // exact cooling over one step
        out(1) = x(1);                                                  // the rate does not change
        return out;
    }

    Kalman::Vector h(const Kalman::Vector &x, double) const override
    {
        Kalman::Vector z(1);
        z(0) = x(0);                       // the thermometer reads the temperature
        return z;
    }

protected:
    void ParametersChanged() override { fRoom = Parameter("room"); }
};

KALMAN_REGISTER_MODEL(CoolingCoffee, "toy-coffee");   // this line makes it findable by name
```

The last line is the whole integration. Any program that loads the library can
now create the model by the name `toy-coffee` without seeing the class.

### Using it from C++

```cpp
Kalman::Factory &factory = Kalman::Factory::Instance();
factory.LoadPlugin("libKalmanPlugin_toy");
std::unique_ptr<Kalman::Model> coffee = factory.Create("toy-coffee");
coffee->SetParameter("room", 21.);

Kalman::Sampler sensor(coffee.get(), 2026);       // the test sensor, with a fixed seed
sensor.Generate(1800, truth0);                    // 30 minutes of readings

Kalman::Filters::EKF ekf(coffee.get());
Kalman::Record estimate = ekf.Forward(sensor.Measurements());
```

### Using it from Python

```python
from kalman.framework.native import core
from kalman.framework.models.native import NativeModel

core.load_plugin(core.resolve_plugin("KalmanPlugin_toy"))
coffee = NativeModel("toy-coffee", "thermometer")
coffee.set_parameter("room", 21.0)
print(coffee.parameters())                # {'room': 21.0}

record = coffee.filter(readings)          # Python workflow, C++ physics
record = coffee.filter_native(readings)   # the whole pass in C++
coffee.plot(record)
```

`coffee` is an ordinary model here. `filter`, `smooth`, `plot`, the diagnostics
and the JSON API all work on it.

### What you should see

```
sensor  : 1800 readings, thermometer error 0.486 degC (RMS)
filter  : temperature error 0.080 degC (RMS)
learned : k = 0.001964 +/- 0.000035 /s   (truth 0.001977)
agreement with the C++ program (largest difference, relative to the largest value):
    Python filter, C++ model : T 0.0e+00   k 1.1e-16
    fused C++ pass           : T 0.0e+00   k 0.0e+00
```

Three things to read from it. The filter is six times more accurate than the
thermometer. It learned the cooling rate to within its own error bar, starting
from a guess that was wrong by a factor of two. And the Python and C++ runs give
the same numbers, because they are the same compiled model.

## Named parameters

A model declares its settings once, with `DeclareParameter(name, default)`, and
refreshes anything derived from them in `ParametersChanged()`. From then on
every caller sets them by name:

| From | Read | Set |
|---|---|---|
| C++ | `model.Parameter("room")` | `model.SetParameter("room", 21.)` |
| Python | `model.parameters()` | `model.set_parameter("room", 21.0)` |
| C | `kalman_model_parameter(...)` | `kalman_model_set_parameter(...)` |

An unknown name is an error that lists the names the model does declare.
Parameters are numbers. A model whose size depends on a setting (two or three
states per axis, say) registers one name per size instead, as the ballistic
model does with `ballistic` and `ballistic-order2`.

## Giving an existing Python model a C++ twin

When a Python class already has users, keep it and put the C++ class behind it.
The four application models that do this are listed in
[the C++ library page](15_Cxx-Library.md), with the rules a twin must follow.
In short:

```python
from kalman.framework.models.native import NativeTwin

class StormModel(NativeTwin, ExtendedKalmanModel):

    def native_name(self):                       # which C++ model stands in for this one
        if self.steering is not None:            # a Python callback has no C++ counterpart
            return None
        return "storm"

    def native_parameters(self):                 # the settings to hand over, by name
        return {"omega": self.omega, "beta_drift": self.beta_drift}
```

`model.filter(z)` then runs in C++ when it can and in Python when it cannot.
The user of `StormModel` sees no difference except the time it takes.

To compare the two paths, switch the twins off:

```python
import kalman.framework.models.native as native
native.USE_NATIVE_TWINS = False        # or KALMAN_NATIVE_TWINS=0 in the environment
```

## When something goes wrong

| Symptom | Cause | What to do |
|---|---|---|
| `HAS_C_ACCEL` is `False` | the library was not built or not found | `make build`, `cmake --install var/build`, or set `KALMANFRAMEWORK_ROOT` |
| `plugin: ... does not exist` | the plugin was not built | `make examples`; plugins are looked up in `var/install/lib` and `var/build/lib`, or in `KALMAN_PLUGIN_PATH` |
| `loaded but registered no model` | the plugin was linked against another copy of the library | rebuild the plugin with the library it will run with |
| A crash right after changing `Kalman/Model.h` | a library or plugin still compiled against the old header | rebuild everything together: the library, `libKalmanApps`, every plugin, and ROOT+ |
| `import torch` fails with a symbol error | `DYLD_LIBRARY_PATH` points at another libtorch | `unset DYLD_LIBRARY_PATH` for this environment |
| A twin is not used | the configuration has no C++ counterpart | `model.native_handle()` returns `None`; see the table in [the C++ library page](15_Cxx-Library.md) |
