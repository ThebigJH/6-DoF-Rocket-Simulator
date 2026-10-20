# 6-DoF Rocket Flight Dynamics & TVC Simulation

A reproducible, quaternion-based rigid-body flight simulation with variable
mass properties, an altitude-dependent atmosphere, stochastic crosswinds, and
a discrete two-axis PID thrust-vector controller.

## Run

Python 3.10 or newer is required.

```powershell
python -m pip install -r requirements.txt
python main.py
```

The run prints an ASCII flight-performance report, writes `flight_log.csv`,
and saves a high-resolution `flight_plots.png`. Use `python main.py --no-show`
to save the plots without opening a GUI, `--output-dir results` to choose an
output folder, or `--seed 42` to select a different reproducible gust history.

## Model scope

- `environment.py` evaluates the 1976 U.S. Standard Atmosphere layer equations
  through 86 km geometric altitude and seeded Gaussian crosswind pulses.
- `rocket.py` computes propellant depletion, a smooth thrust curve, shifting
  centre of mass, and diagonal inertia using cylindrical components and the
  parallel-axis theorem.
- `physics_engine.py` integrates three-axis translation and rotation with a
  normalized body-to-world quaternion and classical RK4. Drag varies with
  angle of attack; lift and force-at-centre-of-pressure moments are included.
- `controller.py` runs discrete pitch/yaw PID loops with conditional
  anti-windup, +/-5 degree gimbal hard stops, and a 30 degree/second servo rate.
- `telemetry_panel.py` exports synchronized SI telemetry, computes flight and
  control metrics, and produces the 3-D trajectory plus time-history plots.
- `main.py` coordinates the 100 Hz flight computer, integration, burnout, and
  ascent apogee detection.

This is an educational engineering model, not a flight-certified vehicle
design or a substitute for validated propulsion, structural, or aerodynamic
data. Gravity is locally constant and aerodynamic/structural flexibility,
Earth rotation, and fin effects are outside this model's scope.
