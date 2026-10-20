"""Run the deterministic 6-DoF rocket launch-to-apogee simulation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from controller import FlightController
from environment import Environment
from physics_engine import PhysicsEngine
from rocket import Rocket
from telemetry_panel import (
    FlightRecord,
    export_csv,
    make_flight_record,
    plot_flight,
    print_flight_report,
)


def run_simulation(
    time_step_s: float = 0.01,
    maximum_flight_time_s: float = 120.0,
    random_seed: int = 7,
) -> tuple[list[FlightRecord], Rocket]:
    """Run the discrete flight computer and continuous 6-DoF vehicle model.

    The controller samples at 100 Hz and holds each actuator position through
    one RK4 interval. A step is shortened at exact engine burnout and the
    ascent is terminated at the interpolated vertical-velocity zero crossing.

    Args:
        time_step_s: Nominal fixed integration/controller interval in seconds.
        maximum_flight_time_s: Safety timeout for the simulation.
        random_seed: Seed for reproducible wind gust realization.

    Returns:
        Time-ordered telemetry and the rocket model used for the run.

    Raises:
        ValueError: If integration interval or flight timeout is invalid.
        RuntimeError: If the vehicle fails to reach apogee before timeout.
    """
    if time_step_s <= 0.0 or maximum_flight_time_s <= 0.0:
        raise ValueError("Simulation timestep and timeout must be positive.")
    rocket = Rocket()
    environment = Environment(random_seed=random_seed)
    physics = PhysicsEngine(rocket, environment)
    controller = FlightController(sample_rate_hz=1.0 / time_step_s)
    state = physics.initial_state()
    time_s = 0.0
    controller_output = controller.update(state[PhysicsEngine.QUATERNION], time_step_s)
    records = [
        make_flight_record(
            time_s,
            state,
            rocket,
            physics,
            controller_output.pitch_error_rad,
            controller_output.yaw_error_rad,
            controller_output.pitch_gimbal_rad,
            controller_output.yaw_gimbal_rad,
        )
    ]
    apogee_reached = False

    while time_s < maximum_flight_time_s:
        delta_time_s = min(time_step_s, maximum_flight_time_s - time_s)
        if time_s < rocket.burn_time_s:
            delta_time_s = min(delta_time_s, rocket.burn_time_s - time_s)
        if delta_time_s <= 1e-12:
            time_s = rocket.burn_time_s
            continue
        previous_time_s = time_s
        previous_state = state
        state = physics.step(
            time_s,
            state,
            delta_time_s,
            controller_output.pitch_gimbal_rad,
            controller_output.yaw_gimbal_rad,
        )
        time_s += delta_time_s
        controller_output = controller.update(
            state[PhysicsEngine.QUATERNION], delta_time_s
        )

        crossed_apogee = (
            previous_state[PhysicsEngine.VELOCITY][2] > 0.0
            and state[PhysicsEngine.VELOCITY][2] <= 0.0
            and time_s > 0.1
        )
        if crossed_apogee:
            vertical_velocity_before_mps = previous_state[
                PhysicsEngine.VELOCITY
            ][2]
            vertical_velocity_after_mps = state[PhysicsEngine.VELOCITY][2]
            interpolation_fraction = vertical_velocity_before_mps / (
                vertical_velocity_before_mps - vertical_velocity_after_mps
            )
            time_s = previous_time_s + interpolation_fraction * delta_time_s
            state = previous_state + interpolation_fraction * (state - previous_state)
            quaternion = state[PhysicsEngine.QUATERNION]
            quaternion_norm = float(np.linalg.norm(quaternion))
            state[PhysicsEngine.QUATERNION] = quaternion / quaternion_norm
            state[PhysicsEngine.VELOCITY][2] = 0.0
            apogee_reached = True

        records.append(
            make_flight_record(
                time_s,
                state,
                rocket,
                physics,
                controller_output.pitch_error_rad,
                controller_output.yaw_error_rad,
                controller_output.pitch_gimbal_rad,
                controller_output.yaw_gimbal_rad,
            )
        )
        if apogee_reached:
            break

    if not apogee_reached:
        raise RuntimeError(
            f"Apogee was not reached within {maximum_flight_time_s:.1f} seconds."
        )
    return records, rocket


def main() -> None:
    """Parse output options, run the flight, and emit reports and plots."""
    parser = argparse.ArgumentParser(
        description="Simulate a quaternion-based 6-DoF rocket with active TVC."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for flight_log.csv and flight_plots.png.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plots without opening an interactive window.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the stochastic crosswind gusts.",
    )
    args = parser.parse_args()

    flight_records, rocket = run_simulation(random_seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_flight_report(flight_records, rocket)
    csv_path = export_csv(flight_records, args.output_dir / "flight_log.csv")
    plot_path = plot_flight(
        flight_records,
        args.output_dir / "flight_plots.png",
        show=not args.no_show,
    )
    print(f"Flight log: {csv_path}")
    print(f"Flight plots: {plot_path}")


if __name__ == "__main__":
    main()
