"""Six-degree-of-freedom rigid-body flight dynamics and RK4 integration."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray

from environment import Environment
from rocket import Rocket


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AerodynamicState:
    """Aerodynamic force and flow quantities at the rocket centre of mass.

    Attributes:
        drag_force_world_N: Drag vector in newtons, world frame.
        lift_force_world_N: Lift vector in newtons, world frame.
        angle_of_attack_rad: Angle between body axis and relative airflow.
        dynamic_pressure_Pa: Local dynamic pressure in pascals.
        relative_air_velocity_mps: Velocity relative to the surrounding air.
    """

    drag_force_world_N: FloatArray
    lift_force_world_N: FloatArray
    angle_of_attack_rad: float
    dynamic_pressure_Pa: float
    relative_air_velocity_mps: FloatArray


@dataclass(frozen=True)
class FlightForces:
    """Instantaneous force and moment breakdown.

    Attributes:
        thrust_force_world_N: World-frame engine force.
        drag_force_world_N: World-frame aerodynamic drag.
        lift_force_world_N: World-frame aerodynamic lift.
        gravity_force_world_N: World-frame gravitational force.
        total_force_world_N: Vector sum of external forces.
        total_moment_body_Nm: Body-frame moment about the centre of mass.
        angle_of_attack_rad: Angle of attack in radians.
        dynamic_pressure_Pa: Dynamic pressure in pascals.
        wind_velocity_world_mps: World-frame wind vector.
        gust_active: Whether the rocket is within a configured gust profile.
        thrust_N: Engine thrust magnitude in newtons.
        mass_kg: Current total rocket mass in kilograms.
    """

    thrust_force_world_N: FloatArray
    drag_force_world_N: FloatArray
    lift_force_world_N: FloatArray
    gravity_force_world_N: FloatArray
    total_force_world_N: FloatArray
    total_moment_body_Nm: FloatArray
    angle_of_attack_rad: float
    dynamic_pressure_Pa: float
    wind_velocity_world_mps: FloatArray
    gust_active: bool
    thrust_N: float
    mass_kg: float


class PhysicsEngine:
    """Integrate translation, quaternion attitude, and body angular rates.

    State vector layout is ``[r(3), v(3), q_wxyz(4), omega_body(3)]``.
    Position and velocity are in the local world frame; angular rate is in the
    body frame. The local gravity vector is constant and points along -Z.
    """

    STATE_SIZE = 13
    POSITION = slice(0, 3)
    VELOCITY = slice(3, 6)
    QUATERNION = slice(6, 10)
    ANGULAR_RATE = slice(10, 13)
    GRAVITY_MPS2 = 9.80665
    MIN_AIRSPEED_FOR_AOA_MPS = 5.0
    NOZZLE_Z_FROM_COM_SIGN = -1.0
    ANGULAR_DAMPING_NMS = 0.035

    def __init__(self, rocket: Rocket, environment: Environment) -> None:
        """Connect a vehicle model and atmosphere to the dynamics solver.

        Args:
            rocket: Variable-mass vehicle and propulsion model.
            environment: Standard atmosphere and crosswind model.
        """
        self.rocket = rocket
        self.environment = environment
        self.reference_area_m2 = rocket.configuration.reference_area_m2
        self._last_forces: FlightForces | None = None

    @staticmethod
    def initial_state(position_world_m: FloatArray | None = None) -> FloatArray:
        """Create a stationary, upright 13-element flight state.

        Args:
            position_world_m: Initial position; defaults to 0.1 m above ground.

        Returns:
            State vector ``[position, velocity, quaternion, angular rate]``.
        """
        position_m = (
            np.array([0.0, 0.0, 0.1], dtype=float)
            if position_world_m is None
            else np.asarray(position_world_m, dtype=float)
        )
        if position_m.shape != (3,):
            raise ValueError("Initial position must have exactly three components.")
        state = np.zeros(PhysicsEngine.STATE_SIZE, dtype=float)
        state[PhysicsEngine.POSITION] = position_m
        state[PhysicsEngine.QUATERNION] = np.array([1.0, 0.0, 0.0, 0.0])
        return state

    @staticmethod
    def quaternion_to_rotation_matrix(quaternion_wxyz: FloatArray) -> FloatArray:
        """Return the body-to-world rotation matrix for a quaternion.

        Args:
            quaternion_wxyz: Quaternion in scalar-first ``[w, x, y, z]`` order.

        Returns:
            A 3-by-3 orthonormal body-to-world rotation matrix.

        Raises:
            ValueError: If the input is not a four-vector or has zero norm.
        """
        quaternion = np.asarray(quaternion_wxyz, dtype=float)
        if quaternion.shape != (4,):
            raise ValueError("Attitude quaternion must have four components.")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("Attitude quaternion must have non-zero norm.")
        w, x, y, z = quaternion / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
                 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
                 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
                 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=float,
        )

    @staticmethod
    def _quaternion_product(left_wxyz: FloatArray, right_wxyz: FloatArray) -> FloatArray:
        """Multiply scalar-first quaternions.

        Args:
            left_wxyz: Left quaternion.
            right_wxyz: Right quaternion.

        Returns:
            Quaternion Hamilton product ``left * right``.
        """
        left_w, left_x, left_y, left_z = left_wxyz
        right_w, right_x, right_y, right_z = right_wxyz
        return np.array(
            [
                left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
                left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
                left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
                left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
            ],
            dtype=float,
        )

    @staticmethod
    def _gimbal_direction_body(pitch_gimbal_rad: float, yaw_gimbal_rad: float) -> FloatArray:
        """Compute the unit thrust direction from two nozzle deflections.

        Args:
            pitch_gimbal_rad: Deflection toward body +X, in radians.
            yaw_gimbal_rad: Deflection toward body +Y, in radians.

        Returns:
            Unit engine-force vector in body coordinates.
        """
        return np.array(
            [
                math.sin(pitch_gimbal_rad) * math.cos(yaw_gimbal_rad),
                math.sin(yaw_gimbal_rad),
                math.cos(pitch_gimbal_rad) * math.cos(yaw_gimbal_rad),
            ],
            dtype=float,
        )

    def aerodynamic_state(
        self, state: FloatArray, time_s: float
    ) -> AerodynamicState:
        """Evaluate AoA-dependent drag and lift in the local wind field.

        Args:
            state: Current 13-element flight state.
            time_s: Simulation time in seconds.

        Returns:
            Aerodynamic forces and relative-flow data.
        """
        position_world_m = state[self.POSITION]
        velocity_world_mps = state[self.VELOCITY]
        rotation_body_to_world = self.quaternion_to_rotation_matrix(
            state[self.QUATERNION]
        )
        altitude_m = float(position_world_m[2])
        wind_velocity_mps, _ = self.environment.wind_velocity_mps(altitude_m)
        relative_velocity_mps = velocity_world_mps - wind_velocity_mps
        airspeed_mps = float(np.linalg.norm(relative_velocity_mps))
        if airspeed_mps < 1e-9:
            zero_force_N = np.zeros(3, dtype=float)
            return AerodynamicState(
                drag_force_world_N=zero_force_N,
                lift_force_world_N=zero_force_N.copy(),
                angle_of_attack_rad=0.0,
                dynamic_pressure_Pa=0.0,
                relative_air_velocity_mps=relative_velocity_mps,
            )

        atmosphere = self.environment.state(altitude_m, velocity_world_mps)
        dynamic_pressure_Pa = 0.5 * atmosphere.density_kgm3 * airspeed_mps**2
        velocity_hat = relative_velocity_mps / airspeed_mps
        body_axis_world = rotation_body_to_world @ np.array([0.0, 0.0, 1.0])
        axis_dot_velocity = float(np.clip(np.dot(body_axis_world, velocity_hat), -1.0, 1.0))
        perpendicular_axis = body_axis_world - axis_dot_velocity * velocity_hat
        sin_alpha = float(np.linalg.norm(perpendicular_axis))
        if airspeed_mps < self.MIN_AIRSPEED_FOR_AOA_MPS:
            # At launch, AoA is ill-conditioned because the vehicle has barely
            # acquired a meaningful air-relative flight direction.
            sin_alpha = 0.0
            angle_of_attack_rad = 0.0
        else:
            angle_of_attack_rad = math.atan2(sin_alpha, axis_dot_velocity)

        config = self.rocket.configuration
        drag_coefficient = (
            config.drag_coefficient_0
            + config.drag_alpha_gain * math.sin(angle_of_attack_rad) ** 2
        )
        drag_force_world_N = (
            -dynamic_pressure_Pa
            * self.reference_area_m2
            * drag_coefficient
            * velocity_hat
        )
        if sin_alpha < 1e-10:
            lift_force_world_N = np.zeros(3, dtype=float)
        else:
            lift_direction_world = perpendicular_axis / sin_alpha
            lift_coefficient = (
                config.lift_slope_per_rad
                * math.sin(angle_of_attack_rad)
                * math.cos(angle_of_attack_rad)
            )
            lift_force_world_N = (
                dynamic_pressure_Pa
                * self.reference_area_m2
                * lift_coefficient
                * lift_direction_world
            )
        return AerodynamicState(
            drag_force_world_N=drag_force_world_N,
            lift_force_world_N=lift_force_world_N,
            angle_of_attack_rad=angle_of_attack_rad,
            dynamic_pressure_Pa=dynamic_pressure_Pa,
            relative_air_velocity_mps=relative_velocity_mps,
        )

    def forces_and_moments(
        self,
        time_s: float,
        state: FloatArray,
        pitch_gimbal_rad: float,
        yaw_gimbal_rad: float,
    ) -> FlightForces:
        """Sum thrust, aerodynamic, and gravity forces and body moments.

        Thrust acts at the nozzle plane and aerodynamic forces act at the
        centre of pressure. Gravity acts through the centre of mass and
        therefore contributes no rotational moment.

        Args:
            time_s: Time since ignition in seconds.
            state: Current 13-element state vector.
            pitch_gimbal_rad: Actual pitch-channel nozzle angle.
            yaw_gimbal_rad: Actual yaw-channel nozzle angle.

        Returns:
            Complete force and moment breakdown in SI units.
        """
        if state.shape != (self.STATE_SIZE,):
            raise ValueError(f"State must have shape ({self.STATE_SIZE},).")
        mass_properties = self.rocket.mass_properties(time_s)
        rotation_body_to_world = self.quaternion_to_rotation_matrix(
            state[self.QUATERNION]
        )
        thrust_N = self.rocket.thrust_N_at_time(time_s)
        thrust_direction_body = self._gimbal_direction_body(
            pitch_gimbal_rad, yaw_gimbal_rad
        )
        thrust_force_body_N = thrust_N * thrust_direction_body
        thrust_force_world_N = rotation_body_to_world @ thrust_force_body_N

        aero = self.aerodynamic_state(state, time_s)
        aerodynamic_force_world_N = (
            aero.drag_force_world_N + aero.lift_force_world_N
        )
        aerodynamic_force_body_N = (
            rotation_body_to_world.T @ aerodynamic_force_world_N
        )
        center_of_pressure_offset_m = (
            self.rocket.configuration.center_of_pressure_from_nozzle_m
            - mass_properties.center_of_mass_from_nozzle_m
        )
        aerodynamic_moment_body_Nm = np.cross(
            np.array([0.0, 0.0, center_of_pressure_offset_m]),
            aerodynamic_force_body_N,
        )
        nozzle_offset_body_m = np.array(
            [
                0.0,
                0.0,
                self.NOZZLE_Z_FROM_COM_SIGN
                * mass_properties.center_of_mass_from_nozzle_m,
            ]
        )
        thrust_moment_body_Nm = np.cross(
            nozzle_offset_body_m, thrust_force_body_N
        )
        angular_rate_body_radps = state[self.ANGULAR_RATE]
        damping_moment_body_Nm = (
            -self.ANGULAR_DAMPING_NMS * angular_rate_body_radps
        )
        total_moment_body_Nm = (
            aerodynamic_moment_body_Nm
            + thrust_moment_body_Nm
            + damping_moment_body_Nm
        )
        gravity_force_world_N = np.array(
            [0.0, 0.0, -mass_properties.mass_kg * self.GRAVITY_MPS2],
            dtype=float,
        )
        total_force_world_N = (
            thrust_force_world_N
            + aero.drag_force_world_N
            + aero.lift_force_world_N
            + gravity_force_world_N
        )
        wind_velocity_mps, gust_active = self.environment.wind_velocity_mps(
            float(state[self.POSITION][2])
        )
        return FlightForces(
            thrust_force_world_N=thrust_force_world_N,
            drag_force_world_N=aero.drag_force_world_N,
            lift_force_world_N=aero.lift_force_world_N,
            gravity_force_world_N=gravity_force_world_N,
            total_force_world_N=total_force_world_N,
            total_moment_body_Nm=total_moment_body_Nm,
            angle_of_attack_rad=aero.angle_of_attack_rad,
            dynamic_pressure_Pa=aero.dynamic_pressure_Pa,
            wind_velocity_world_mps=wind_velocity_mps,
            gust_active=gust_active,
            thrust_N=thrust_N,
            mass_kg=mass_properties.mass_kg,
        )

    def _derivative(
        self,
        time_s: float,
        state: FloatArray,
        pitch_gimbal_rad: float,
        yaw_gimbal_rad: float,
    ) -> FloatArray:
        """Evaluate the coupled translational and rotational state derivative.

        Args:
            time_s: Evaluation time in seconds.
            state: State vector at the evaluation point.
            pitch_gimbal_rad: Held pitch actuator angle.
            yaw_gimbal_rad: Held yaw actuator angle.

        Returns:
            Derivative of all 13 state elements.
        """
        forces = self.forces_and_moments(
            time_s, state, pitch_gimbal_rad, yaw_gimbal_rad
        )
        mass_properties = self.rocket.mass_properties(time_s)
        acceleration_world_mps2 = (
            forces.total_force_world_N / mass_properties.mass_kg
        )

        quaternion_wxyz = state[self.QUATERNION]
        angular_rate_quaternion = np.concatenate(
            ([0.0], state[self.ANGULAR_RATE])
        )
        quaternion_derivative = 0.5 * self._quaternion_product(
            quaternion_wxyz, angular_rate_quaternion
        )

        inertia_kgm2 = mass_properties.inertia_kgm2
        inertia_rate_kgm2ps = self._inertia_rate_kgm2ps(time_s)
        angular_momentum_body = inertia_kgm2 @ state[self.ANGULAR_RATE]
        angular_acceleration_body_radps2 = np.linalg.solve(
            inertia_kgm2,
            forces.total_moment_body_Nm
            - np.cross(state[self.ANGULAR_RATE], angular_momentum_body)
            - inertia_rate_kgm2ps @ state[self.ANGULAR_RATE],
        )

        derivative = np.zeros(self.STATE_SIZE, dtype=float)
        derivative[self.POSITION] = state[self.VELOCITY]
        derivative[self.VELOCITY] = acceleration_world_mps2
        derivative[self.QUATERNION] = quaternion_derivative
        derivative[self.ANGULAR_RATE] = angular_acceleration_body_radps2
        self._last_forces = forces
        return derivative

    def _inertia_rate_kgm2ps(self, time_s: float) -> FloatArray:
        """Estimate the diagonal inertia derivative during propellant burn.

        Args:
            time_s: Evaluation time in seconds.

        Returns:
            Body-axis inertia derivative in kg m^2/s.
        """
        step_s = 1e-4
        time_before_s = max(0.0, time_s - step_s)
        time_after_s = min(self.rocket.burn_time_s, time_s + step_s)
        if time_after_s <= time_before_s:
            return np.zeros((3, 3), dtype=float)
        inertia_before = self.rocket.mass_properties(time_before_s).inertia_kgm2
        inertia_after = self.rocket.mass_properties(time_after_s).inertia_kgm2
        return (inertia_after - inertia_before) / (time_after_s - time_before_s)

    def step(
        self,
        time_s: float,
        state: FloatArray,
        delta_time_s: float,
        pitch_gimbal_rad: float,
        yaw_gimbal_rad: float,
    ) -> FloatArray:
        """Advance the flight state one classical fourth-order RK step.

        Actuator positions are held constant for the integration interval.
        The quaternion is normalized after the RK4 update to prevent numerical
        drift away from the unit sphere.

        Args:
            time_s: Current simulation time in seconds.
            state: Current 13-element state vector.
            delta_time_s: Positive integration interval in seconds.
            pitch_gimbal_rad: Held pitch-channel angle in radians.
            yaw_gimbal_rad: Held yaw-channel angle in radians.

        Returns:
            New 13-element state vector.

        Raises:
            ValueError: If the timestep is non-positive or state has wrong size.
        """
        if delta_time_s <= 0.0:
            raise ValueError("Integration interval must be positive.")
        if state.shape != (self.STATE_SIZE,):
            raise ValueError(f"State must have shape ({self.STATE_SIZE},).")
        derivative = lambda stage_time_s, stage_state: self._derivative(
            stage_time_s, stage_state, pitch_gimbal_rad, yaw_gimbal_rad
        )
        k1 = derivative(time_s, state)
        k2 = derivative(time_s + 0.5 * delta_time_s, state + 0.5 * delta_time_s * k1)
        k3 = derivative(time_s + 0.5 * delta_time_s, state + 0.5 * delta_time_s * k2)
        k4 = derivative(time_s + delta_time_s, state + delta_time_s * k3)
        next_state = state + delta_time_s * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        quaternion = next_state[self.QUATERNION]
        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-12 or not np.all(np.isfinite(next_state)):
            raise FloatingPointError("Flight integration produced a non-finite state.")
        next_state[self.QUATERNION] = quaternion / quaternion_norm
        return next_state

    def current_forces(
        self,
        time_s: float,
        state: FloatArray,
        pitch_gimbal_rad: float,
        yaw_gimbal_rad: float,
    ) -> FlightForces:
        """Return forces evaluated at the requested state and time.

        Args:
            time_s: Simulation time in seconds.
            state: Current 13-element state vector.
            pitch_gimbal_rad: Actual pitch-channel deflection.
            yaw_gimbal_rad: Actual yaw-channel deflection.

        Returns:
            Instantaneous force and moment breakdown.
        """
        return self.forces_and_moments(
            time_s, state, pitch_gimbal_rad, yaw_gimbal_rad
        )
