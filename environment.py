"""Atmospheric properties and reproducible, altitude-localized wind gusts."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AtmosphericState:
    """Atmosphere and wind conditions at one point in the flight.

    Attributes:
        temperature_K: Static air temperature in kelvin.
        pressure_Pa: Static air pressure in pascals.
        density_kgm3: Air density in kilograms per cubic metre.
        speed_of_sound_mps: Local speed of sound in metres per second.
        dynamic_pressure_Pa: Dynamic pressure relative to the local air mass.
        wind_velocity_mps: World-frame wind velocity in metres per second.
        gust_active: Whether a configured gust is locally significant.
    """

    temperature_K: float
    pressure_Pa: float
    density_kgm3: float
    speed_of_sound_mps: float
    dynamic_pressure_Pa: float
    wind_velocity_mps: FloatArray
    gust_active: bool


@dataclass(frozen=True)
class WindGust:
    """A seeded, stochastic crosswind pulse centred on a flight altitude.

    Attributes:
        altitude_m: Geometric altitude at the centre of the gust.
        sigma_altitude_m: Altitude standard deviation of the pulse.
        velocity_peak_mps: World-frame peak horizontal wind vector.
    """

    altitude_m: float
    sigma_altitude_m: float
    velocity_peak_mps: FloatArray


class Environment:
    """1976 U.S. Standard Atmosphere (0--86 km) and seeded crosswinds.

    The atmosphere uses geopotential altitude and the standard layer lapse
    rates through the mesosphere. Wind gust amplitudes are drawn once from a
    seeded random generator so repeated runs with the same seed are identical.
    """

    _EARTH_RADIUS_M = 6_356_766.0
    _G0_MPS2 = 9.80665
    _R_AIR_JKGK = 287.05287
    _GAMMA_AIR = 1.4
    _SEA_LEVEL_TEMPERATURE_K = 288.15
    _SEA_LEVEL_PRESSURE_PA = 101_325.0
    _LAYER_BASE_ALTITUDE_M = (0.0, 11_000.0, 20_000.0, 32_000.0,
                              47_000.0, 51_000.0, 71_000.0, 84_852.0)
    _LAYER_LAPSE_KPM = (-0.0065, 0.0, 0.0010, 0.0028,
                        0.0, -0.0028, -0.0020)

    def __init__(
        self,
        wind_gust_altitudes_m: Sequence[float] = (30.0, 80.0),
        random_seed: int = 7,
        mean_crosswind_mps: tuple[float, float] = (2.0, 0.0),
        gust_sigma_mps: float = 5.0,
        gust_width_m: float = 6.0,
    ) -> None:
        """Initialize standard-atmosphere layer bases and wind realizations.

        Args:
            wind_gust_altitudes_m: Altitude centres for gust pulses.
            random_seed: Seed controlling stochastic gust amplitudes.
            mean_crosswind_mps: Constant east/north mean wind.
            gust_sigma_mps: Standard deviation of each gust component.
            gust_width_m: Gaussian altitude standard deviation of each gust.

        Raises:
            ValueError: If a gust parameter is not physically meaningful.
        """
        if gust_sigma_mps < 0.0 or gust_width_m <= 0.0:
            raise ValueError("Gust amplitude must be non-negative and width positive.")
        if any(altitude_m < 0.0 for altitude_m in wind_gust_altitudes_m):
            raise ValueError("Gust altitudes must be non-negative.")
        self._mean_wind_mps = np.array(
            [mean_crosswind_mps[0], mean_crosswind_mps[1], 0.0], dtype=float
        )
        self.gusts: tuple[WindGust, ...] = self._make_gusts(
            wind_gust_altitudes_m, random_seed, gust_sigma_mps, gust_width_m
        )
        self._layer_base_temperature_K, self._layer_base_pressure_Pa = (
            self._build_layer_bases()
        )

    @classmethod
    def _build_layer_bases(cls) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Calculate temperature and pressure at every standard layer base.

        Returns:
            Paired immutable tuples of layer-base temperatures and pressures.
        """
        temperatures_K = [cls._SEA_LEVEL_TEMPERATURE_K]
        pressures_Pa = [cls._SEA_LEVEL_PRESSURE_PA]
        for index, lapse_Kpm in enumerate(cls._LAYER_LAPSE_KPM):
            delta_altitude_m = (
                cls._LAYER_BASE_ALTITUDE_M[index + 1]
                - cls._LAYER_BASE_ALTITUDE_M[index]
            )
            temperature_base_K = temperatures_K[index]
            temperature_next_K = temperature_base_K + lapse_Kpm * delta_altitude_m
            if lapse_Kpm == 0.0:
                pressure_next_Pa = pressures_Pa[index] * math.exp(
                    -cls._G0_MPS2 * delta_altitude_m
                    / (cls._R_AIR_JKGK * temperature_base_K)
                )
            else:
                pressure_next_Pa = pressures_Pa[index] * (
                    temperature_next_K / temperature_base_K
                ) ** (
                    -cls._G0_MPS2 / (cls._R_AIR_JKGK * lapse_Kpm)
                )
            temperatures_K.append(temperature_next_K)
            pressures_Pa.append(pressure_next_Pa)
        return tuple(temperatures_K), tuple(pressures_Pa)

    @staticmethod
    def _make_gusts(
        altitudes_m: Sequence[float],
        random_seed: int,
        sigma_mps: float,
        width_m: float,
    ) -> tuple[WindGust, ...]:
        """Create fixed stochastic gust vectors for repeatable simulations.

        Args:
            altitudes_m: Gust centre altitudes.
            random_seed: Seed for the random-number generator.
            sigma_mps: Standard deviation for each horizontal component.
            width_m: Gaussian altitude width for every pulse.

        Returns:
            Immutable gust definitions with zero vertical wind components.
        """
        generator = random.Random(random_seed)
        return tuple(
            WindGust(
                altitude_m=float(altitude_m),
                sigma_altitude_m=width_m,
                velocity_peak_mps=np.array(
                    [generator.gauss(0.0, sigma_mps),
                     generator.gauss(0.0, sigma_mps), 0.0],
                    dtype=float,
                ),
            )
            for altitude_m in altitudes_m
        )

    def _standard_atmosphere(
        self, altitude_m: float
    ) -> tuple[float, float, float, float]:
        """Evaluate temperature, pressure, density, and sound speed.

        Args:
            altitude_m: Geometric altitude above mean sea level in metres.

        Returns:
            Temperature (K), pressure (Pa), density (kg/m^3), and sound speed
            (m/s), evaluated at geopotential altitude.
        """
        altitude_clamped_m = max(0.0, min(float(altitude_m), 86_000.0))
        geopotential_altitude_m = (
            self._EARTH_RADIUS_M * altitude_clamped_m
            / (self._EARTH_RADIUS_M + altitude_clamped_m)
        )
        layer_index = max(
            index
            for index, base_altitude_m in enumerate(self._LAYER_BASE_ALTITUDE_M[:-1])
            if geopotential_altitude_m >= base_altitude_m
        )
        delta_altitude_m = (
            geopotential_altitude_m - self._LAYER_BASE_ALTITUDE_M[layer_index]
        )
        temperature_base_K = self._layer_base_temperature_K[layer_index]
        pressure_base_Pa = self._layer_base_pressure_Pa[layer_index]
        lapse_Kpm = self._LAYER_LAPSE_KPM[layer_index]
        temperature_K = temperature_base_K + lapse_Kpm * delta_altitude_m
        if lapse_Kpm == 0.0:
            pressure_Pa = pressure_base_Pa * math.exp(
                -self._G0_MPS2 * delta_altitude_m
                / (self._R_AIR_JKGK * temperature_base_K)
            )
        else:
            pressure_Pa = pressure_base_Pa * (
                temperature_K / temperature_base_K
            ) ** (-self._G0_MPS2 / (self._R_AIR_JKGK * lapse_Kpm))
        density_kgm3 = pressure_Pa / (self._R_AIR_JKGK * temperature_K)
        speed_of_sound_mps = math.sqrt(
            self._GAMMA_AIR * self._R_AIR_JKGK * temperature_K
        )
        return temperature_K, pressure_Pa, density_kgm3, speed_of_sound_mps

    def wind_velocity_mps(self, altitude_m: float) -> tuple[FloatArray, bool]:
        """Return local mean-plus-gust wind and gust activity.

        Args:
            altitude_m: Geometric altitude in metres.

        Returns:
            A world-frame wind vector and whether any gust has at least ten
            percent of its peak altitude-profile amplitude.
        """
        wind_velocity_mps = self._mean_wind_mps.copy()
        gust_active = False
        for gust in self.gusts:
            normalized_altitude = (
                altitude_m - gust.altitude_m
            ) / gust.sigma_altitude_m
            profile = math.exp(-0.5 * normalized_altitude**2)
            wind_velocity_mps += gust.velocity_peak_mps * profile
            gust_active = gust_active or profile >= 0.1
        return wind_velocity_mps, gust_active

    def state(
        self, altitude_m: float, velocity_world_mps: FloatArray
    ) -> AtmosphericState:
        """Calculate thermodynamic properties and dynamic pressure.

        Args:
            altitude_m: Geometric altitude in metres.
            velocity_world_mps: Rocket velocity in the world frame (m/s).

        Returns:
            An immutable snapshot containing local atmosphere and wind.
        """
        temperature_K, pressure_Pa, density_kgm3, speed_of_sound_mps = (
            self._standard_atmosphere(altitude_m)
        )
        wind_velocity_mps, gust_active = self.wind_velocity_mps(altitude_m)
        relative_velocity_mps = velocity_world_mps - wind_velocity_mps
        dynamic_pressure_Pa = (
            0.5 * density_kgm3 * float(np.dot(relative_velocity_mps, relative_velocity_mps))
        )
        return AtmosphericState(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            density_kgm3=density_kgm3,
            speed_of_sound_mps=speed_of_sound_mps,
            dynamic_pressure_Pa=dynamic_pressure_Pa,
            wind_velocity_mps=wind_velocity_mps,
            gust_active=gust_active,
        )
