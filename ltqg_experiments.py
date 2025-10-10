"""
Log-Time Quantum Gravity Experimental Design Module
===================================================

Author: Denzil James Greenwood
GitHub: https://github.com/DenzilGreenwood/log_time
License: MIT

This module implements the specific experimental protocols and predictions
described in the LTQG paper for testing the theory's falsifiable signatures.

Experimental protocols implemented:
1. σ-Uniform Zeno/Anti-Zeno protocols
2. Near-horizon interferometry (tabletop analogs)
3. Early-universe imprint calculations
4. Clock-transport loop experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import scipy.integrate as integrate
import scipy.optimize as optimize
from ltqg_core import LTQGSimulator, create_ltqg_simulator, LTQGConfig


@dataclass
class ExperimentalSetup:
    """Base class for experimental setup parameters."""
    duration: float  # Total experiment duration
    precision: float  # Measurement precision requirement
    environment: str  # Experimental environment description
    feasibility_score: float  # 0-1 feasibility rating


@dataclass
class ZenoExperiment(ExperimentalSetup):
    """Setup for σ-uniform Zeno effect experiments."""
    n_measurements: int  # Number of measurement interruptions
    redshift_factor: float  # Gravitational redshift α
    two_level_spacing: float  # Energy gap of quantum system
    decoherence_rate: float  # Environmental decoherence rate


@dataclass
class InterferometryExperiment(ExperimentalSetup):
    """Setup for analog gravity interferometry experiments."""
    path_length: float  # Interferometer arm length
    wavelength: float  # Probe wavelength
    redshift_gradient: float  # Spatial gradient of effective α
    beam_splitter_efficiency: float  # Optical efficiency


@dataclass
class CosmologyExperiment(ExperimentalSetup):
    """Setup for early-universe signature detection."""
    frequency_range: Tuple[float, float]  # Observation frequency range
    angular_resolution: float  # Angular resolution in radians
    sensitivity_limit: float  # Minimum detectable signal
    observation_time: float  # Total observation duration


class ExperimentalProtocol(ABC):
    """Abstract base class for LTQG experimental protocols."""
    
    def __init__(self, simulator: LTQGSimulator):
        self.simulator = simulator
    
    @abstractmethod
    def predict_signal(self, setup: ExperimentalSetup) -> Dict:
        """Predict the experimental signal for LTQG."""
        pass
    
    @abstractmethod
    def predict_standard_qm(self, setup: ExperimentalSetup) -> Dict:
        """Predict the signal for standard QM (null hypothesis)."""
        pass
    
    @abstractmethod
    def distinguishability(self, setup: ExperimentalSetup) -> float:
        """Calculate how distinguishable LTQG is from standard QM."""
        pass


class SigmaUniformZenoProtocol(ExperimentalProtocol):
    """
    Implementation of σ-uniform Zeno effect protocols.
    
    Tests the prediction that measurement interruptions at uniform σ intervals
    show different behavior than uniform τ intervals in gravitational fields.
    """
    
    def __init__(self, simulator: LTQGSimulator):
        super().__init__(simulator)
    
    def generate_measurement_schedule(self, setup: ZenoExperiment, 
                                    protocol: str = "sigma") -> np.ndarray:
        """
        Generate measurement times for different protocols.
        
        Args:
            setup: Experimental setup parameters
            protocol: "sigma" for σ-uniform, "tau" for τ-uniform
            
        Returns:
            Array of measurement times in proper time units
        """
        if protocol == "tau":
            # Standard τ-uniform schedule
            return np.linspace(0, setup.duration, setup.n_measurements)
        
        elif protocol == "sigma":
            # σ-uniform schedule requires geometric proper time intervals
            sigma_start = self.simulator.time_transform.sigma_from_tau(
                np.array([setup.duration / 1000]))[0]  # Start early
            sigma_end = self.simulator.time_transform.sigma_from_tau(
                np.array([setup.duration]))[0]
            
            sigma_points = np.linspace(sigma_start, sigma_end, setup.n_measurements)
            return self.simulator.time_transform.tau_from_sigma(sigma_points)
        
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
    
    def quantum_survival_probability(self, measurement_times: np.ndarray,
                                   setup: ZenoExperiment) -> float:
        """
        Calculate quantum survival probability for given measurement schedule.
        
        Uses the quantum Zeno effect formula with gravitational modifications.
        """
        # Time intervals between measurements
        intervals = np.diff(measurement_times)
        
        # Effective evolution rate depends on redshift for σ-uniform protocol
        if len(intervals) == 0:
            return 1.0
        
        # In strong redshift, σ-uniform measurements sample the generator
        # at large negative σ values, suppressing evolution
        effective_rate = setup.two_level_spacing * setup.redshift_factor
        
        # Survival probability with Zeno suppression
        survival_prob = 1.0
        for dt in intervals:
            # Probability of no transition during interval dt
            p_no_transition = np.exp(-effective_rate * dt)
            survival_prob *= p_no_transition
        
        # Include decoherence
        total_time = measurement_times[-1] - measurement_times[0]
        decoherence_factor = np.exp(-setup.decoherence_rate * total_time)
        
        return survival_prob * decoherence_factor
    
    def predict_signal(self, setup: ZenoExperiment) -> Dict:
        """Predict LTQG signal for σ-uniform Zeno protocol."""
        sigma_times = self.generate_measurement_schedule(setup, "sigma")
        survival_prob = self.quantum_survival_probability(sigma_times, setup)
        
        # LTQG predicts enhanced Zeno effect in strong redshift
        zeno_enhancement = self.simulator.protocols.zeno_suppression_factor(
            setup.redshift_factor, setup.n_measurements)
        
        return {
            'survival_probability': survival_prob,
            'measurement_times': sigma_times,
            'zeno_enhancement': zeno_enhancement,
            'protocol': 'sigma_uniform'
        }
    
    def predict_standard_qm(self, setup: ZenoExperiment) -> Dict:
        """Predict standard QM signal (τ-uniform protocol)."""
        tau_times = self.generate_measurement_schedule(setup, "tau")
        survival_prob = self.quantum_survival_probability(tau_times, setup)
        
        return {
            'survival_probability': survival_prob,
            'measurement_times': tau_times,
            'zeno_enhancement': 1.0,  # No gravitational enhancement
            'protocol': 'tau_uniform'
        }
    
    def distinguishability(self, setup: ZenoExperiment) -> float:
        """Calculate distinguishability between LTQG and standard QM."""
        ltqg_pred = self.predict_signal(setup)
        qm_pred = self.predict_standard_qm(setup)
        
        # Signal difference relative to measurement precision
        signal_diff = abs(ltqg_pred['survival_probability'] - 
                         qm_pred['survival_probability'])
        
        # Statistical significance (simplified)
        measurement_noise = setup.precision
        significance = signal_diff / measurement_noise
        
        return min(significance, 10.0)  # Cap at 10σ


class AnalogGravityInterferometry(ExperimentalProtocol):
    """
    Implementation of analog gravity interferometry experiments.
    
    Uses optical systems, BECs, or other analogs to simulate gravitational
    redshift and test σ-phase predictions.
    """
    
    def __init__(self, simulator: LTQGSimulator):
        super().__init__(simulator)
    
    def effective_redshift_profile(self, position: np.ndarray,
                                 setup: InterferometryExperiment) -> np.ndarray:
        """
        Generate effective redshift profile for analog system.
        
        Args:
            position: Spatial coordinates along interferometer path
            setup: Experimental setup parameters
            
        Returns:
            Effective redshift factor α(x) along path
        """
        # Example: linear gradient in effective gravitational potential
        # α(x) = 1 - gradient * x (approximation for weak fields)
        alpha_profile = 1.0 - setup.redshift_gradient * position
        return np.maximum(alpha_profile, 0.01)  # Avoid unphysical values
    
    def phase_accumulation_tau(self, path_positions: np.ndarray,
                             setup: InterferometryExperiment) -> float:
        """Calculate phase accumulation in standard (τ-uniform) approach."""
        alpha_profile = self.effective_redshift_profile(path_positions, setup)
        
        # Standard phase: φ = ∫ k(x) dx with multiplicative redshift
        path_length = path_positions[-1] - path_positions[0]
        wavenumber = 2 * np.pi / setup.wavelength
        
        # Geometric phase accumulation
        geometric_phase = wavenumber * path_length
        
        # Gravitational redshift affects frequency multiplicatively
        redshift_phase = wavenumber * np.trapz(alpha_profile - 1.0, path_positions)
        
        return geometric_phase + redshift_phase
    
    def phase_accumulation_sigma(self, path_positions: np.ndarray,
                               setup: InterferometryExperiment) -> float:
        """
        Calculate phase accumulation in LTQG (σ-uniform) approach.
        
        **CORRECTED**: Now includes proper measure transformation.
        
        When calculating phase along a spatial path with σ-dependent effects,
        we must account for the proper measure if time parameterization matters.
        """
        alpha_profile = self.effective_redshift_profile(path_positions, setup)
        
        # LTQG phase: additive σ-shifts
        sigma_shifts = self.simulator.redshift.sigma_shift_gravitational(alpha_profile)
        
        wavenumber = 2 * np.pi / setup.wavelength
        
        # For spatial path integration, the σ-dependence affects the 
        # effective interaction strength. If this represents a time-of-flight
        # measurement, we need the τ factor: dτ = τ dσ
        
        # Check if this is pure spatial (no time element) or spacetime path
        if hasattr(setup, 'include_time_evolution') and setup.include_time_evolution:
            # Include τ Jacobian for spacetime path
            tau_factors = self.simulator.time_transform.tau_from_sigma(sigma_shifts)
            weighted_sigma_shifts = sigma_shifts * tau_factors
            sigma_phase = wavenumber * np.trapz(weighted_sigma_shifts, path_positions)
        else:
            # Pure spatial path - direct σ integration is correct
            sigma_phase = wavenumber * np.trapz(sigma_shifts, path_positions)
        
        return sigma_phase
    
    def interferometric_visibility(self, phase_difference: float,
                                 setup: InterferometryExperiment) -> float:
        """Calculate interferometric fringe visibility."""
        # Visibility = |cos(Δφ/2)| with experimental imperfections
        ideal_visibility = np.abs(np.cos(phase_difference / 2.0))
        
        # Include beam splitter efficiency and other losses
        experimental_visibility = (ideal_visibility * 
                                 setup.beam_splitter_efficiency**2)
        
        return experimental_visibility
    
    def predict_signal(self, setup: InterferometryExperiment) -> Dict:
        """Predict LTQG interferometric signal."""
        # Two interferometer arms with different redshift exposure
        path_1 = np.linspace(0, setup.path_length, 1000)
        path_2 = np.linspace(0, setup.path_length, 1000)
        
        # Arm 2 experiences stronger effective redshift
        modified_setup = InterferometryExperiment(
            duration=setup.duration,
            precision=setup.precision,
            environment=setup.environment,
            feasibility_score=setup.feasibility_score,
            path_length=setup.path_length,
            wavelength=setup.wavelength,
            redshift_gradient=setup.redshift_gradient * 2.0,  # Stronger gradient
            beam_splitter_efficiency=setup.beam_splitter_efficiency
        )
        
        phase_1 = self.phase_accumulation_sigma(path_1, setup)
        phase_2 = self.phase_accumulation_sigma(path_2, modified_setup)
        
        phase_difference = phase_2 - phase_1
        visibility = self.interferometric_visibility(phase_difference, setup)
        
        return {
            'phase_difference': phase_difference,
            'visibility': visibility,
            'phase_arm_1': phase_1,
            'phase_arm_2': phase_2,
            'protocol': 'sigma_interferometry'
        }
    
    def predict_standard_qm(self, setup: InterferometryExperiment) -> Dict:
        """Predict standard QM interferometric signal."""
        path_1 = np.linspace(0, setup.path_length, 1000)
        path_2 = np.linspace(0, setup.path_length, 1000)
        
        modified_setup = InterferometryExperiment(
            duration=setup.duration,
            precision=setup.precision,
            environment=setup.environment,
            feasibility_score=setup.feasibility_score,
            path_length=setup.path_length,
            wavelength=setup.wavelength,
            redshift_gradient=setup.redshift_gradient * 2.0,
            beam_splitter_efficiency=setup.beam_splitter_efficiency
        )
        
        phase_1 = self.phase_accumulation_tau(path_1, setup)
        phase_2 = self.phase_accumulation_tau(path_2, modified_setup)
        
        phase_difference = phase_2 - phase_1
        visibility = self.interferometric_visibility(phase_difference, setup)
        
        return {
            'phase_difference': phase_difference,
            'visibility': visibility,
            'phase_arm_1': phase_1,
            'phase_arm_2': phase_2,
            'protocol': 'tau_interferometry'
        }
    
    def distinguishability(self, setup: InterferometryExperiment) -> float:
        """Calculate distinguishability for interferometry experiment."""
        ltqg_pred = self.predict_signal(setup)
        qm_pred = self.predict_standard_qm(setup)
        
        # Phase difference is the key observable
        phase_diff_ltqg = ltqg_pred['phase_difference']
        phase_diff_qm = qm_pred['phase_difference']
        
        # Distinguishability based on phase resolution
        delta_phase = abs(phase_diff_ltqg - phase_diff_qm)
        phase_resolution = 2 * np.pi * setup.precision  # Phase noise
        
        return delta_phase / phase_resolution
    
    def parameter_sweep_analysis(self, base_setup: InterferometryExperiment,
                                sweep_type: str = "tau0",
                                sweep_range: Tuple[float, float] = (0.1, 10.0),
                                n_points: int = 20) -> Dict:
        """
        Run parameter sweep to test whether large phase differences are physical or scaling artifacts.
        
        This analysis varies key parameters (τ₀, σ range, redshift gradient) to determine
        if the dramatic differences between LTQG and QM predictions are robust physical
        effects or numerical/scaling artifacts.
        
        Args:
            base_setup: Base experimental configuration
            sweep_type: Parameter to sweep ('tau0', 'redshift_gradient', 'wavelength', 'path_length')
            sweep_range: (min, max) values for the parameter sweep
            n_points: Number of points in the sweep
            
        Returns:
            Dictionary with sweep results and analysis
        """
        
        # Generate parameter values for sweep
        if sweep_type == "tau0":
            # Sweep τ₀ from 0.1 to 10.0 (in units of base τ₀)
            tau0_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
            sweep_values = tau0_values
        elif sweep_type == "redshift_gradient":
            # Sweep redshift gradient 
            gradient_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
            sweep_values = gradient_values
        elif sweep_type == "wavelength":
            # Sweep wavelength (affects wavenumber and thus phase accumulation)
            wavelength_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
            sweep_values = wavelength_values
        elif sweep_type == "path_length":
            # Sweep path length
            path_length_values = np.linspace(sweep_range[0], sweep_range[1], n_points)
            sweep_values = path_length_values
        else:
            raise ValueError(f"Unknown sweep type: {sweep_type}")
        
        # Storage for results
        ltqg_phase_diffs = []
        qm_phase_diffs = []
        ltqg_visibilities = []
        qm_visibilities = []
        relative_differences = []
        distinguishabilities = []
        
        # Run sweep
        for value in sweep_values:
            # Create modified setup for this parameter value
            if sweep_type == "tau0":
                # Modify the simulator's τ₀ value
                from ltqg_core import LTQGConfig, LTQGSimulator
                modified_config = LTQGConfig(tau0=value)
                modified_simulator = LTQGSimulator(modified_config)
                modified_protocol = AnalogGravityInterferometry(modified_simulator)
                current_setup = base_setup
                
            elif sweep_type == "redshift_gradient":
                modified_protocol = self
                current_setup = InterferometryExperiment(
                    duration=base_setup.duration,
                    precision=base_setup.precision,
                    environment=base_setup.environment,
                    feasibility_score=base_setup.feasibility_score,
                    path_length=base_setup.path_length,
                    wavelength=base_setup.wavelength,
                    redshift_gradient=value,
                    beam_splitter_efficiency=base_setup.beam_splitter_efficiency
                )
                
            elif sweep_type == "wavelength":
                modified_protocol = self
                current_setup = InterferometryExperiment(
                    duration=base_setup.duration,
                    precision=base_setup.precision,
                    environment=base_setup.environment,
                    feasibility_score=base_setup.feasibility_score,
                    path_length=base_setup.path_length,
                    wavelength=value,
                    redshift_gradient=base_setup.redshift_gradient,
                    beam_splitter_efficiency=base_setup.beam_splitter_efficiency
                )
                
            elif sweep_type == "path_length":
                modified_protocol = self
                current_setup = InterferometryExperiment(
                    duration=base_setup.duration,
                    precision=base_setup.precision,
                    environment=base_setup.environment,
                    feasibility_score=base_setup.feasibility_score,
                    path_length=value,
                    wavelength=base_setup.wavelength,
                    redshift_gradient=base_setup.redshift_gradient,
                    beam_splitter_efficiency=base_setup.beam_splitter_efficiency
                )
            
            # Get predictions for this parameter value
            ltqg_pred = modified_protocol.predict_signal(current_setup)
            qm_pred = modified_protocol.predict_standard_qm(current_setup)
            
            # Store results
            ltqg_phase_diff = ltqg_pred['phase_difference']
            qm_phase_diff = qm_pred['phase_difference']
            
            ltqg_phase_diffs.append(ltqg_phase_diff)
            qm_phase_diffs.append(qm_phase_diff)
            ltqg_visibilities.append(ltqg_pred['visibility'])
            qm_visibilities.append(qm_pred['visibility'])
            
            # Calculate relative difference
            if abs(qm_phase_diff) > 1e-10:
                rel_diff = abs(ltqg_phase_diff - qm_phase_diff) / abs(qm_phase_diff)
            else:
                rel_diff = abs(ltqg_phase_diff - qm_phase_diff)
            relative_differences.append(rel_diff)
            
            # Calculate distinguishability
            distinguishability = modified_protocol.distinguishability(current_setup)
            distinguishabilities.append(distinguishability)
        
        # Convert to numpy arrays for analysis
        ltqg_phase_diffs = np.array(ltqg_phase_diffs)
        qm_phase_diffs = np.array(qm_phase_diffs)
        ltqg_visibilities = np.array(ltqg_visibilities)
        qm_visibilities = np.array(qm_visibilities)
        relative_differences = np.array(relative_differences)
        distinguishabilities = np.array(distinguishabilities)
        
        # Analyze scaling behavior
        scaling_analysis = self._analyze_scaling_behavior(
            sweep_values, ltqg_phase_diffs, qm_phase_diffs, relative_differences
        )
        
        return {
            'sweep_type': sweep_type,
            'sweep_values': sweep_values,
            'ltqg_phase_differences': ltqg_phase_diffs,
            'qm_phase_differences': qm_phase_diffs,
            'ltqg_visibilities': ltqg_visibilities,
            'qm_visibilities': qm_visibilities,
            'relative_differences': relative_differences,
            'distinguishabilities': distinguishabilities,
            'scaling_analysis': scaling_analysis,
            'summary': {
                'mean_relative_difference': np.mean(relative_differences),
                'std_relative_difference': np.std(relative_differences),
                'min_relative_difference': np.min(relative_differences),
                'max_relative_difference': np.max(relative_differences),
                'mean_distinguishability': np.mean(distinguishabilities),
                'stable_effect': scaling_analysis['is_stable']
            }
        }
    
    def _analyze_scaling_behavior(self, sweep_values: np.ndarray,
                                ltqg_phases: np.ndarray,
                                qm_phases: np.ndarray,
                                relative_diffs: np.ndarray) -> Dict:
        """
        Analyze the scaling behavior to determine if effects are physical or artifacts.
        
        Physical effects should:
        1. Scale predictably with physical parameters
        2. Maintain relative differences within reasonable bounds
        3. Not show pathological scaling behavior
        
        Scaling artifacts typically:
        1. Show exponential or unbounded growth
        2. Have relative differences that vary wildly
        3. Depend sensitively on numerical parameters like τ₀
        """
        
        # Check for exponential scaling (potential artifact)
        log_sweep = np.log10(np.maximum(sweep_values, 1e-10))
        log_rel_diff = np.log10(np.maximum(relative_diffs, 1e-10))
        
        # Linear fit in log-log space to detect power-law scaling
        try:
            poly_coeffs = np.polyfit(log_sweep, log_rel_diff, 1)
            power_law_exponent = poly_coeffs[0]
            correlation = np.corrcoef(log_sweep, log_rel_diff)[0, 1]
        except:
            power_law_exponent = 0.0
            correlation = 0.0
        
        # Check stability criteria
        rel_diff_variation = np.std(relative_diffs) / np.mean(relative_diffs) if np.mean(relative_diffs) > 0 else np.inf
        
        # Determine if effect is stable (physical) or unstable (artifact)
        is_stable = (
            abs(power_law_exponent) < 2.0 and  # Not too steep scaling
            rel_diff_variation < 5.0 and        # Reasonable variation
            np.mean(relative_diffs) > 0.01 and  # Meaningful difference
            abs(correlation) < 0.95              # Not perfectly correlated (would suggest pure scaling)
        )
        
        # Check for specific scaling patterns
        if abs(power_law_exponent) > 3.0:
            scaling_type = "exponential_artifact"
        elif abs(power_law_exponent) < 0.1:
            scaling_type = "scale_invariant_physical"
        elif 0.1 <= abs(power_law_exponent) <= 2.0:
            scaling_type = "power_law_physical"
        else:
            scaling_type = "unstable_artifact"
        
        return {
            'power_law_exponent': power_law_exponent,
            'correlation_coefficient': correlation,
            'relative_variation': rel_diff_variation,
            'is_stable': is_stable,
            'scaling_type': scaling_type,
            'interpretation': self._interpret_scaling(scaling_type, power_law_exponent, is_stable)
        }
    
    def _interpret_scaling(self, scaling_type: str, exponent: float, is_stable: bool) -> str:
        """Provide human-readable interpretation of scaling analysis."""
        
        if scaling_type == "scale_invariant_physical":
            return (f"Effect appears PHYSICAL: Relative differences remain roughly constant "
                   f"across parameter range (exponent ≈ {exponent:.2f}). This suggests a "
                   f"genuine physical difference between LTQG and standard QM.")
        
        elif scaling_type == "power_law_physical":
            return (f"Effect appears PHYSICAL: Shows moderate power-law scaling "
                   f"(exponent = {exponent:.2f}) consistent with physical parameter dependence. "
                   f"The effect scales predictably with the varied parameter.")
        
        elif scaling_type == "exponential_artifact":
            return (f"Effect likely NUMERICAL ARTIFACT: Shows steep scaling "
                   f"(exponent = {exponent:.2f}) suggesting exponential growth. "
                   f"This indicates the large phase differences may be due to "
                   f"numerical scaling rather than genuine physics.")
        
        elif scaling_type == "unstable_artifact":
            return (f"Effect likely NUMERICAL ARTIFACT: Unstable scaling behavior "
                   f"(exponent = {exponent:.2f}) with high variation. "
                   f"This suggests the effect is sensitive to numerical parameters "
                   f"rather than representing genuine physical differences.")
        
        else:
            stable_str = "stable" if is_stable else "unstable"
            return (f"Effect classification uncertain: {scaling_type} with {stable_str} "
                   f"behavior (exponent = {exponent:.2f}). Further analysis recommended.")


class EarlyUniverseSignatures(ExperimentalProtocol):
    """
    Implementation of early-universe LTQG signature calculations.
    
    Computes predicted modifications to cosmological observables
    due to σ-time evolution in the early universe.
    """
    
    def __init__(self, simulator: LTQGSimulator):
        super().__init__(simulator)
    
    def primordial_power_spectrum_standard(self, k: np.ndarray) -> np.ndarray:
        """
        Standard primordial power spectrum P(k) ∝ k^(ns-1).
        
        Args:
            k: Comoving wavenumber array
            
        Returns:
            Primordial power spectrum
        """
        # Standard slow-roll prediction
        ns = 0.965  # Spectral index from Planck
        As = 2.1e-9  # Amplitude from Planck
        
        k_pivot = 0.05  # Mpc^-1
        return As * (k / k_pivot)**(ns - 1)
    
    def primordial_power_spectrum_ltqg(self, k: np.ndarray) -> np.ndarray:
        """
        LTQG-modified primordial power spectrum.
        
        The σ-time evolution predicts a soft UV suppression and
        reduced trans-Planckian sensitivity.
        """
        # Base spectrum
        P_standard = self.primordial_power_spectrum_standard(k)
        
        # LTQG modifications
        k_planck = 1e22  # Planck scale in Mpc^-1 (rough)
        
        # Soft UV cutoff from asymptotic silence condition
        uv_suppression = np.exp(-(k / k_planck)**0.1)
        
        # Enhanced large-scale power from σ-uniform evolution
        ir_enhancement = 1.0 + 0.01 * np.exp(-(k * 1000)**2)
        
        return P_standard * uv_suppression * ir_enhancement
    
    def cmb_temperature_spectrum(self, ell: np.ndarray, 
                               primordial_spectrum: Callable) -> np.ndarray:
        """
        Compute CMB temperature power spectrum from primordial spectrum.
        
        This is a simplified calculation - real analysis requires CAMB/CLASS.
        """
        # Simplified transfer function
        k_values = ell / 14000.0  # Rough k-ell relation
        
        # Primordial power
        P_primordial = primordial_spectrum(k_values)
        
        # Transfer function (simplified)
        transfer = np.exp(-0.5 * (ell / 1000)**2)  # Approximate damping
        
        # CMB power spectrum
        C_ell = P_primordial * transfer * ell * (ell + 1) / (2 * np.pi)
        
        return C_ell
    
    def predict_signal(self, setup: CosmologyExperiment) -> Dict:
        """Predict LTQG signatures in CMB."""
        # Multipole range corresponding to frequency range
        ell_min = int(setup.frequency_range[0] * 1000)  # Rough conversion
        ell_max = int(setup.frequency_range[1] * 1000)
        ell_range = np.logspace(np.log10(max(ell_min, 2)), 
                               np.log10(ell_max), 1000)
        
        # LTQG CMB spectrum
        C_ell_ltqg = self.cmb_temperature_spectrum(ell_range, 
                                                  self.primordial_power_spectrum_ltqg)
        
        # Observational constraints
        noise_level = setup.sensitivity_limit
        observable_range = C_ell_ltqg > noise_level
        
        return {
            'ell_range': ell_range,
            'C_ell': C_ell_ltqg,
            'observable_range': observable_range,
            'total_signal': np.sum(C_ell_ltqg[observable_range]),
            'protocol': 'ltqg_cosmology'
        }
    
    def predict_standard_qm(self, setup: CosmologyExperiment) -> Dict:
        """Predict standard cosmological signatures."""
        ell_min = int(setup.frequency_range[0] * 1000)
        ell_max = int(setup.frequency_range[1] * 1000)
        ell_range = np.logspace(np.log10(max(ell_min, 2)), 
                               np.log10(ell_max), 1000)
        
        # Standard CMB spectrum
        C_ell_standard = self.cmb_temperature_spectrum(ell_range,
                                                      self.primordial_power_spectrum_standard)
        
        noise_level = setup.sensitivity_limit
        observable_range = C_ell_standard > noise_level
        
        return {
            'ell_range': ell_range,
            'C_ell': C_ell_standard,
            'observable_range': observable_range,
            'total_signal': np.sum(C_ell_standard[observable_range]),
            'protocol': 'standard_cosmology'
        }
    
    def distinguishability(self, setup: CosmologyExperiment) -> float:
        """Calculate distinguishability for cosmological observations."""
        ltqg_pred = self.predict_signal(setup)
        standard_pred = self.predict_standard_qm(setup)
        
        # Compare total signals
        signal_ltqg = ltqg_pred['total_signal']
        signal_standard = standard_pred['total_signal']
        
        # Relative difference
        if signal_standard > 0:
            relative_diff = abs(signal_ltqg - signal_standard) / signal_standard
        else:
            relative_diff = 0.0
        
        # Convert to significance (rough estimate)
        cosmic_variance = 0.01  # Typical cosmic variance for CMB
        significance = relative_diff / cosmic_variance
        
        return min(significance, 20.0)  # Cap at 20σ


class ClockTransportProtocol(ExperimentalProtocol):
    """
    Implementation of clock transport loop experiments.
    
    Tests whether quantum processes controlled uniformly in σ during
    transport show measurable differences based on redshift path history.
    """
    
    def __init__(self, simulator: LTQGSimulator):
        super().__init__(simulator)
    
    def gravitational_path_integral(self, path_redshifts: np.ndarray,
                                  path_times: np.ndarray) -> float:
        """
        Calculate accumulated gravitational phase along a path.
        
        **CORRECTED**: Now includes proper σ-Jacobian factor.
        
        When integrating a σ-dependent quantity f(σ(t)) over proper time,
        the correct measure is: ∫ f(σ(t)) dτ = ∫ f(σ) τ(σ) dσ
        where τ(σ) = τ₀ exp(σ) is the Jacobian.
        
        Args:
            path_redshifts: α(t) along the path
            path_times: Time points along the path
            
        Returns:
            Total accumulated gravitational phase with correct measure
        """
        # Ensure path_times are positive
        if np.any(path_times <= 0):
            # Handle case where path starts at t=0
            path_times = np.maximum(path_times, 1e-10)
        
        # σ-shifts along the path: σ = log(α)
        sigma_shifts = self.simulator.redshift.sigma_shift_gravitational(path_redshifts)
        
        # CRITICAL CORRECTION: For path integrals in spacetime, we integrate over proper time
        # The phase accumulation is: ∫ K(σ(t)) dt where K(σ) = τ₀ exp(σ) H
        
        # Direct integration over time with σ-dependent generator
        # At each time point t, we have redshift α(t) → σ(t) = log(α(t))
        # The generator magnitude is K(t) = τ₀ exp(σ(t)) H
        
        # Phase integrand: |K(σ(t))| = τ₀ exp(σ(t)) |H|
        # For demonstration, assume |H| = 1
        phase_integrand = self.simulator.config.tau0 * np.exp(sigma_shifts)
        
        # Integrate over proper time
        if len(path_times) > 1:
            total_phase = np.trapz(phase_integrand, path_times)
        else:
            total_phase = 0.0
        
        return total_phase
    
    def design_transport_paths(self, setup: ExperimentalSetup) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design two different transport paths with same start/end points
        but different redshift histories.
        
        Returns:
            Tuple of (path_1_redshifts, path_2_redshifts)
        """
        n_points = 1000
        time_points = np.linspace(0, setup.duration, n_points)
        
        # Path 1: Stays at constant weak redshift
        alpha_path_1 = np.full(n_points, 0.9)
        
        # Path 2: Goes through strong redshift region and back
        alpha_path_2 = 0.9 - 0.8 * np.exp(-((time_points - setup.duration/2) / (setup.duration/8))**2)
        alpha_path_2 = np.maximum(alpha_path_2, 0.01)  # Avoid unphysical values
        
        return alpha_path_1, alpha_path_2
    
    def predict_signal(self, setup: ExperimentalSetup) -> Dict:
        """Predict LTQG signal for clock transport experiment."""
        alpha_1, alpha_2 = self.design_transport_paths(setup)
        time_points = np.linspace(0, setup.duration, len(alpha_1))
        
        # Calculate phases for both paths
        phase_1 = self.gravitational_path_integral(alpha_1, time_points)
        phase_2 = self.gravitational_path_integral(alpha_2, time_points)
        
        # LTQG predicts path-dependent phase accumulation
        phase_difference = phase_2 - phase_1
        
        # Observable quantity: phase coherence after loop closure
        coherence = np.cos(phase_difference)
        
        return {
            'phase_path_1': phase_1,
            'phase_path_2': phase_2,
            'phase_difference': phase_difference,
            'coherence': coherence,
            'protocol': 'ltqg_clock_transport'
        }
    
    def predict_standard_qm(self, setup: ExperimentalSetup) -> Dict:
        """Predict standard QM signal (no path dependence expected)."""
        alpha_1, alpha_2 = self.design_transport_paths(setup)
        
        # Standard QM: no path dependence for closed loops
        # Phase difference should be zero (up to experimental noise)
        phase_difference = 0.0
        coherence = 1.0  # Perfect coherence
        
        return {
            'phase_path_1': 0.0,
            'phase_path_2': 0.0,
            'phase_difference': phase_difference,
            'coherence': coherence,
            'protocol': 'standard_clock_transport'
        }
    
    def distinguishability(self, setup: ExperimentalSetup) -> float:
        """Calculate distinguishability for clock transport experiment."""
        ltqg_pred = self.predict_signal(setup)
        qm_pred = self.predict_standard_qm(setup)
        
        # Key observable is the coherence difference
        coherence_diff = abs(ltqg_pred['coherence'] - qm_pred['coherence'])
        
        # Measurement precision determines distinguishability
        return coherence_diff / setup.precision


class ExperimentalSuite:
    """
    Comprehensive experimental suite for testing LTQG predictions.
    
    Coordinates multiple experimental protocols and provides
    overall assessment of theoretical distinguishability.
    """
    
    def __init__(self, simulator: LTQGSimulator = None):
        self.simulator = simulator or create_ltqg_simulator()
        
        # Initialize experimental protocols
        self.zeno_protocol = SigmaUniformZenoProtocol(self.simulator)
        self.interferometry_protocol = AnalogGravityInterferometry(self.simulator)
        self.cosmology_protocol = EarlyUniverseSignatures(self.simulator)
        self.clock_transport_protocol = ClockTransportProtocol(self.simulator)
    
    def design_optimal_zeno_experiment(self) -> ZenoExperiment:
        """Design an optimal Zeno experiment for current technology."""
        return ZenoExperiment(
            duration=1e-3,  # 1 ms total duration
            precision=1e-6,  # 1 ppm precision
            environment="Trapped ion with optical transitions",
            feasibility_score=0.7,
            n_measurements=20,
            redshift_factor=0.1,  # Strong simulated redshift
            two_level_spacing=1e15,  # Optical transition frequency
            decoherence_rate=1e3  # 1 kHz decoherence
        )
    
    def design_optimal_interferometry_experiment(self) -> InterferometryExperiment:
        """Design an optimal interferometry experiment."""
        return InterferometryExperiment(
            duration=1e-1,  # 100 ms
            precision=1e-9,  # nanorad phase precision
            environment="BEC in time-varying trap",
            feasibility_score=0.5,
            path_length=1e-3,  # 1 mm path length
            wavelength=589e-9,  # Sodium D-line
            redshift_gradient=1e3,  # Strong gradient
            beam_splitter_efficiency=0.95
        )
    
    def design_optimal_cosmology_experiment(self) -> CosmologyExperiment:
        """Design an optimal cosmology observation."""
        return CosmologyExperiment(
            duration=3.15e7,  # 1 year observation
            precision=1e-6,  # μK temperature precision
            environment="Next-generation CMB survey",
            feasibility_score=0.3,
            frequency_range=(30e9, 857e9),  # Planck frequency range
            angular_resolution=5e-6,  # 5 μrad
            sensitivity_limit=1e-12,  # Very sensitive
            observation_time=3.15e7
        )
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run comprehensive analysis of all experimental protocols.
        
        Returns:
            Dictionary with results from all protocols
        """
        results = {}
        
        # Zeno experiment
        zeno_setup = self.design_optimal_zeno_experiment()
        results['zeno'] = {
            'setup': zeno_setup,
            'ltqg_prediction': self.zeno_protocol.predict_signal(zeno_setup),
            'qm_prediction': self.zeno_protocol.predict_standard_qm(zeno_setup),
            'distinguishability': self.zeno_protocol.distinguishability(zeno_setup)
        }
        
        # Interferometry experiment
        interferometry_setup = self.design_optimal_interferometry_experiment()
        results['interferometry'] = {
            'setup': interferometry_setup,
            'ltqg_prediction': self.interferometry_protocol.predict_signal(interferometry_setup),
            'qm_prediction': self.interferometry_protocol.predict_standard_qm(interferometry_setup),
            'distinguishability': self.interferometry_protocol.distinguishability(interferometry_setup)
        }
        
        # Cosmology experiment
        cosmology_setup = self.design_optimal_cosmology_experiment()
        results['cosmology'] = {
            'setup': cosmology_setup,
            'ltqg_prediction': self.cosmology_protocol.predict_signal(cosmology_setup),
            'qm_prediction': self.cosmology_protocol.predict_standard_qm(cosmology_setup),
            'distinguishability': self.cosmology_protocol.distinguishability(cosmology_setup)
        }
        
        # Clock transport experiment
        clock_setup = ExperimentalSetup(
            duration=1e-2, precision=1e-8, 
            environment="Atomic clock transport", feasibility_score=0.6)
        results['clock_transport'] = {
            'setup': clock_setup,
            'ltqg_prediction': self.clock_transport_protocol.predict_signal(clock_setup),
            'qm_prediction': self.clock_transport_protocol.predict_standard_qm(clock_setup),
            'distinguishability': self.clock_transport_protocol.distinguishability(clock_setup)
        }
        
        return results
    
    def generate_experimental_summary(self) -> str:
        """Generate a summary report of experimental predictions."""
        results = self.run_comprehensive_analysis()
        
        summary = "LTQG Experimental Predictions Summary\n"
        summary += "=" * 40 + "\n\n"
        
        for protocol_name, result in results.items():
            setup = result['setup']
            distinguishability = result['distinguishability']
            
            summary += f"{protocol_name.upper()} EXPERIMENT:\n"
            summary += f"  Environment: {setup.environment}\n"
            summary += f"  Feasibility: {setup.feasibility_score:.1f}/1.0\n"
            summary += f"  Distinguishability: {distinguishability:.2f}σ\n"
            summary += f"  Precision required: {setup.precision:.2e}\n"
            summary += f"  Duration: {setup.duration:.2e} s\n\n"
        
        # Overall assessment
        avg_distinguishability = np.mean([r['distinguishability'] for r in results.values()])
        avg_feasibility = np.mean([r['setup'].feasibility_score for r in results.values()])
        
        summary += "OVERALL ASSESSMENT:\n"
        summary += f"  Average distinguishability: {avg_distinguishability:.2f}σ\n"
        summary += f"  Average feasibility: {avg_feasibility:.2f}/1.0\n"
        
        if avg_distinguishability > 3.0:
            summary += "  Status: LTQG predictions are experimentally accessible!\n"
        elif avg_distinguishability > 1.0:
            summary += "  Status: LTQG predictions may be detectable with effort.\n"
        else:
            summary += "  Status: LTQG predictions are challenging to detect.\n"
        
        return summary


def demo_experimental_suite():
    """Demonstration of the experimental suite."""
    print("LTQG Experimental Design Suite")
    print("==============================")
    
    suite = ExperimentalSuite()
    summary = suite.generate_experimental_summary()
    print(summary)
    
    # Generate a detailed analysis for one protocol
    results = suite.run_comprehensive_analysis()
    zeno_result = results['zeno']
    
    print("\nDetailed Zeno Experiment Analysis:")
    print("-" * 35)
    ltqg = zeno_result['ltqg_prediction']
    qm = zeno_result['qm_prediction']
    
    print(f"LTQG survival probability: {ltqg['survival_probability']:.6f}")
    print(f"Standard QM survival prob: {qm['survival_probability']:.6f}")
    print(f"Relative difference: {abs(ltqg['survival_probability'] - qm['survival_probability']):.6f}")
    print(f"Zeno enhancement factor: {ltqg['zeno_enhancement']:.6f}")


if __name__ == "__main__":
    demo_experimental_suite()