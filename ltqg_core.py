"""
Log-Time Quantum Gravity (LTQG) Core Library
============================================

Author: Denzil James Greenwood
GitHub: https://github.com/DenzilGreenwood/log_time
License: MIT

This module implements the core mathematical framework for Log-Time Quantum Gravity,
a theory that reconciles General Relativity's multiplicative time-dilation structure
with Quantum Mechanics' additive phase evolution through the logarithmic time variable:

    σ = log(τ/τ₀)

where τ is proper time and τ₀ is a reference constant (typically Planck time).

Key Features:
- Time transformations and coordinate mappings
- Singularity regularization
- Modified Schrödinger evolution in σ-time
- Gravitational redshift as additive σ-shifts
- Asymptotic silence condition
- Quantum measurement protocols in σ-time
"""

import numpy as np
import scipy.linalg as linalg
from typing import Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Physical constants (in natural units where ℏ = c = 1)
PLANCK_TIME = 1.0  # τ₀ default value
PLANCK_MASS = 1.0
PLANCK_LENGTH = 1.0


@dataclass
class LTQGConfig:
    """Configuration parameters for LTQG calculations."""
    tau0: float = PLANCK_TIME
    hbar: float = 1.0
    c: float = 1.0
    
    def __post_init__(self):
        if self.tau0 <= 0:
            raise ValueError("Reference time τ₀ must be positive")


class TimeTransform:
    """
    Core time transformation utilities for LTQG.
    
    Handles conversions between proper time τ and log-time σ,
    ensuring numerical stability and physical consistency.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
    
    def sigma_from_tau(self, tau: np.ndarray) -> np.ndarray:
        """
        Convert proper time τ to log-time σ.
        
        σ = log(τ/τ₀)
        
        Args:
            tau: Proper time values (must be positive)
            
        Returns:
            Log-time values σ
        """
        if np.any(tau <= 0):
            raise ValueError("Proper time τ must be positive")
        return np.log(tau / self.config.tau0)
    
    def tau_from_sigma(self, sigma: np.ndarray) -> np.ndarray:
        """
        Convert log-time σ back to proper time τ.
        
        τ = τ₀ × exp(σ)
        
        Args:
            sigma: Log-time values
            
        Returns:
            Proper time values τ
        """
        return self.config.tau0 * np.exp(sigma)
    
    def dtau_dsigma(self, sigma: np.ndarray) -> np.ndarray:
        """
        Compute the Jacobian dτ/dσ = τ₀ exp(σ).
        
        This factor appears in the chain rule for time derivatives.
        """
        return self.config.tau0 * np.exp(sigma)
    
    def safe_log(self, x: np.ndarray, floor: float = 1e-300) -> np.ndarray:
        """
        Numerically safe logarithm with positive floor to avoid -∞.
        
        Args:
            x: Input values
            floor: Minimum value to clamp to before taking log
            
        Returns:
            Safe logarithm values
        """
        return np.log(np.maximum(x, floor))


class SingularityRegularization:
    """
    Handles the regularization of classical singularities in σ-time.
    
    Converts polynomial divergences Q(τ) ∝ τ^(-n) into exponential
    decay Q(σ) ∝ exp(-nσ) as σ → -∞.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.transform = TimeTransform(config)
    
    def regularize_power_divergence(self, sigma: np.ndarray, n: float) -> np.ndarray:
        """
        Regularize a quantity that diverges as Q(τ) ∝ τ^(-n).
        
        In σ-space: Q(σ) = (1/τ₀^n) × exp(-nσ)
        
        Args:
            sigma: Log-time values
            n: Power of divergence (n > 0 for divergent behavior)
            
        Returns:
            Regularized quantity Q(σ)
        """
        return (1.0 / self.config.tau0**n) * np.exp(-n * sigma)
    
    def curvature_scalar(self, sigma: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Example curvature scalar that diverges as R ∝ 1/τ² in classical GR.
        
        In σ-space: R(σ) = amplitude × exp(-2σ) / τ₀²
        """
        return self.regularize_power_divergence(sigma, n=2.0) * amplitude
    
    def energy_density(self, sigma: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """
        Example energy density that diverges as ρ ∝ 1/τ⁴.
        
        In σ-space: ρ(σ) = amplitude × exp(-4σ) / τ₀⁴
        """
        return self.regularize_power_divergence(sigma, n=4.0) * amplitude


class QuantumEvolution:
    """
    Implements quantum evolution in σ-time via the modified Schrödinger equation.
    
    The evolution is governed by:
    iℏ ∂|ψ⟩/∂σ = K(σ)|ψ⟩
    
    where K(σ) = τ₀ exp(σ) H is the σ-time generator.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.transform = TimeTransform(config)
    
    def sigma_generator(self, H: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Compute the σ-time evolution generator K(σ).
        
        K(σ) = τ₀ exp(σ) H
        
        Args:
            H: Hamiltonian matrix
            sigma: Log-time value (scalar)
            
        Returns:
            Generator matrix K(σ)
        """
        return self.config.tau0 * np.exp(sigma) * H
    
    def evolution_operator(self, H: np.ndarray, sigma_i: float, sigma_f: float, 
                          steps: int = 1000) -> np.ndarray:
        """
        Compute the σ-time evolution operator U(σf, σi).
        
        Uses time-ordered integration for the σ-dependent generator.
        
        Args:
            H: Time-independent Hamiltonian
            sigma_i: Initial σ-time
            sigma_f: Final σ-time
            steps: Number of integration steps
            
        Returns:
            Evolution operator U(σf, σi)
        """
        sigma_points = np.linspace(sigma_i, sigma_f, steps)
        dsigma = (sigma_f - sigma_i) / (steps - 1)
        
        # Initialize evolution operator
        U = np.eye(H.shape[0], dtype=complex)
        
        # Time-ordered evolution
        for sigma in sigma_points[1:]:
            K = self.sigma_generator(H, sigma)
            dU = -1j * K * dsigma / self.config.hbar
            U = np.dot(linalg.expm(dU), U)
        
        return U
    
    def asymptotic_silence_condition(self, sigma: np.ndarray, threshold: float = -20) -> np.ndarray:
        """
        Check the asymptotic silence condition: K(σ) → 0 as σ → -∞.
        
        Returns a mask indicating where the generator is effectively zero.
        
        Args:
            sigma: Log-time values
            threshold: σ threshold below which generator is considered silent
            
        Returns:
            Boolean mask for silence condition
        """
        return sigma < threshold
    
    def tau_invariant_phase(self, H: np.ndarray, tau_i: float, tau_f: float) -> float:
        """
        Compute the total accumulated phase in proper time.
        
        This should be independent of the σ parameterization, demonstrating
        gauge invariance of the framework.
        
        Args:
            H: Hamiltonian (assuming it's a scalar eigenvalue)
            tau_i: Initial proper time
            tau_f: Final proper time
            
        Returns:
            Total phase accumulated
        """
        return np.trace(H) * (tau_f - tau_i) / self.config.hbar


class GravitationalRedshift:
    """
    Implements gravitational redshift effects in the σ-time framework.
    
    Gravitational time dilation becomes additive shifts in σ-coordinates,
    unifying the multiplicative structure of GR with the additive structure of QM.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.transform = TimeTransform(config)
    
    def redshift_factor_schwarzschild(self, r: np.ndarray, rs: float) -> np.ndarray:
        """
        Schwarzschild redshift factor α(r) = √(1 - rs/r).
        
        Args:
            r: Radial coordinate (must be > rs)
            rs: Schwarzschild radius
            
        Returns:
            Redshift factor α(r)
        """
        if np.any(r <= rs):
            raise ValueError("Radial coordinate must be outside Schwarzschild radius")
        return np.sqrt(1.0 - rs / r)
    
    def sigma_shift_gravitational(self, alpha: np.ndarray) -> np.ndarray:
        """
        Convert gravitational redshift to additive σ-shift.
        
        If proper times are related by τ_B = α τ_A, then:
        σ_B = σ_A + log(α)
        
        Args:
            alpha: Redshift factor
            
        Returns:
            Additive shift in σ-coordinate
        """
        return self.transform.safe_log(alpha)
    
    def horizon_approach_sigma(self, r: np.ndarray, rs: float, 
                              tau_ratio_log: float = 0.0) -> np.ndarray:
        """
        Compute σ-coordinate near a black hole horizon.
        
        σ(r) = log(α(r)) + log(τ/τ₀)
        
        As r → rs⁺, σ → -∞, implementing the horizon boundary naturally.
        
        Args:
            r: Radial coordinates
            rs: Schwarzschild radius
            tau_ratio_log: Additional log(τ/τ₀) term
            
        Returns:
            σ-coordinates σ(r)
        """
        alpha = self.redshift_factor_schwarzschild(r, rs)
        return self.sigma_shift_gravitational(alpha) + tau_ratio_log


class CosmologicalModels:
    """
    Implements cosmological models in σ-time coordinates.
    
    Converts standard FLRW scale factor evolution a(t) into
    σ-time behavior, revealing the natural exponential structure.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.transform = TimeTransform(config)
    
    def scale_factor_power_law(self, sigma: np.ndarray, n: float) -> np.ndarray:
        """
        Scale factor for power-law cosmology a(t) ∝ t^n.
        
        In σ-time: a(σ) = a₀ exp(nσ)
        
        Args:
            sigma: Log-time coordinates
            n: Power law index (1/2 for radiation, 2/3 for matter)
            
        Returns:
            Scale factor a(σ)
        """
        return np.exp(n * sigma)
    
    def scale_factor_de_sitter(self, sigma: np.ndarray, H: float) -> np.ndarray:
        """
        Scale factor for de Sitter (exponential) expansion.
        
        a(t) = exp(Ht) → a(σ) = exp(H τ₀ exp(σ))
        
        Args:
            sigma: Log-time coordinates
            H: Hubble parameter
            
        Returns:
            Scale factor a(σ)
        """
        t = self.transform.tau_from_sigma(sigma)
        return np.exp(H * t)
    
    def hubble_rate_sigma(self, sigma: np.ndarray, model: str, **kwargs) -> np.ndarray:
        """
        Compute (1/a)(da/dσ) for different cosmological models.
        
        Args:
            sigma: Log-time coordinates
            model: 'radiation', 'matter', or 'de_sitter'
            **kwargs: Model-specific parameters
            
        Returns:
            σ-frame expansion rate
        """
        if model == 'radiation':
            return np.full_like(sigma, 0.5)
        elif model == 'matter':
            return np.full_like(sigma, 2.0/3.0)
        elif model == 'de_sitter':
            H = kwargs.get('H', 1.0)
            return H * self.config.tau0 * np.exp(sigma)
        else:
            raise ValueError(f"Unknown cosmological model: {model}")


class QuantumProtocols:
    """
    Implements experimental protocols for testing LTQG predictions.
    
    Focuses on σ-uniform vs τ-uniform measurement schedules and
    their distinguishable quantum effects.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.transform = TimeTransform(config)
    
    def tau_uniform_schedule(self, tau_start: float, tau_end: float, 
                           n_measurements: int) -> np.ndarray:
        """
        Generate τ-uniform measurement schedule (standard QM).
        
        Args:
            tau_start: Initial proper time
            tau_end: Final proper time
            n_measurements: Number of measurement points
            
        Returns:
            Array of proper time measurement points
        """
        return np.linspace(tau_start, tau_end, n_measurements)
    
    def sigma_uniform_schedule(self, sigma_start: float, sigma_end: float,
                             n_measurements: int) -> np.ndarray:
        """
        Generate σ-uniform measurement schedule (LTQG prediction).
        
        Args:
            sigma_start: Initial log-time
            sigma_end: Final log-time
            n_measurements: Number of measurement points
            
        Returns:
            Array of proper time measurement points
        """
        sigma_points = np.linspace(sigma_start, sigma_end, n_measurements)
        return self.transform.tau_from_sigma(sigma_points)
    
    def zeno_suppression_factor(self, alpha: float, n_measurements: int) -> float:
        """
        Compute predicted Zeno suppression in σ-uniform protocol.
        
        For a system in gravitational redshift α, the effective generator
        magnitude changes, leading to modified Zeno effects.
        
        Args:
            alpha: Gravitational redshift factor
            n_measurements: Number of measurements
            
        Returns:
            Suppression factor relative to flat spacetime
        """
        # In strong redshift (α ≪ 1), σ becomes very negative,
        # leading to suppressed evolution generator K(σ) ∝ exp(σ)
        effective_generator_ratio = alpha  # Simplified model
        return effective_generator_ratio**(n_measurements - 1)
    
    def interferometry_phase_shift(self, path_alpha_1: np.ndarray, 
                                 path_alpha_2: np.ndarray,
                                 base_phase: float) -> float:
        """
        Compute interferometric phase difference between two paths
        with different gravitational redshift histories.
        
        Args:
            path_alpha_1: Redshift factors along path 1
            path_alpha_2: Redshift factors along path 2
            base_phase: Base quantum phase
            
        Returns:
            Differential phase shift Δφ
        """
        sigma_shift_1 = np.sum(self.transform.safe_log(path_alpha_1))
        sigma_shift_2 = np.sum(self.transform.safe_log(path_alpha_2))
        
        # Phase difference is proportional to σ-path difference
        return base_phase * (sigma_shift_1 - sigma_shift_2)
    
    def compare_measurement_protocols(self, tau_range: Tuple[float, float],
                                    alpha: float = 1.0, n_measurements: int = 10) -> dict:
        """
        Compare τ-uniform vs σ-uniform measurement protocols.
        
        **UPDATED**: Now includes proper τ-weighting for σ-uniform averages.
        
        Args:
            tau_range: (τ_start, τ_end) for measurements
            alpha: Gravitational redshift factor
            n_measurements: Number of measurements
            
        Returns:
            Dictionary comparing both protocols with proper weighting
        """
        tau_start, tau_end = tau_range
        
        # Convert to σ-coordinates
        sigma_start = self.transform.sigma_from_tau(np.array([tau_start]))[0]
        sigma_end = self.transform.sigma_from_tau(np.array([tau_end]))[0]
        
        # Generate schedules
        tau_uniform = self.tau_uniform_schedule(tau_start, tau_end, n_measurements)
        sigma_uniform = self.sigma_uniform_schedule(sigma_start, sigma_end, n_measurements)
        
        # Compute Zeno suppression
        zeno_factor = self.zeno_suppression_factor(alpha, n_measurements)
        
        # **CRITICAL ADDITION**: Validate σ-uniform protocol with proper τ-weighting
        validation_results = self.validate_sigma_uniform_protocol(
            sigma_uniform, tau_uniform, sigma_start, sigma_end
        )
        
        return {
            'tau_uniform_times': tau_uniform,
            'sigma_uniform_times': sigma_uniform,
            'zeno_suppression': zeno_factor,
            'protocol_difference': np.mean(np.abs(tau_uniform - sigma_uniform)),
            'sigma_protocol_validation': validation_results
        }
    
    def validate_sigma_uniform_protocol(self, sigma_uniform_times: np.ndarray,
                                      tau_uniform_times: np.ndarray,
                                      sigma_start: float, sigma_end: float) -> dict:
        """
        Validate σ-uniform protocol implementation according to rigorous identities.
        
        This method checks:
        1. Proper τ-weighting for σ-uniform averages
        2. Cutoff dependence and reporting
        3. Correct measure transformations
        
        Args:
            sigma_uniform_times: σ-uniform measurement times (in τ)
            tau_uniform_times: τ-uniform measurement times
            sigma_start: Initial σ value
            sigma_end: Final σ value
            
        Returns:
            Dictionary with validation results and warnings
        """
        # Convert times to σ coordinates for analysis
        sigma_coords = self.transform.sigma_from_tau(sigma_uniform_times)
        tau_coords = sigma_uniform_times
        
        # Check 1: Verify exponential spacing in τ for σ-uniform protocol
        tau_ratios = tau_coords[1:] / tau_coords[:-1]
        expected_ratio = np.exp((sigma_end - sigma_start) / (len(sigma_coords) - 1))
        spacing_error = np.std(tau_ratios) / np.mean(tau_ratios)
        
        # Check 2: Compute τ-weights for proper σ-uniform averaging
        tau_weights = tau_coords  # dτ = τ dσ, so weight by τ
        normalized_weights = tau_weights / np.sum(tau_weights)
        
        # Check 3: Assess cutoff sensitivity
        sigma_min = np.min(sigma_coords)
        tau_min = np.exp(sigma_min) * self.config.tau0
        cutoff_sensitivity = abs(sigma_min) > 10  # Warn if |σ_min| > 10
        
        # Check 4: Compare weighted vs unweighted averages (demonstration)
        # For a test observable O(τ) = τ^(-1) (example divergent quantity)
        test_observable = 1.0 / tau_coords
        
        # Unweighted σ-average (INCORRECT for τ-defined observables)
        unweighted_sigma_avg = np.mean(test_observable)
        
        # Properly weighted σ-average (CORRECT)
        weighted_sigma_avg = np.sum(test_observable * normalized_weights)
        
        # τ-uniform average for comparison
        tau_test_observable = 1.0 / tau_uniform_times
        tau_avg = np.mean(tau_test_observable)
        
        # Relative errors
        unweighted_error = abs(unweighted_sigma_avg - tau_avg) / abs(tau_avg) if tau_avg != 0 else np.inf
        weighted_error = abs(weighted_sigma_avg - tau_avg) / abs(tau_avg) if tau_avg != 0 else np.inf
        
        return {
            'spacing_error': spacing_error,
            'expected_exponential_ratio': expected_ratio,
            'tau_weights': normalized_weights,
            'sigma_cutoff': sigma_min,
            'tau_cutoff': tau_min,
            'cutoff_sensitive': cutoff_sensitivity,
            'test_observable_comparison': {
                'tau_uniform_average': tau_avg,
                'sigma_unweighted_average': unweighted_sigma_avg,
                'sigma_weighted_average': weighted_sigma_avg,
                'unweighted_relative_error': unweighted_error,
                'weighted_relative_error': weighted_error
            },
            'validation_passed': spacing_error < 0.01 and weighted_error < unweighted_error,
            'warnings': self._generate_protocol_warnings(cutoff_sensitivity, spacing_error, 
                                                        unweighted_error, weighted_error)
        }
    
    def _generate_protocol_warnings(self, cutoff_sensitive: bool, spacing_error: float,
                                  unweighted_error: float, weighted_error: float) -> list:
        """Generate warnings for σ-uniform protocol validation."""
        warnings = []
        
        if cutoff_sensitive:
            warnings.append("CUTOFF WARNING: σ_min < -10, results may be cutoff-dependent. "
                          "Report σ_min explicitly in experimental protocol.")
        
        if spacing_error > 0.05:
            warnings.append("SPACING WARNING: σ-uniform schedule has irregular τ-spacing. "
                          "Check implementation of exponential time intervals.")
        
        if unweighted_error < weighted_error:
            warnings.append("MEASURE WARNING: Unweighted σ-average outperforming weighted. "
                          "This suggests a problem with the weighting implementation.")
        
        if weighted_error > 0.1:
            warnings.append("ACCURACY WARNING: Weighted σ-average has >10% error compared to "
                          "τ-uniform. Consider increasing measurement resolution.")
        
        return warnings


class LTQGSimulator:
    """
    High-level simulator for LTQG phenomena.
    
    Provides convenient methods for running complete simulations
    and generating predictions for experimental verification.
    """
    
    def __init__(self, config: LTQGConfig = None):
        self.config = config or LTQGConfig()
        self.time_transform = TimeTransform(config)
        self.singularity = SingularityRegularization(config)
        self.evolution = QuantumEvolution(config)
        self.redshift = GravitationalRedshift(config)
        self.cosmology = CosmologicalModels(config)
        self.protocols = QuantumProtocols(config)
    
    def simulate_early_universe(self, sigma_range: Tuple[float, float],
                              n_points: int = 1000) -> dict:
        """
        Simulate early universe evolution in σ-time.
        
        Args:
            sigma_range: (σ_start, σ_end) for simulation
            n_points: Number of time points
            
        Returns:
            Dictionary with simulation results
        """
        sigma = np.linspace(sigma_range[0], sigma_range[1], n_points)
        tau = self.time_transform.tau_from_sigma(sigma)
        
        # Physical quantities
        curvature = self.singularity.curvature_scalar(sigma)
        energy_density = self.singularity.energy_density(sigma)
        
        # Scale factors for different epochs
        a_radiation = self.cosmology.scale_factor_power_law(sigma, n=0.5)
        a_matter = self.cosmology.scale_factor_power_law(sigma, n=2/3)
        
        # Asymptotic silence region
        silence_mask = self.evolution.asymptotic_silence_condition(sigma, threshold=-10)
        
        return {
            'sigma': sigma,
            'tau': tau,
            'curvature_scalar': curvature,
            'energy_density': energy_density,
            'scale_factor_radiation': a_radiation,
            'scale_factor_matter': a_matter,
            'asymptotic_silence': silence_mask
        }
    
    def simulate_black_hole_approach(self, r_range: Tuple[float, float],
                                   rs: float = 1.0, n_points: int = 1000) -> dict:
        """
        Simulate approach to black hole horizon in σ-coordinates.
        
        Args:
            r_range: (r_min, r_max) in units of rs
            rs: Schwarzschild radius
            n_points: Number of radial points
            
        Returns:
            Dictionary with simulation results
        """
        r = np.linspace(r_range[0] * rs, r_range[1] * rs, n_points)
        
        # Avoid exact horizon
        r = np.maximum(r, rs + 1e-10)
        
        alpha = self.redshift.redshift_factor_schwarzschild(r, rs)
        sigma_shift = self.redshift.sigma_shift_gravitational(alpha)
        
        # Generator suppression near horizon
        generator_magnitude = np.exp(sigma_shift)
        
        return {
            'r_over_rs': r / rs,
            'redshift_factor': alpha,
            'sigma_shift': sigma_shift,
            'generator_magnitude': generator_magnitude,
            'horizon_approach': r / rs - 1.0  # Distance from horizon
        }


# Convenience function for quick setup
def create_ltqg_simulator(tau0: float = PLANCK_TIME) -> LTQGSimulator:
    """
    Create an LTQG simulator with specified reference time.
    
    Args:
        tau0: Reference time scale (default: Planck time)
        
    Returns:
        Configured LTQGSimulator instance
    """
    config = LTQGConfig(tau0=tau0)
    return LTQGSimulator(config)


if __name__ == "__main__":
    # Demo usage
    simulator = create_ltqg_simulator()
    
    print("Log-Time Quantum Gravity Core Library")
    print("=====================================")
    print(f"Reference time τ₀ = {simulator.config.tau0}")
    
    # Quick singularity regularization demo
    sigma_demo = np.array([-10, -5, 0, 5])
    curvature_demo = simulator.singularity.curvature_scalar(sigma_demo)
    print(f"\nSingularity regularization demo:")
    print(f"σ values: {sigma_demo}")
    print(f"R(σ) values: {curvature_demo}")
    
    print("\nLibrary ready for use!")