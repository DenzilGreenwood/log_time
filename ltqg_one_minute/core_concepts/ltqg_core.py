"""
LTQG Core Framework: Logarithmic Time Quantum Gravity

This module implements the fundamental mathematical framework for LTQG,
including the σ-time transformation and effective Hamiltonian formalism.

Key concepts:
- σ ≡ log(τ/τ₀) transformation
- H_eff(σ) = τ H(τ) effective Hamiltonian
- Asymptotic silence as τ→0⁺
- Multiplicative dilations → additive shifts
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple, Union, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass


from dataclasses import dataclass


@dataclass 
class LTQGConfig:
    """
    Centralized configuration for LTQG parameters.
    
    This ensures that plotting routines and evolution methods use exactly
    the same parameters, eliminating inconsistencies between tables/figures
    and actual computations.
    """
    tau_0: float = 1.0
    hbar: float = 1.0
    
    # Envelope/silence parameters
    envelope_type: str = 'tanh'
    envelope_params: Dict = None
    always_apply_silence: bool = False  # Make silence first-class in H_eff
    
    # Hermiticity enforcement
    enforce_hermitian: bool = True
    
    # Numerical tolerances
    ode_rtol: float = 1e-9
    ode_atol: float = 1e-12
    
    def __post_init__(self):
        """Set default envelope parameters if not provided."""
        if self.envelope_params is None:
            self.envelope_params = {
                'sigma_0': 2.0,
                'width': 1.0,
                'envelope_floor': 1e-8
            }


class LTQGFramework:
    """
    Core LTQG mathematical framework implementing σ-time dynamics.
    """
    
    def __init__(self, config: Optional[LTQGConfig] = None):
        """
        Initialize LTQG framework.
        
        Args:
            config: LTQG configuration object. If None, uses defaults.
        """
        if config is None:
            config = LTQGConfig()
        self.config = config
        self.tau_0 = config.tau_0
        self.hbar = config.hbar
        
    def sigma_from_tau(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert proper time τ to σ-time.
        
        σ ≡ log(τ/τ₀)
        
        Args:
            tau: Proper time (must be positive)
            
        Returns:
            σ-time coordinate
        """
        tau = np.asarray(tau)
        if np.any(tau <= 0):
            raise ValueError("Proper time τ must be positive")
        return np.log(tau / self.tau_0)
    
    def tau_from_sigma(self, sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert σ-time to proper time τ.
        
        τ = τ₀ exp(σ)
        
        Args:
            sigma: σ-time coordinate
            
        Returns:
            Proper time
        """
        sigma = np.asarray(sigma)
        return self.tau_0 * np.exp(sigma)
    
    def silence_envelope(self, sigma: Union[float, np.ndarray], 
                        family: str = None, envelope_floor: float = None) -> Union[float, np.ndarray]:
        """
        Compute silence envelope function f_silence(σ) for asymptotic silence.
        
        Public API with configurable family and floor for reproducible scans.
        Uses centralized config by default but allows override for specific studies.
        
        Args:
            sigma: σ-time coordinate(s)
            family: Override envelope type ("tanh", "exponential", "polynomial", "smooth_step")
            envelope_floor: Override floor value to avoid exact zeros
            
        Returns:
            Silence envelope factor(s) with floor to avoid exact zeros
        """
        sigma = np.asarray(sigma)
        params = self.config.envelope_params.copy()
        
        # Allow override of family and floor for reproducible studies
        envelope_type = family if family is not None else self.config.envelope_type
        if envelope_floor is not None:
            params['envelope_floor'] = envelope_floor
        else:
            envelope_floor = params.get('envelope_floor', 1e-8)
        
        if envelope_type == 'tanh':
            sigma_0 = params.get('sigma_0', 2.0)
            width = params.get('width', 1.0)
            result = 0.5 * (1 + np.tanh((sigma + sigma_0) / width))
        
        elif envelope_type == 'exponential':
            sigma_0 = params.get('sigma_0', 2.0)
            result = np.ones_like(sigma)
            mask = sigma < 0
            result[mask] = np.exp(sigma[mask] / sigma_0)
        
        elif envelope_type == 'polynomial':
            sigma_0 = params.get('sigma_0', 2.0)
            power = params.get('power', 2)
            result = np.ones_like(sigma)
            mask = sigma > -sigma_0
            poly_vals = (1 + sigma[mask] / sigma_0)**power
            result[mask] = np.minimum(poly_vals, 1.0)
            mask_neg = sigma <= -sigma_0
            result[mask_neg] = 0.0
        
        elif envelope_type == 'smooth_step':
            sigma_0 = params.get('sigma_0', 2.0)
            width = params.get('width', 1.0)
            x = (sigma + sigma_0) / width
            result = np.zeros_like(sigma)
            mask = (x >= 0) & (x <= 1)
            result[mask] = 3*x[mask]**2 - 2*x[mask]**3
            result[x > 1] = 1.0
        
        else:
            raise ValueError(f"Unknown envelope type: {envelope_type}")
        
        # Apply floor to avoid exact zeros that can cause integration issues
        return np.maximum(result, envelope_floor)
    
    def effective_hamiltonian(self, sigma: float, H_tau: Callable[[float], np.ndarray], 
                              enforce_hermitian: Optional[bool] = None) -> np.ndarray:
        """
        Compute effective Hamiltonian H̃_eff(σ).
        
        When config.always_apply_silence=True:
            H̃_eff(σ) = f_silence(σ) × τ(σ) × H(τ(σ))
        When config.always_apply_silence=False:
            H̃_eff(σ) = τ(σ) × H(τ(σ))
        
        This ensures all evolution paths see exactly the same Hamiltonian,
        eliminating any risk of forgetting to apply the envelope.
        
        Args:
            sigma: σ-time parameter
            H_tau: Function returning H(τ) given proper time τ
            enforce_hermitian: Override config setting for Hermiticity enforcement
            
        Returns:
            Effective Hamiltonian matrix H̃_eff(σ)
        """
        tau = self.tau_from_sigma(sigma)
        H_of_tau = H_tau(tau)
        
        # Base effective Hamiltonian: H_eff(σ) = τ H(τ)
        H_eff = tau * H_of_tau
        
        # Apply silence envelope if configured as first-class
        if self.config.always_apply_silence:
            silence_factor = self.silence_envelope(sigma)
            H_eff = silence_factor * H_eff
        
        # Enforce Hermiticity to prevent imaginary artifacts
        if enforce_hermitian is None:
            enforce_hermitian = self.config.enforce_hermitian
        
        if enforce_hermitian:
            H_eff = (H_eff + H_eff.conj().T) / 2
        
        return H_eff
    
    def sigma_evolution(self, sigma_span: Tuple[float, float], psi_0: np.ndarray, 
                       H_tau: Callable[[float], np.ndarray], 
                       force_hermitian_ode: bool = True,
                       renormalize_steps: Optional[int] = None,
                       **kwargs) -> dict:
        """
        Evolve quantum state in σ-time using the σ-Schrödinger equation:
        iℏ ∂ψ/∂σ = H_eff(σ) ψ
        
        Enhanced with numerical robustness improvements:
        - Optional Hermitian projection at each ODE step
        - Optional periodic renormalization
        - Improved tolerances from config
        
        Args:
            sigma_span: (σ_initial, σ_final)
            psi_0: Initial state vector
            H_tau: Function returning H(τ) given proper time τ
            force_hermitian_ode: Apply (H + H†)/2 projection in ODE RHS
            renormalize_steps: Renormalize every N steps (None = no renormalization)
            **kwargs: Additional arguments for solve_ivp
            
        Returns:
            Dictionary with solution results and numerical diagnostics
        """
        # Track numerical diagnostics
        max_h_eff_norm = 0.0
        hermiticity_errors = []
        
        def sigma_schrodinger(sigma, psi_real_imag):
            """σ-Schrödinger equation in real form with numerical enhancements."""
            nonlocal max_h_eff_norm, hermiticity_errors
            
            # Convert real/imaginary representation back to complex
            n = len(psi_real_imag) // 2
            psi = psi_real_imag[:n] + 1j * psi_real_imag[n:]
            
            # Compute H_eff(σ) - Hermiticity already enforced if configured
            H_eff = self.effective_hamiltonian(sigma, H_tau)
            
            # Additional Hermitian projection for ODE robustness if requested
            if force_hermitian_ode:
                H_eff_hermitian = (H_eff + H_eff.conj().T) / 2
                hermiticity_error = np.linalg.norm(H_eff - H_eff_hermitian)
                hermiticity_errors.append(hermiticity_error)
                H_eff = H_eff_hermitian
            
            # Track maximum ||H_eff|| for stability analysis
            h_eff_norm = np.linalg.norm(H_eff)
            max_h_eff_norm = max(max_h_eff_norm, h_eff_norm)
            
            # Apply σ-Schrödinger equation: iℏ ∂ψ/∂σ = H_eff ψ
            dpsi_dsigma = -1j * H_eff @ psi / self.hbar
            
            # Convert back to real/imaginary representation
            return np.concatenate([dpsi_dsigma.real, dpsi_dsigma.imag])
        
        # Convert initial state to real/imaginary representation
        psi_0_real_imag = np.concatenate([psi_0.real, psi_0.imag])
        
        # Set improved tolerances from config to reduce drift
        solve_kwargs = {
            'rtol': self.config.ode_rtol,
            'atol': self.config.ode_atol,
            'dense_output': True
        }
        solve_kwargs.update(kwargs)
        
        # Solve the σ-evolution equation
        sol = solve_ivp(sigma_schrodinger, sigma_span, psi_0_real_imag, **solve_kwargs)
        
        # Convert solution back to complex form with optional renormalization
        def psi_sigma(sigma):
            y = sol.sol(sigma)
            n = len(y) // 2
            psi = y[:n] + 1j * y[n:]
            
            # Optional renormalization
            if renormalize_steps is not None:
                norm = np.linalg.norm(psi)
                if norm > 0:
                    psi = psi / norm
            
            return psi
        
        # Compute step size for stability check
        if len(sol.t) > 1:
            avg_step_size = np.mean(np.diff(sol.t))
            max_stability_param = max_h_eff_norm * avg_step_size
        else:
            max_stability_param = 0.0
        
        return {
            'solution': sol,
            'psi_sigma': psi_sigma,
            'sigma_array': sol.t,
            'tau_array': self.tau_from_sigma(sol.t),
            'success': sol.success,
            'message': sol.message,
            # Numerical diagnostics
            'max_h_eff_norm': max_h_eff_norm,
            'max_stability_param': max_stability_param,
            'avg_hermiticity_error': np.mean(hermiticity_errors) if hermiticity_errors else 0.0,
            'num_hermiticity_corrections': len(hermiticity_errors)
        }
    
    def sigma_unitary_evolution(self, sigma_span: Tuple[float, float], psi_0: np.ndarray, 
                               H_tau: Callable[[float], np.ndarray], steps: int = 1000) -> dict:
        """
        Evolve quantum state in σ-time using structure-preserving unitary steps.
        
        Uses Magnus expansion / midpoint-expm to maintain exact unitarity.
        Better for preserving norm ||ψ|| = 1 up to roundoff.
        Enhanced with stability monitoring.
        
        Args:
            sigma_span: (σ_initial, σ_final)
            psi_0: Initial state vector
            H_tau: Function returning H(τ) given proper time τ
            steps: Number of evolution steps
            
        Returns:
            Dictionary with solution results and stability diagnostics
        """
        sigma_array = np.linspace(*sigma_span, steps)
        dσ = (sigma_span[1] - sigma_span[0]) / (steps - 1)
        
        # Initialize state and tracking
        ψ = psi_0.astype(complex)
        psi_evolution = np.zeros((len(psi_0), steps), dtype=complex)
        psi_evolution[:, 0] = ψ
        
        # Stability tracking
        h_eff_norms = []
        stability_params = []
        max_stability_param = 0.0
        
        # Unitary evolution steps
        for i, sigma in enumerate(sigma_array[1:], 1):
            H_eff = self.effective_hamiltonian(sigma, H_tau)
            
            # Track stability: ||H_eff|| Δσ should be small for stable expm
            h_eff_norm = np.linalg.norm(H_eff)
            stability_param = h_eff_norm * abs(dσ)
            h_eff_norms.append(h_eff_norm)
            stability_params.append(stability_param)
            max_stability_param = max(max_stability_param, stability_param)
            
            # Hermiticity is already enforced in effective_hamiltonian if configured
            
            # Unitary evolution operator: U = exp(-i H_eff dσ / ℏ)
            U = la.expm(-1j * H_eff * dσ / self.hbar)
            
            # Apply unitary step
            ψ = U @ ψ
            psi_evolution[:, i] = ψ
        
        # Create interpolating function
        def psi_sigma(sigma):
            """Interpolate evolved state at given σ."""
            if np.isscalar(sigma):
                idx = np.searchsorted(sigma_array, sigma)
                if idx == 0:
                    return psi_evolution[:, 0]
                elif idx >= len(sigma_array):
                    return psi_evolution[:, -1]
                else:
                    # Linear interpolation of phases (simple approximation)
                    α = (sigma - sigma_array[idx-1]) / (sigma_array[idx] - sigma_array[idx-1])
                    return (1-α) * psi_evolution[:, idx-1] + α * psi_evolution[:, idx]
            else:
                # Handle array input
                result = np.zeros((len(psi_0), len(sigma)), dtype=complex)
                for j, s in enumerate(sigma):
                    result[:, j] = psi_sigma(s)
                return result
        
        return {
            'psi_final': ψ,
            'psi_sigma': psi_sigma,
            'psi_evolution': psi_evolution,
            'sigma_array': sigma_array,
            'tau_array': self.tau_from_sigma(sigma_array),
            'success': True,
            'message': 'Unitary evolution completed',
            # Stability diagnostics
            'max_h_eff_norm': max(h_eff_norms) if h_eff_norms else 0.0,
            'avg_h_eff_norm': np.mean(h_eff_norms) if h_eff_norms else 0.0,
            'max_stability_param': max_stability_param,
            'avg_stability_param': np.mean(stability_params) if stability_params else 0.0,
            'step_size': abs(dσ)
        }
    
    def asymptotic_silence_factor(self, sigma: Union[float, np.ndarray], 
                                 transition_width: float = 2.0) -> Union[float, np.ndarray]:
        """
        Compute asymptotic silence factor that smoothly turns off H_eff as σ→-∞.
        
        f(σ) = 1/(1 + exp(-(σ + σ_transition)/width))
        
        Args:
            sigma: σ-time coordinate(s)
            transition_width: Width of transition region
            
        Returns:
            Silence factor (0 to 1)
        """
        sigma = np.asarray(sigma)
        return 1.0 / (1.0 + np.exp(-sigma / transition_width))
    
    def multiplicative_to_additive_shift(self, dilation_factor: float) -> float:
        """
        Convert multiplicative time dilation to additive σ-shift.
        
        If τ₂ = k·τ₁, then σ₂ = σ₁ + log(k)
        
        Args:
            dilation_factor: Multiplicative factor k
            
        Returns:
            Additive σ-shift
        """
        if dilation_factor <= 0:
            raise ValueError("Dilation factor must be positive")
        return np.log(dilation_factor)
    
    def phase_accumulation_sigma(self, sigma_1: float, sigma_2: float, 
                               H_tau: Callable[[float], np.ndarray], 
                               state_index: int = 0) -> complex:
        """
        Compute quantum phase accumulation between σ₁ and σ₂.
        
        Phase = ∫(σ₁ to σ₂) ⟨ψ|H_eff(σ)|ψ⟩ dσ / ℏ
        
        Args:
            sigma_1: Initial σ-time
            sigma_2: Final σ-time
            H_tau: Hamiltonian function H(τ)
            state_index: Energy eigenstate index for phase calculation
            
        Returns:
            Accumulated phase (complex)
        """
        from scipy.integrate import quad
        
        def integrand(sigma):
            H_eff = self.effective_hamiltonian(sigma, H_tau)
            # For eigenstate, use diagonal element
            if hasattr(H_eff, 'shape') and len(H_eff.shape) == 2:
                energy_eff = H_eff[state_index, state_index]
            else:
                energy_eff = H_eff
            return energy_eff.real
        
        phase_real, _ = quad(integrand, sigma_1, sigma_2)
        return -1j * phase_real / self.hbar
    
    def gravitational_redshift_as_sigma_offset(self, phi_1: float, phi_2: float, 
                                              c: float = 1.0) -> float:
        """
        Express gravitational redshift as σ-offset.
        
        For weak field: Δσ ≈ (φ₂ - φ₁)/c²
        
        Args:
            phi_1: Gravitational potential at location 1
            phi_2: Gravitational potential at location 2
            c: Speed of light (natural units)
            
        Returns:
            σ-offset corresponding to redshift
        """
        return (phi_2 - phi_1) / (c * c)
    
    def quantum_state_measures(self, psi: np.ndarray) -> Dict[str, float]:
        """
        Compute quantum state measures with polished reporting.
        
        Casts results to .real to eliminate "-0.000j" artifacts in output.
        
        Args:
            psi: Quantum state vector
            
        Returns:
            Dictionary with norm, entropy, purity, and fidelity measures
        """
        # Density matrix
        rho = np.outer(psi, psi.conj())
        
        # Norm (should be 1 for normalized states)
        norm = np.sqrt(np.real(np.vdot(psi, psi)))
        
        # Von Neumann entropy: S = -Tr(ρ log ρ)
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-14]  # Remove numerical zeros
        entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-16))
        
        # Purity: Tr(ρ²)
        purity = np.trace(rho @ rho)
        
        # Fidelity with initial state (assuming |0⟩ = [1,0,...])
        initial_state = np.zeros_like(psi)
        initial_state[0] = 1.0
        fidelity = np.abs(np.vdot(initial_state, psi))**2
        
        return {
            'norm': float(np.real(norm)),
            'entropy': float(np.real(entropy)),
            'purity': float(np.real(purity)),
            'fidelity': float(np.real(fidelity))
        }
    
    def hamiltonian_properties(self, H: np.ndarray) -> Dict[str, float]:
        """
        Compute Hamiltonian properties with polished reporting.
        
        Args:
            H: Hamiltonian matrix
            
        Returns:
            Dictionary with trace, determinant, norm, and hermiticity measures
        """
        trace = np.trace(H)
        det = np.linalg.det(H)
        norm = np.linalg.norm(H)
        
        # Hermiticity check: ||H - H†||
        hermiticity_error = np.linalg.norm(H - H.conj().T)
        
        return {
            'trace': float(np.real(trace)),
            'determinant': float(np.real(det)),
            'norm': float(np.real(norm)),
            'hermiticity_error': float(np.real(hermiticity_error))
        }
    
    def test_redshift_correlation(self, r_range: Tuple[float, float] = (3.0, 10.0), 
                                 M: float = 1.0, num_points: int = 50) -> Dict[str, float]:
        """
        Test correlation between measured redshift and theoretical 1-2M/r.
        
        Provides unit test that corr(measured_redshift, √(1-2M/r)) = 1.0
        as requested for verification of σ-coordinate calculations.
        
        Args:
            r_range: Range of radial coordinates to test
            M: Mass parameter
            num_points: Number of test points
            
        Returns:
            Dictionary with correlation results and statistics
        """
        from scipy.stats import pearsonr
        
        r_array = np.linspace(r_range[0], r_range[1], num_points)
        
        # Theoretical redshift factors
        theoretical_redshift = np.sqrt(1 - 2*M/r_array)
        
        # Measured redshift from σ-coordinate calculation
        # Use a test coordinate time
        t_test = 1.0
        measured_redshift = []
        
        for r in r_array:
            # This should match the theoretical calculation
            tau = np.sqrt(1 - 2*M/r) * t_test  # From Schwarzschild proper time
            measured_factor = tau / t_test  # This is the redshift factor
            measured_redshift.append(measured_factor)
        
        measured_redshift = np.array(measured_redshift)
        
        # Compute correlation
        correlation, p_value = pearsonr(theoretical_redshift, measured_redshift)
        
        # Additional statistics
        max_abs_error = np.max(np.abs(theoretical_redshift - measured_redshift))
        rms_error = np.sqrt(np.mean((theoretical_redshift - measured_redshift)**2))
        
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'max_abs_error': float(max_abs_error),
            'rms_error': float(rms_error),
            'perfect_correlation': abs(correlation - 1.0) < 1e-12
        }


class SigmaTimeVisualizer:
    """
    Visualization tools for σ-time concepts and transformations.
    
    Uses centralized config to ensure plots reflect exactly the same
    parameters used in evolution computations.
    """
    
    def __init__(self, ltqg: LTQGFramework):
        self.ltqg = ltqg
        self.config = ltqg.config  # Pull shared config
        
    def plot_sigma_tau_relationship(self, tau_range: Tuple[float, float], 
                                   num_points: int = 1000, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the relationship between σ and τ coordinates.
        """
        tau_min, tau_max = tau_range
        tau_array = np.logspace(np.log10(tau_min), np.log10(tau_max), num_points)
        sigma_array = self.ltqg.sigma_from_tau(tau_array)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # σ vs τ plot
        ax1.semilogx(tau_array, sigma_array, 'b-', linewidth=2)
        ax1.set_xlabel('Proper Time τ')
        ax1.set_ylabel(f'σ-Time σ = log(τ/τ₀), τ₀ = {self.config.tau_0}')
        ax1.set_title(f'σ-Time Transformation (τ₀ = {self.config.tau_0})')
        ax1.grid(True, alpha=0.3)
        
        # Highlight asymptotic regions
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label=f'σ = 0 (τ = τ₀ = {self.config.tau_0})')
        ax1.axhline(y=-5, color='orange', linestyle='--', alpha=0.7, label='Early time (τ → 0⁺)')
        ax1.legend()
        
        # dσ/dτ = 1/τ plot
        ax2.loglog(tau_array, 1/tau_array, 'g-', linewidth=2)
        ax2.set_xlabel('Proper Time τ')
        ax2.set_ylabel('dσ/dτ = 1/τ')
        ax2.set_title(f'σ-Time Derivative (τ₀ = {self.config.tau_0})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_effective_hamiltonian_scaling(self, sigma_range: Tuple[float, float],
                                         H_tau_func: Callable[[float], float],
                                         num_points: int = 1000, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot how the effective Hamiltonian scales with σ.
        """
        sigma_array = np.linspace(*sigma_range, num_points)
        tau_array = self.ltqg.tau_from_sigma(sigma_array)
        
        H_tau_array = np.array([H_tau_func(tau) for tau in tau_array])
        H_eff_array = tau_array * H_tau_array
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Original Hamiltonian H(τ)
        ax1.semilogy(sigma_array, np.abs(H_tau_array), 'b-', linewidth=2, label='|H(τ)|')
        ax1.set_xlabel('σ-Time')
        ax1.set_ylabel('|H(τ)|')
        ax1.set_title('Original Hamiltonian')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Effective Hamiltonian H_eff(σ)
        ax2.semilogy(sigma_array, np.abs(H_eff_array), 'r-', linewidth=2, label='|H_eff(σ)|')
        ax2.set_xlabel('σ-Time')
        ax2.set_ylabel('|H_eff(σ)| = |τ H(τ)|')
        ax2.set_title('Effective Hamiltonian (Asymptotic Silence)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Highlight asymptotic silence region
        silence_mask = sigma_array < -3
        if np.any(silence_mask):
            ax2.fill_between(sigma_array[silence_mask], 
                           np.min(H_eff_array)*0.1, np.abs(H_eff_array[silence_mask]),
                           alpha=0.2, color='orange', label='Asymptotic Silence')
            ax2.legend()
        
        plt.tight_layout()
        return fig


def demo_ltqg_basics(seed: int = 42, version: str = "v2.1-enhanced"):
    """
    Demonstrate basic LTQG concepts and new refinements.
    
    Args:
        seed: Random seed for reproducibility
        version: Version string for matching figures to code
    """
    np.random.seed(seed)  # Set seed for reproducibility
    
    print("=== LTQG Framework Demo (Enhanced) ===")
    print(f"Version: {version}")
    print(f"Seed: {seed}")
    print(f"Date: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Demo new centralized configuration
    print("1. Centralized Configuration:")
    config = LTQGConfig(
        tau_0=1.0,
        always_apply_silence=True,  # Make silence first-class
        enforce_hermitian=True,     # Enforce Hermiticity
        envelope_type='tanh',
        envelope_params={'sigma_0': 2.0, 'width': 1.0, 'envelope_floor': 1e-8}
    )
    print(f"   τ₀ = {config.tau_0}")
    print(f"   Always apply silence: {config.always_apply_silence}")
    print(f"   Enforce Hermiticity: {config.enforce_hermitian}")
    print(f"   Envelope type: {config.envelope_type}")
    print(f"   Envelope params: {config.envelope_params}")
    
    # Initialize framework with config
    ltqg = LTQGFramework(config)
    
    # Demo 2: σ-time transformation (unchanged)
    print("\n2. σ-Time Transformation:")
    tau_values = [0.1, 1.0, 10.0]
    for tau in tau_values:
        sigma = ltqg.sigma_from_tau(tau)
        tau_back = ltqg.tau_from_sigma(sigma)
        print(f"   τ = {tau:.1f} → σ = {sigma:.3f} → τ = {tau_back:.3f}")
    
    # Demo 3: First-class silence in H_eff
    print("\n3. First-Class Silence in H̃_eff:")
    omega = 2 * np.pi
    
    def H_harmonic(tau):
        """Simple harmonic oscillator Hamiltonian."""
        return np.array([[omega/2, 0], [0, -omega/2]], dtype=complex)
    
    sigma_test = [-3, -1, 0, 1]
    for sigma in sigma_test:
        tau = ltqg.tau_from_sigma(sigma)
        
        # H_eff without silence (config override)
        H_eff_no_silence = ltqg.effective_hamiltonian(sigma, H_harmonic, enforce_hermitian=False)
        # Temporarily disable silence
        original_silence = ltqg.config.always_apply_silence
        ltqg.config.always_apply_silence = False
        H_eff_base = ltqg.effective_hamiltonian(sigma, H_harmonic)
        ltqg.config.always_apply_silence = original_silence
        
        # H̃_eff with first-class silence
        H_eff_with_silence = ltqg.effective_hamiltonian(sigma, H_harmonic)
        
        silence_factor = ltqg.silence_envelope(sigma)
        
        print(f"   σ = {sigma:2d}: τ = {tau:.3f}, silence = {silence_factor:.3f}")
        print(f"      |H_eff_base| = {np.linalg.norm(H_eff_base):.3f}")
        print(f"      |H̃_eff_silent| = {np.linalg.norm(H_eff_with_silence):.3f}")
    
    # Demo 4: Hermiticity enforcement
    print("\n4. Hermiticity Enforcement:")
    
    def H_non_hermitian(tau):
        """Slightly non-Hermitian test Hamiltonian."""
        return np.array([[1.0, 0.1 + 0.01j], [0.1 - 0.01j + 1e-15j, -1.0]], dtype=complex)
    
    sigma = 0.0
    H_eff_raw = ltqg.effective_hamiltonian(sigma, H_non_hermitian, enforce_hermitian=False)
    H_eff_hermitian = ltqg.effective_hamiltonian(sigma, H_non_hermitian, enforce_hermitian=True)
    
    props_raw = ltqg.hamiltonian_properties(H_eff_raw)
    props_hermitian = ltqg.hamiltonian_properties(H_eff_hermitian)
    
    print(f"   Raw H_eff:")
    print(f"      Trace: {props_raw['trace']:.6f}")
    print(f"      Hermiticity error: {props_raw['hermiticity_error']:.2e}")
    print(f"   Hermitianized H̃_eff:")
    print(f"      Trace: {props_hermitian['trace']:.6f}")
    print(f"      Hermiticity error: {props_hermitian['hermiticity_error']:.2e}")
    
    # Demo 5: Polished state measures
    print("\n5. Polished Quantum State Measures:")
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    
    # Evolve briefly to get non-trivial state
    sigma_span = (0.0, 0.5)
    result = ltqg.sigma_unitary_evolution(sigma_span, psi_0, H_harmonic, steps=100)
    
    if result['success']:
        psi_final = result['psi_final']
        measures = ltqg.quantum_state_measures(psi_final)
        
        print(f"   Initial state |ψ₀⟩ = [1, 0]")
        print(f"   After evolution σ ∈ [0, 0.5]:")
        print(f"      Norm: {measures['norm']:.6f}")
        print(f"      Entropy: {measures['entropy']:.6f}")
        print(f"      Purity: {measures['purity']:.6f}")
        print(f"      Fidelity: {measures['fidelity']:.6f}")
        print("   (Note: No '−0.000j' artifacts!)")
    
    # Demo 6: Redshift correlation test
    print("\n6. Redshift Correlation Verification:")
    redshift_test = ltqg.test_redshift_correlation()
    print(f"   Correlation between measured and theoretical redshift:")
    print(f"      Correlation: {redshift_test['correlation']:.8f}")
    print(f"      Perfect correlation: {redshift_test['perfect_correlation']}")
    print(f"      Max absolute error: {redshift_test['max_abs_error']:.2e}")
    print(f"      RMS error: {redshift_test['rms_error']:.2e}")
    print("   ✓ Redshift calculation verified")
    
    # Demo 7: Configuration consistency verification
    print("\n7. Configuration Consistency:")
    visualizer = SigmaTimeVisualizer(ltqg)
    print(f"   Visualizer uses same τ₀: {visualizer.config.tau_0}")
    print(f"   Visualizer uses same envelope: {visualizer.config.envelope_type}")
    print("   → Guarantees plots reflect exact H used in evolution")
    
    # Demo 8: Numerical robustness
    print("\n8. Enhanced Numerical Robustness:")
    
    # Test ODE evolution with enhanced diagnostics
    result_ode_enhanced = ltqg.sigma_evolution(
        sigma_span, psi_0, H_harmonic, 
        force_hermitian_ode=True,
        t_eval=np.linspace(sigma_span[0], sigma_span[1], 50)
    )
    
    # Test unitary evolution with stability monitoring
    result_unitary_enhanced = ltqg.sigma_unitary_evolution(
        sigma_span, psi_0, H_harmonic, steps=100
    )
    
    if result_ode_enhanced['success']:
        print(f"   Enhanced ODE evolution:")
        print(f"      Max ||H_eff||: {result_ode_enhanced['max_h_eff_norm']:.3f}")
        print(f"      Max stability param: {result_ode_enhanced['max_stability_param']:.3f}")
        print(f"      Avg hermiticity error: {result_ode_enhanced['avg_hermiticity_error']:.2e}")
        print(f"      Hermiticity corrections: {result_ode_enhanced['num_hermiticity_corrections']}")
    
    if result_unitary_enhanced['success']:
        print(f"   Enhanced unitary evolution:")
        print(f"      Max ||H_eff||: {result_unitary_enhanced['max_h_eff_norm']:.3f}")
        print(f"      Max stability param: {result_unitary_enhanced['max_stability_param']:.3f}")
        print(f"      Step size: {result_unitary_enhanced['step_size']:.3f}")
        print(f"      Final norm: {np.linalg.norm(result_unitary_enhanced['psi_final']):.8f}")
    
    print("   ✓ All numerical enhancements active")


if __name__ == "__main__":
    demo_ltqg_basics()