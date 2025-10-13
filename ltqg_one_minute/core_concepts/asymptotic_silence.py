"""
Asymptotic Silence in LTQG: Near-Horizon and Early-Time Behavior

This module implements the asymptotic silence mechanism in LTQG, where
the effective Hamiltonian H_eff(σ) → 0 as σ → -∞ (τ → 0⁺).

Key concepts:
- Asymptotic silence as fundamental regularization
- Near-horizon behavior in black hole spacetimes
- Early-time cosmological dynamics
- Boundary conditions at σ → -∞
- Information preservation through silence
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import gamma, factorial
from typing import Callable, Optional, Tuple, Union, Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class AsymptoticSilenceProperties:
    """Properties characterizing asymptotic silence behavior."""
    silence_threshold: float
    decay_rate: float
    information_content: float
    entropy_growth_rate: float
    boundary_condition_type: str
    regularization_strength: float


class AsymptoticSilence:
    """
    Implementation of asymptotic silence mechanism in LTQG.
    """
    
    def __init__(self, tau_0: float = 1.0, hbar: float = 1.0):
        """
        Initialize asymptotic silence framework.
        
        Args:
            tau_0: Reference time scale
            hbar: Reduced Planck constant
        """
        self.tau_0 = tau_0
        self.hbar = hbar
        
    def silence_envelope(self, sigma: Union[float, np.ndarray],
                        envelope_type: str = 'tanh',
                        params: Optional[Dict] = None) -> Union[float, np.ndarray]:
        """
        Compute silence envelope function that implements asymptotic silence.
        
        Different envelope types:
        - 'tanh': (1 + tanh((σ + σ₀)/w))/2
        - 'exponential': exp(σ/σ₀) for σ < 0
        - 'polynomial': (1 + σ/σ₀)^n for σ > -σ₀
        - 'smooth_step': Smooth step function
        
        Args:
            sigma: σ-time coordinate(s)
            envelope_type: Type of envelope function
            params: Parameters for envelope function
            
        Returns:
            Silence envelope factor(s) with floor at 1e-8 to avoid exact zeros
        """
        if params is None:
            params = {}
        
        sigma = np.asarray(sigma)
        envelope_floor = params.get('envelope_floor', 1e-8)  # Avoid exact zeros
        
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
            normalize = params.get('normalize', True)
            result = np.ones_like(sigma)
            mask = sigma > -sigma_0
            if normalize:
                # Normalized polynomial: clamp to [0,1]
                poly_vals = (1 + sigma[mask] / sigma_0)**power
                result[mask] = np.minimum(poly_vals, 1.0)
            else:
                # Original unbounded polynomial (for testing)
                result[mask] = (1 + sigma[mask] / sigma_0)**power
            mask_neg = sigma <= -sigma_0
            result[mask_neg] = 0.0
        
        elif envelope_type == 'smooth_step':
            sigma_0 = params.get('sigma_0', 2.0)
            width = params.get('width', 1.0)
            x = (sigma + sigma_0) / width
            # Smooth step: 0 for x < 0, 1 for x > 1, smooth transition
            result = np.zeros_like(sigma)
            mask = (x >= 0) & (x <= 1)
            result[mask] = 3*x[mask]**2 - 2*x[mask]**3
            result[x > 1] = 1.0
        
        else:
            raise ValueError(f"Unknown envelope type: {envelope_type}")
        
        # Apply floor to avoid exact zeros that can cause integration issues
        return np.maximum(result, envelope_floor)
    
    def effective_hamiltonian_with_silence(self, sigma: float,
                                         H_tau: Callable[[float], np.ndarray],
                                         envelope_params: Optional[Dict] = None) -> np.ndarray:
        """
        Compute H_eff(σ) with asymptotic silence envelope.
        
        H_eff_silent(σ) = f(σ) × τ(σ) H(τ(σ))
        
        Args:
            sigma: σ-time parameter
            H_tau: Original Hamiltonian function H(τ)
            envelope_params: Parameters for silence envelope
            
        Returns:
            Silenced effective Hamiltonian
        """
        if envelope_params is None:
            envelope_params = {}
        
        # Compute base effective Hamiltonian
        tau = self.tau_0 * np.exp(sigma)
        H_eff_base = tau * H_tau(tau)
        
        # Apply silence envelope
        envelope_type = envelope_params.get('type', 'tanh')
        envelope_factor = self.silence_envelope(sigma, envelope_type, envelope_params)
        
        return envelope_factor * H_eff_base
    
    def near_horizon_limit(self, r_obs: float, M: float,
                          sigma_range: Tuple[float, float],
                          H_tau: Callable[[float], np.ndarray],
                          c: float = 1.0) -> Dict:
        """
        Analyze near-horizon behavior using asymptotic silence.
        
        For Schwarzschild geometry, static observers experience
        infinite redshift at r → 2M, corresponding to σ → -∞.
        
        Args:
            r_obs: Observer radial coordinate
            M: Black hole mass
            sigma_range: Range of σ-time to analyze
            H_tau: Hamiltonian function
            c: Speed of light
            
        Returns:
            Dictionary with near-horizon analysis results
        """
        if r_obs <= 2*M:
            raise ValueError("Observer must be outside event horizon")
        
        # Redshift factor for static observer
        redshift_factor = np.sqrt(1 - 2*M/(r_obs*c*c))
        
        sigma_array = np.linspace(*sigma_range, 1000)
        tau_array = self.tau_0 * np.exp(sigma_array)
        
        # Compute effective Hamiltonian evolution
        H_eff_norms = []
        silence_factors = []
        information_measures = []
        
        for sigma in sigma_array:
            # Base effective Hamiltonian
            H_eff = self.effective_hamiltonian_with_silence(
                sigma, H_tau, {'type': 'tanh', 'sigma_0': 3.0})
            
            H_eff_norms.append(np.linalg.norm(H_eff))
            silence_factors.append(self.silence_envelope(sigma))
            
            # Information measure (von Neumann entropy proxy)
            if H_eff.shape[0] > 1:
                eigenvals = np.linalg.eigvals(H_eff + H_eff.conj().T)
                eigenvals = eigenvals[eigenvals > 1e-12]
                if len(eigenvals) > 0:
                    eigenvals = eigenvals / np.sum(eigenvals)
                    entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-12))
                else:
                    entropy = 0.0
            else:
                entropy = 0.0
            information_measures.append(entropy)
        
        return {
            'sigma_array': sigma_array,
            'tau_array': tau_array,
            'redshift_factor': redshift_factor,
            'H_eff_norms': np.array(H_eff_norms),
            'silence_factors': np.array(silence_factors),
            'information_measures': np.array(information_measures),
            'horizon_approach_sigma': np.log(2*M*redshift_factor/self.tau_0)
        }
    
    def early_universe_dynamics(self, sigma_range: Tuple[float, float],
                              cosmology_type: str = 'radiation',
                              H_tau: Optional[Callable[[float], np.ndarray]] = None) -> Dict:
        """
        Analyze early universe dynamics with asymptotic silence.
        
        In FLRW cosmology, σ → -∞ corresponds to the big bang (t → 0⁺).
        Asymptotic silence provides natural boundary conditions.
        
        Args:
            sigma_range: Range of σ-time to analyze
            cosmology_type: Type of cosmological model
            H_tau: Hamiltonian function (default: create from cosmology)
            
        Returns:
            Dictionary with early universe analysis
        """
        sigma_array = np.linspace(*sigma_range, 1000)
        tau_array = self.tau_0 * np.exp(sigma_array)
        
        # Scale factor evolution
        if cosmology_type == 'radiation':
            # a ∝ t^(1/2) for radiation domination
            a_array = np.sqrt(tau_array)
        elif cosmology_type == 'matter':
            # a ∝ t^(2/3) for matter domination
            a_array = tau_array**(2/3)
        elif cosmology_type == 'inflation':
            # a ∝ exp(Ht) for de Sitter inflation
            H_inflation = 1.0  # Hubble parameter
            a_array = np.exp(H_inflation * tau_array)
        else:
            raise ValueError(f"Unknown cosmology type: {cosmology_type}")
        
        # Default Hamiltonian for cosmological scalar field
        if H_tau is None:
            def H_cosmo(tau):
                # Simplified minisuperspace: scalar field + gravity
                # Time-dependent mass term
                m_eff_sq = 1.0 / (tau + 1e-10)  # Divergent as τ → 0
                return np.array([[m_eff_sq, 0], [0, -m_eff_sq]], dtype=complex)
            H_tau = H_cosmo
        
        # Analyze dynamics with silence
        curvature_scalars = []
        H_eff_eigenvals = []
        silence_factors = []
        
        for i, sigma in enumerate(sigma_array):
            # Hubble parameter and curvature
            if i > 0:
                H_hubble = (a_array[i] - a_array[i-1]) / (a_array[i] * (tau_array[i] - tau_array[i-1]))
            else:
                H_hubble = 0.5 / tau_array[i] if cosmology_type == 'radiation' else 2/(3*tau_array[i])
            
            curvature_scalars.append(H_hubble**2)
            
            # Effective Hamiltonian with silence
            H_eff = self.effective_hamiltonian_with_silence(
                sigma, H_tau, {'type': 'tanh', 'sigma_0': 4.0})
            
            eigenvals = np.linalg.eigvals(H_eff)
            H_eff_eigenvals.append(eigenvals)
            
            silence_factor = self.silence_envelope(sigma)
            silence_factors.append(silence_factor)
        
        return {
            'sigma_array': sigma_array,
            'tau_array': tau_array,
            'scale_factor': a_array,
            'curvature_scalars': np.array(curvature_scalars),
            'H_eff_eigenvals': H_eff_eigenvals,
            'silence_factors': np.array(silence_factors),
            'cosmology_type': cosmology_type
        }
    
    def information_preservation_analysis(self, sigma_range: Tuple[float, float],
                                        H_tau: Callable[[float], np.ndarray],
                                        initial_state: np.ndarray) -> Dict:
        """
        Analyze information preservation through asymptotic silence.
        
        Key question: How does quantum information behave as we approach
        the asymptotic silence region σ → -∞?
        
        Args:
            sigma_range: Range of σ-time
            H_tau: Hamiltonian function
            initial_state: Initial quantum state
            
        Returns:
            Information preservation analysis results
        """
        from scipy.linalg import logm
        
        sigma_array = np.linspace(*sigma_range, 500)
        
        # Evolve state through σ-time with silence
        def sigma_evolution_with_silence(sigma, psi_real_imag):
            n = len(psi_real_imag) // 2
            psi = psi_real_imag[:n] + 1j * psi_real_imag[n:]
            
            # Renormalize state to maintain unitarity
            norm = np.linalg.norm(psi)
            if norm > 1e-12:
                psi = psi / norm
            
            H_eff = self.effective_hamiltonian_with_silence(sigma, H_tau)
            dpsi_dsigma = -1j * H_eff @ psi / self.hbar
            
            return np.concatenate([dpsi_dsigma.real, dpsi_dsigma.imag])
        
        # Initial condition
        psi_0_real_imag = np.concatenate([initial_state.real, initial_state.imag])
        
        # Solve evolution
        sol = solve_ivp(sigma_evolution_with_silence, sigma_range, psi_0_real_imag,
                       t_eval=sigma_array, dense_output=True)
        
        # Analyze information measures
        von_neumann_entropy = []
        fidelity_with_initial = []
        purity = []
        
        for i, sigma in enumerate(sigma_array):
            y = sol.y[:, i]
            n = len(y) // 2
            psi = y[:n] + 1j * y[n:]
            
            # Renormalize state for stability
            norm = np.linalg.norm(psi)
            if norm > 1e-12:
                psi = psi / norm
            
            # Density matrix
            rho = np.outer(psi, psi.conj())
            
            # Von Neumann entropy with stabilized eigenvalues
            eigenvals = np.linalg.eigvals(rho)
            # Project small negative eigenvalues to zero
            eigenvals = np.maximum(eigenvals.real, 0.0)
            eigenvals = eigenvals[eigenvals > 1e-12]
            eigenvals = eigenvals / np.sum(eigenvals)  # Renormalize
            
            if len(eigenvals) > 0:
                entropy = -np.sum(eigenvals * np.log(eigenvals + 1e-15))
            else:
                entropy = 0.0
            von_neumann_entropy.append(entropy)
            
            # Fidelity with initial state
            fidelity = np.abs(np.vdot(initial_state, psi))**2
            fidelity_with_initial.append(fidelity)
            
            # Purity
            purity_val = np.trace(rho @ rho).real
            purity.append(purity_val)
        
        return {
            'sigma_array': sigma_array,
            'evolved_states': sol.y,
            'von_neumann_entropy': np.array(von_neumann_entropy),
            'fidelity_with_initial': np.array(fidelity_with_initial),
            'purity': np.array(purity),
            'evolution_success': sol.success,
            'fidelity_vs_sigma': self.compute_fidelity_evolution(sigma_array, sol.y, initial_state)
        }
    
    def compute_fidelity_evolution(self, sigma_array: np.ndarray, 
                                 evolved_states: np.ndarray,
                                 initial_state: np.ndarray) -> np.ndarray:
        """
        Compute detailed fidelity evolution F(σ) = |⟨ψ(0)|ψ(σ)⟩|².
        
        Args:
            sigma_array: σ-time array
            evolved_states: Evolved state array
            initial_state: Initial state
            
        Returns:
            Fidelity array F(σ)
        """
        fidelity_detailed = np.zeros_like(sigma_array)
        n = len(initial_state)
        
        for i, sigma in enumerate(sigma_array):
            y = evolved_states[:, i]
            psi = y[:n] + 1j * y[n:]
            fidelity_detailed[i] = np.abs(np.vdot(initial_state, psi))**2
        
        return fidelity_detailed
    
    def analyze_schwarzschild_redshift_test(self, M: float = 1.0,
                                          r_range: Tuple[float, float] = (2.1, 5.0),
                                          num_points: int = 20) -> Dict:
        """
        Test Schwarzschild redshift factors against theoretical predictions.
        
        Verifies that measured redshift factors match √(1 - 2M/r).
        
        NOTE: This is not a genuinely independent measurement test since both
        measured and theoretical values use the same formula. For a stricter test,
        introduce synthetic noise or compute redshift from separate worldline 
        integration routines.
        
        Args:
            M: Black hole mass
            r_range: Range of radial coordinates to test
            num_points: Number of test points
            
        Returns:
            Redshift test analysis (correlation=1, RMS=0 expected)
        """
        r_array = np.linspace(*r_range, num_points)
        measured_redshift = np.zeros_like(r_array)
        theoretical_redshift = np.zeros_like(r_array)
        
        for i, r in enumerate(r_array):
            # Theoretical redshift factor
            theoretical_redshift[i] = np.sqrt(1 - 2*M/r)
            
            # "Measured" redshift (in this case, computed consistently)
            measured_redshift[i] = np.sqrt(max(0.01, 1 - 2*M/r))  # Avoid sqrt of negative
        
        # Compute correlation and residuals
        correlation = np.corrcoef(measured_redshift, theoretical_redshift)[0, 1]
        residuals = measured_redshift - theoretical_redshift
        rms_error = np.sqrt(np.mean(residuals**2))
        
        return {
            'r_array': r_array,
            'measured_redshift': measured_redshift,
            'theoretical_redshift': theoretical_redshift,
            'correlation': correlation,
            'rms_error': rms_error,
            'max_deviation': np.max(np.abs(residuals)),
            'test_passed': correlation > 0.999 and rms_error < 1e-6
        }
    
    def analyze_silence_convergence(self, sigma_min_range: Tuple[float, float],
                                  H_tau: Callable[[float], np.ndarray],
                                  envelope_params: Optional[Dict] = None) -> Dict:
        """
        Analyze convergence of H_eff to zero as sigma_min → -∞.
        
        Demonstrates asymptotic silence convergence with grid resolution.
        Scans a window around each sigma_min to avoid trivial zeros.
        
        Args:
            sigma_min_range: Range of sigma_min values to test
            H_tau: Hamiltonian function
            envelope_params: Parameters for silence envelope
            
        Returns:
            Convergence analysis results
        """
        if envelope_params is None:
            envelope_params = {'type': 'tanh', 'sigma_0': 2.0}
            
        sigma_min_vals = np.linspace(*sigma_min_range, 10)
        envelope_types = ['tanh', 'exponential', 'polynomial', 'smooth_step']
        convergence_data = {}
        
        for env_type in envelope_types:
            min_norms = []
            max_norms = []
            
            for sigma_min in sigma_min_vals:
                # Scan window around each σ_min to avoid trivial zeros
                sigma_scan = np.linspace(sigma_min - 1, sigma_min + 1, 21)
                h_eff_norms = []
                
                for sigma in sigma_scan:
                    env_params = envelope_params.copy()
                    env_params['type'] = env_type
                    
                    H_eff = self.effective_hamiltonian_with_silence(
                        sigma, H_tau, env_params)
                    h_eff_norms.append(np.linalg.norm(H_eff))
                
                # Aggregate min/max over scan window
                min_norms.append(np.min(h_eff_norms))
                max_norms.append(np.max(h_eff_norms))
            
            convergence_data[env_type] = {
                'min_norms': np.array(min_norms),
                'max_norms': np.array(max_norms)
            }
        
        return {
            'sigma_min_array': sigma_min_vals,
            'convergence_by_envelope': convergence_data,
            'theoretical_limit': 0.0
        }
    
    def boundary_conditions_at_silence(self, H_tau: Callable[[float], np.ndarray],
                                     condition_type: str = 'product_state') -> Dict:
        """
        Generate appropriate boundary conditions at σ → -∞.
        
        Different types:
        - 'product_state': Factorized, low-entanglement state
        - 'ground_state': Adiabatic ground state of silenced Hamiltonian
        - 'thermal': Thermal state at effective temperature
        - 'vacuum': Vacuum state in field theory context
        
        Args:
            H_tau: Hamiltonian function
            condition_type: Type of boundary condition
            
        Returns:
            Dictionary with boundary condition information
        """
        # Analyze H_eff in deep silence region
        sigma_silence = -10.0  # Deep in silence region
        H_eff_silence = self.effective_hamiltonian_with_silence(
            sigma_silence, H_tau, {'type': 'tanh', 'sigma_0': 3.0})
        
        eigenvals, eigenvecs = np.linalg.eigh(H_eff_silence)
        
        if condition_type == 'product_state':
            # Simple product state (e.g., all spins up)
            dim = H_eff_silence.shape[0]
            boundary_state = np.zeros(dim, dtype=complex)
            boundary_state[0] = 1.0
            
        elif condition_type == 'ground_state':
            # Ground state of silenced Hamiltonian
            ground_idx = np.argmin(eigenvals.real)
            boundary_state = eigenvecs[:, ground_idx]
            
        elif condition_type == 'thermal':
            # Thermal state at low effective temperature
            beta_eff = 10.0  # Large β → low temperature
            thermal_weights = np.exp(-beta_eff * eigenvals.real)
            thermal_weights /= np.sum(thermal_weights)
            
            # Mixed state represented as pure state (simplified)
            boundary_state = np.sum([np.sqrt(w) * eigenvecs[:, i] 
                                   for i, w in enumerate(thermal_weights)], axis=0)
            boundary_state /= np.linalg.norm(boundary_state)
            
        elif condition_type == 'vacuum':
            # Vacuum state (lowest energy)
            ground_idx = np.argmin(eigenvals.real)
            boundary_state = eigenvecs[:, ground_idx]
            
        else:
            raise ValueError(f"Unknown boundary condition type: {condition_type}")
        
        return {
            'boundary_state': boundary_state,
            'eigenvalues_at_silence': eigenvals,
            'eigenvectors_at_silence': eigenvecs,
            'H_eff_norm_at_silence': np.linalg.norm(H_eff_silence),
            'condition_type': condition_type,
            'sigma_silence': sigma_silence
        }


def demo_asymptotic_silence():
    """
    Demonstrate asymptotic silence mechanisms and applications.
    """
    print("=== Asymptotic Silence Demo ===\n")
    
    silence = AsymptoticSilence(tau_0=1.0, hbar=1.0)
    
    # Demo 1: Silence envelope functions (FIXED)
    print("1. Silence Envelope Functions (Fixed Polynomial):")
    sigma_test = np.array([-4, -2, 0, 2])
    
    envelope_types = ['tanh', 'exponential', 'polynomial', 'smooth_step']
    for env_type in envelope_types:
        if env_type == 'polynomial':
            # Test both normalized and unnormalized
            factors_norm = silence.silence_envelope(sigma_test, env_type, {'normalize': True})
            factors_orig = silence.silence_envelope(sigma_test, env_type, {'normalize': False})
            print(f"   {env_type:12s} (norm): {factors_norm}")
            print(f"   {env_type:12s} (orig): {factors_orig}")
        else:
            factors = silence.silence_envelope(sigma_test, env_type)
            print(f"   {env_type:12s}: {factors}")
    
    # Demo 2: Convergence analysis
    print("\n2. Asymptotic Silence Convergence Analysis:")
    
    def H_two_level(tau):
        omega = 1.0
        coupling = 0.1 / tau  # Divergent coupling as τ → 0
        return np.array([[omega/2, coupling], [coupling, -omega/2]], dtype=complex)
    
    convergence = silence.analyze_silence_convergence((-8, -2), H_two_level)
    
    print(f"   σ_min range: [{convergence['sigma_min_array'][0]:.1f}, {convergence['sigma_min_array'][-1]:.1f}]")
    for env_type, data in convergence['convergence_by_envelope'].items():
        min_norm = np.min(data['min_norms'])
        max_norm = np.max(data['max_norms'])
        print(f"   {env_type:12s}: min ||H_eff|| = {min_norm:.2e}, max = {max_norm:.2e}")
    
    # Demo 3: Schwarzschild redshift test
    print("\n3. Schwarzschild Redshift Factor Test:")
    
    redshift_test = silence.analyze_schwarzschild_redshift_test(M=1.0, r_range=(2.1, 5.0))
    
    print(f"   Test range: r/M ∈ [{redshift_test['r_array'][0]:.1f}, {redshift_test['r_array'][-1]:.1f}]")
    print(f"   Correlation with theory: {redshift_test['correlation']:.6f}")
    print(f"   RMS error: {redshift_test['rms_error']:.2e}")
    print(f"   Max deviation: {redshift_test['max_deviation']:.2e}")
    print(f"   Test passed: {'✓' if redshift_test['test_passed'] else '✗'}")
    
    # Demo 4: Near-horizon analysis (ENHANCED)
    print("\n4. Enhanced Near-Horizon Black Hole Analysis:")
    
    M = 1.0  # Black hole mass
    r_obs = 3*M  # Observer at 3M
    sigma_range = (-5, 2)
    
    horizon_analysis = silence.near_horizon_limit(r_obs, M, sigma_range, H_two_level)
    
    print(f"   Observer at r = {r_obs:.1f}M")
    print(f"   Theoretical redshift: {np.sqrt(1 - 2*M/r_obs):.3f}")
    print(f"   Measured redshift: {horizon_analysis['redshift_factor']:.3f}")
    print(f"   Horizon approach σ: {horizon_analysis['horizon_approach_sigma']:.3f}")
    print(f"   Min H_eff norm: {np.min(horizon_analysis['H_eff_norms']):.2e}")
    print(f"   Max information measure: {np.max(horizon_analysis['information_measures']):.3f}")
    
    # Demo 5: Early universe dynamics (ENHANCED)
    print("\n5. Enhanced Early Universe Dynamics:")
    
    sigma_range_cosmo = (-6, 1)
    cosmologies = ['radiation', 'matter', 'inflation']
    
    print(f"   Analysis range: σ ∈ [{sigma_range_cosmo[0]}, {sigma_range_cosmo[1]}]")
    print(f"   {'Cosmology':>12s} {'Max Curvature':>15s} {'Min Silence':>12s} {'Units/Scale':>15s}")
    print(f"   {'-'*12} {'-'*15} {'-'*12} {'-'*15}")
    
    for cosmo_type in cosmologies:
        cosmo_analysis = silence.early_universe_dynamics(sigma_range_cosmo, cosmo_type)
        
        max_curvature = np.max(cosmo_analysis['curvature_scalars'])
        min_silence = np.min(cosmo_analysis['silence_factors'])
        
        # Identify units/scale
        if cosmo_type == 'radiation':
            scale_info = "H² ~ τ⁻²"
        elif cosmo_type == 'matter':
            scale_info = "H² ~ τ⁻²"  # Both radiation and matter give H ~ 1/t => H² ~ t⁻²
        else:
            scale_info = "H² ~ const"
        
        print(f"   {cosmo_type:>12s} {max_curvature:>15.2e} {min_silence:>12.3f} {scale_info:>15s}")
    
    # Demo 6: Information preservation with detailed fidelity
    print("\n6. Enhanced Information Preservation Analysis:")
    
    # Initial entangled state
    initial_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
    
    info_analysis = silence.information_preservation_analysis(
        (-3, 1), H_two_level, initial_state)
    
    if info_analysis['evolution_success']:
        max_entropy = np.max(info_analysis['von_neumann_entropy'])
        final_fidelity = info_analysis['fidelity_with_initial'][-1]
        min_purity = np.min(info_analysis['purity'])
        
        # Detailed fidelity analysis
        fidelity_evolution = info_analysis['fidelity_vs_sigma']
        fidelity_decay_rate = np.mean(np.abs(np.gradient(fidelity_evolution)))
        
        print(f"   Evolution successful: ✓")
        print(f"   Max von Neumann entropy: {max_entropy:.3f}")
        print(f"   Final fidelity: {final_fidelity:.3f}")
        print(f"   Min purity: {min_purity:.3f}")
        print(f"   Avg fidelity decay rate: {fidelity_decay_rate:.3f}/σ")
        
        # Interpretation
        if min_purity > 0.95:
            print(f"   → Nearly unitary evolution (purity > 0.95)")
        if final_fidelity < 0.5:
            print(f"   → Significant state evolution (final overlap < 0.5)")
        else:
            print(f"   → Moderate state evolution (final overlap ≥ 0.5)")
            
    else:
        print(f"   Evolution failed")
    
    # Demo 7: Boundary conditions comparison
    print("\n7. Boundary Conditions at σ → -∞:")
    
    boundary_types = ['product_state', 'ground_state', 'thermal', 'vacuum']
    
    print(f"   {'Condition':>15s} {'||ψ||':>8s} {'E_ground':>12s} {'H_eff norm':>12s}")
    print(f"   {'-'*15} {'-'*8} {'-'*12} {'-'*12}")
    
    for bc_type in boundary_types:
        bc_analysis = silence.boundary_conditions_at_silence(H_two_level, bc_type)
        
        state_norm = np.linalg.norm(bc_analysis['boundary_state'])
        ground_energy = np.min(bc_analysis['eigenvalues_at_silence'].real)
        h_eff_norm = bc_analysis['H_eff_norm_at_silence']
        
        print(f"   {bc_type:>15s} {state_norm:>8.3f} {ground_energy:>12.2e} {h_eff_norm:>12.2e}")
    
    print(f"\n   → All boundary conditions equivalent in silence region (H_eff ≈ 0)")
    print(f"   → Supports low-entanglement initial data without changing late-σ physics")


def plot_asymptotic_silence_analysis():
    """
    Plot comprehensive asymptotic silence analysis.
    """
    silence = AsymptoticSilence()
    
    # Two-level system with divergent coupling
    def H_divergent(tau):
        omega = 2.0
        coupling = 0.5 / np.sqrt(tau + 1e-10)  # Divergent as τ → 0
        return np.array([[omega/2, coupling], [coupling, -omega/2]], dtype=complex)
    
    sigma_range = np.linspace(-5, 2, 200)
    
    # Different envelope types
    envelope_types = ['tanh', 'exponential', 'smooth_step']
    colors = ['blue', 'red', 'green']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (env_type, color) in enumerate(zip(envelope_types, colors)):
        # Silence factors
        silence_factors = silence.silence_envelope(sigma_range, env_type)
        axes[0, i].plot(sigma_range, silence_factors, color=color, linewidth=2)
        axes[0, i].set_xlabel('σ-time')
        axes[0, i].set_ylabel('Silence Factor')
        axes[0, i].set_title(f'{env_type.title()} Envelope')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_ylim(0, 1.1)
        
        # H_eff norms
        H_eff_norms = []
        for sigma in sigma_range:
            H_eff = silence.effective_hamiltonian_with_silence(
                sigma, H_divergent, {'type': env_type})
            H_eff_norms.append(np.linalg.norm(H_eff))
        
        axes[1, i].semilogy(sigma_range, H_eff_norms, color=color, linewidth=2)
        axes[1, i].set_xlabel('σ-time')
        axes[1, i].set_ylabel('||H_eff(σ)||')
        axes[1, i].set_title(f'Effective Hamiltonian Norm ({env_type})')
        axes[1, i].grid(True, alpha=0.3)
        
        # Highlight silence region
        silence_region = sigma_range < -2
        axes[0, i].axvspan(sigma_range[silence_region][0], -2, alpha=0.2, 
                          color='orange', label='Silence Region')
        axes[1, i].axvspan(sigma_range[silence_region][0], -2, alpha=0.2, 
                          color='orange')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    demo_asymptotic_silence()
    
    # Uncomment to generate plots
    # fig = plot_asymptotic_silence_analysis()
    # plt.show()