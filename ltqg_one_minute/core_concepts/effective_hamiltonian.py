"""
Effective Hamiltonian in σ-Time: H_eff(σ) = τ H(τ)

This module implements the effective Hamiltonian formalism for LTQG,
showing how the σ-Schrödinger equation iℏ ∂ψ/∂σ = H_eff(σ) ψ
preserves standard unitary quantum mechanics while providing asymptotic silence.

Key concepts:
- H_eff(σ) = τ(σ) H(τ(σ)) construction
- Asymptotic silence as σ → -∞
- Continuity and boundedness properties
- Spectral analysis in σ-coordinates
"""

import numpy as np
import scipy.linalg as la
from scipy.integrate import solve_ivp
from typing import Callable, Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class EffectiveHamiltonianProperties:
    """Container for H_eff properties and analysis results."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_norm: float
    trace: complex
    determinant: complex
    condition_number: float
    is_hermitian: bool
    asymptotic_silence_factor: float


class EffectiveHamiltonian:
    """
    Implementation of effective Hamiltonian H_eff(σ) = τ H(τ) in LTQG.
    """
    
    def __init__(self, tau_0: float = 1.0, hbar: float = 1.0):
        """
        Initialize effective Hamiltonian framework.
        
        Args:
            tau_0: Reference time scale
            hbar: Reduced Planck constant
        """
        self.tau_0 = tau_0
        self.hbar = hbar
    
    def compute_effective_hamiltonian(self, sigma: float, 
                                    H_tau: Callable[[float], np.ndarray]) -> np.ndarray:
        """
        Compute H_eff(σ) = τ(σ) H(τ(σ)).
        
        Args:
            sigma: σ-time parameter
            H_tau: Function returning H(τ) for given proper time τ
            
        Returns:
            Effective Hamiltonian matrix H_eff(σ)
        """
        tau = self.tau_0 * np.exp(sigma)
        H_of_tau = H_tau(tau)
        return tau * H_of_tau
    
    def asymptotic_silence_envelope(self, sigma: Union[float, np.ndarray],
                                   silence_scale: float = 2.0,
                                   smoothness: float = 1.0) -> Union[float, np.ndarray]:
        """
        Compute smooth envelope function that implements asymptotic silence.
        
        f(σ) = (1 + tanh((σ + σ_silence)/smoothness))/2
        
        As σ → -∞: f(σ) → 0 (silence)
        As σ → +∞: f(σ) → 1 (full dynamics)
        
        Args:
            sigma: σ-time coordinate(s)
            silence_scale: Scale at which silence becomes significant
            smoothness: Transition smoothness parameter
            
        Returns:
            Envelope factor(s) between 0 and 1
        """
        sigma = np.asarray(sigma)
        return 0.5 * (1 + np.tanh((sigma + silence_scale) / smoothness))
    
    def silenced_effective_hamiltonian(self, sigma: float,
                                     H_tau: Callable[[float], np.ndarray],
                                     apply_silence: bool = True,
                                     silence_params: Optional[dict] = None) -> np.ndarray:
        """
        Compute H_eff with optional asymptotic silence envelope.
        
        H_eff_silenced(σ) = f(σ) × τ(σ) H(τ(σ))
        
        Args:
            sigma: σ-time parameter
            H_tau: Hamiltonian function H(τ)
            apply_silence: Whether to apply asymptotic silence
            silence_params: Parameters for silence envelope
            
        Returns:
            Silenced effective Hamiltonian
        """
        H_eff = self.compute_effective_hamiltonian(sigma, H_tau)
        
        if apply_silence:
            if silence_params is None:
                silence_params = {}
            envelope = self.asymptotic_silence_envelope(sigma, **silence_params)
            H_eff = envelope * H_eff
        
        return H_eff
    
    def analyze_effective_hamiltonian(self, sigma: float,
                                    H_tau: Callable[[float], np.ndarray],
                                    **silence_kwargs) -> EffectiveHamiltonianProperties:
        """
        Comprehensive analysis of H_eff at given σ.
        
        Args:
            sigma: σ-time parameter
            H_tau: Hamiltonian function
            **silence_kwargs: Parameters for asymptotic silence
            
        Returns:
            Properties of the effective Hamiltonian
        """
        H_eff = self.silenced_effective_hamiltonian(sigma, H_tau, **silence_kwargs)
        
        # Spectral analysis
        eigenvals, eigenvecs = la.eigh(H_eff) if np.allclose(H_eff, H_eff.conj().T) else la.eig(H_eff)
        
        # Properties
        spectral_norm = np.max(np.abs(eigenvals))
        trace = np.trace(H_eff)
        det = np.linalg.det(H_eff)
        cond_num = np.linalg.cond(H_eff)
        is_hermitian = np.allclose(H_eff, H_eff.conj().T)
        
        # Asymptotic silence factor
        silence_factor = self.asymptotic_silence_envelope(sigma)
        
        return EffectiveHamiltonianProperties(
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            spectral_norm=spectral_norm,
            trace=trace,
            determinant=det,
            condition_number=cond_num,
            is_hermitian=is_hermitian,
            asymptotic_silence_factor=silence_factor
        )
    
    def evolution_operator_sigma(self, sigma_initial: float, sigma_final: float,
                               H_tau: Callable[[float], np.ndarray],
                               num_steps: int = 1000) -> np.ndarray:
        """
        Compute evolution operator U(σ_f, σ_i) in σ-time.
        
        U(σ_f, σ_i) = T exp(-i/ℏ ∫_{σ_i}^{σ_f} H_eff(σ') dσ')
        
        Args:
            sigma_initial: Initial σ-time
            sigma_final: Final σ-time
            H_tau: Hamiltonian function H(τ)
            num_steps: Number of time evolution steps
            
        Returns:
            Evolution operator matrix
        """
        sigma_array = np.linspace(sigma_initial, sigma_final, num_steps)
        d_sigma = (sigma_final - sigma_initial) / (num_steps - 1)
        
        # Get dimension from H_eff
        H_eff_test = self.compute_effective_hamiltonian(sigma_initial, H_tau)
        dim = H_eff_test.shape[0]
        U = np.eye(dim, dtype=complex)
        
        # Time-ordered exponential (approximate)
        for i in range(1, len(sigma_array)):
            H_eff = self.compute_effective_hamiltonian(sigma_array[i], H_tau)
            U_step = la.expm(-1j * H_eff * d_sigma / self.hbar)
            U = U_step @ U
        
        return U
    
    def adiabatic_approximation_sigma(self, sigma: float,
                                    H_tau: Callable[[float], np.ndarray],
                                    d_H_tau: Optional[Callable[[float], np.ndarray]] = None,
                                    epsilon: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Analyze adiabatic approximation validity in σ-time.
        
        Adiabatic parameter: γ = |⟨n|dH_eff/dσ|m⟩| / (ℏ|E_n - E_m|²)
        
        Args:
            sigma: σ-time parameter
            H_tau: Hamiltonian function H(τ)
            d_H_tau: Derivative dH/dτ (computed numerically if not provided)
            epsilon: Numerical differentiation step
            
        Returns:
            Tuple of (eigenvalues, eigenvectors, max_adiabatic_parameter)
        """
        H_eff = self.compute_effective_hamiltonian(sigma, H_tau)
        eigenvals, eigenvecs = la.eigh(H_eff)
        
        # Compute dH_eff/dσ
        if d_H_tau is not None:
            tau = self.tau_0 * np.exp(sigma)
            dH_dt = d_H_tau(tau)
            # Chain rule: dH_eff/dσ = d(τH)/dσ = H + τ(dH/dτ)(dτ/dσ) = H + τ²(dH/dτ)
            dH_eff_dsigma = H_eff + tau * tau * dH_dt
        else:
            # Numerical differentiation
            H_eff_plus = self.compute_effective_hamiltonian(sigma + epsilon, H_tau)
            H_eff_minus = self.compute_effective_hamiltonian(sigma - epsilon, H_tau)
            dH_eff_dsigma = (H_eff_plus - H_eff_minus) / (2 * epsilon)
        
        # Compute adiabatic parameters
        max_adiabatic_param = 0.0
        
        for n in range(len(eigenvals)):
            for m in range(len(eigenvals)):
                if n != m and abs(eigenvals[n] - eigenvals[m]) > 1e-12:
                    matrix_element = np.abs(np.vdot(eigenvecs[:, n], 
                                                   dH_eff_dsigma @ eigenvecs[:, m]))
                    energy_gap = abs(eigenvals[n] - eigenvals[m])
                    adiabatic_param = matrix_element / (self.hbar * energy_gap**2)
                    max_adiabatic_param = max(max_adiabatic_param, adiabatic_param)
        
        return eigenvals, eigenvecs, max_adiabatic_param


class HamiltonianExamples:
    """
    Collection of example Hamiltonians for testing LTQG effective formalism.
    """
    
    @staticmethod
    def harmonic_oscillator(omega: float = 1.0, mass: float = 1.0) -> Callable[[float], np.ndarray]:
        """
        Simple harmonic oscillator H = p²/(2m) + ½mω²x².
        
        In matrix form for truncated basis.
        """
        def H_ho(tau):
            # Simple 2-level approximation
            return np.array([[omega/2, 0], [0, -omega/2]], dtype=complex)
        return H_ho
    
    @staticmethod
    def two_level_system(omega_0: float = 1.0, coupling: float = 0.1) -> Callable[[float], np.ndarray]:
        """
        Two-level system with time-dependent coupling.
        
        H(τ) = ½ω₀σ_z + g(τ)σ_x
        """
        def H_tls(tau):
            g_tau = coupling * np.exp(-tau)  # Exponential decay coupling
            return 0.5 * omega_0 * np.array([[1, 0], [0, -1]]) + \
                   g_tau * np.array([[0, 1], [1, 0]], dtype=complex)
        return H_tls
    
    @staticmethod
    def anharmonic_oscillator(omega: float = 1.0, lambda_4: float = 0.1, 
                            n_basis: int = 10) -> Callable[[float], np.ndarray]:
        """
        Anharmonic oscillator H = ω(a†a + ½) + λ(a†a)².
        
        Matrix representation in Fock basis.
        """
        def H_anharmonic(tau):
            # Number operator matrix elements
            n_matrix = np.diag(np.arange(n_basis))
            
            # Harmonic part
            H = omega * (n_matrix + 0.5 * np.eye(n_basis))
            
            # Anharmonic part: λ(a†a)²
            H += lambda_4 * n_matrix @ n_matrix
            
            return H.astype(complex)
        return H_anharmonic
    
    @staticmethod
    def cosmological_scalar_field(mass: float = 1.0, lambda_phi: float = 0.1,
                                potential_type: str = 'quartic') -> Callable[[float], np.ndarray]:
        """
        Scalar field in expanding universe.
        
        Simplified minisuperspace model.
        """
        def H_scalar(tau):
            if potential_type == 'quartic':
                # φ⁴ potential with time-dependent mass
                m_eff_sq = mass*mass * (1 + 1/tau)  # Time-dependent effective mass
                return np.array([[m_eff_sq/2, 0], [0, -m_eff_sq/2]], dtype=complex)
            else:
                raise ValueError(f"Unknown potential type: {potential_type}")
        return H_scalar


def demo_effective_hamiltonian():
    """
    Demonstrate effective Hamiltonian properties and asymptotic silence.
    """
    print("=== Effective Hamiltonian Demo ===\n")
    
    eff_ham = EffectiveHamiltonian(tau_0=1.0, hbar=1.0)
    
    # Demo 1: Basic effective Hamiltonian
    print("1. Harmonic Oscillator Effective Hamiltonian:")
    H_ho = HamiltonianExamples.harmonic_oscillator(omega=2*np.pi)
    
    sigma_vals = [-3, -1, 0, 1, 3]
    for sigma in sigma_vals:
        H_eff = eff_ham.compute_effective_hamiltonian(sigma, H_ho)
        tau = eff_ham.tau_0 * np.exp(sigma)
        norm = np.linalg.norm(H_eff)
        print(f"   σ = {sigma:2d}, τ = {tau:.3f}: ||H_eff|| = {norm:.3f}")
    
    # Demo 2: Asymptotic silence analysis
    print("\n2. Asymptotic Silence Analysis:")
    silence_params = {'silence_scale': 2.0, 'smoothness': 1.0}
    
    for sigma in sigma_vals:
        properties = eff_ham.analyze_effective_hamiltonian(sigma, H_ho, 
                                                          apply_silence=True,
                                                          silence_params=silence_params)
        print(f"   σ = {sigma:2d}: silence factor = {properties.asymptotic_silence_factor:.3f}, "
              f"||H_eff|| = {properties.spectral_norm:.3f}")
    
    # Demo 3: Two-level system evolution
    print("\n3. Two-Level System σ-Evolution:")
    H_tls = HamiltonianExamples.two_level_system(omega_0=1.0, coupling=0.2)
    
    # Initial state
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    
    # Evolution operator
    sigma_i, sigma_f = -2.0, 2.0
    U = eff_ham.evolution_operator_sigma(sigma_i, sigma_f, H_tls, num_steps=500)
    
    # Final state
    psi_f = U @ psi_0
    
    print(f"   Initial state: |ψ(σ_i)⟩ = {psi_0}")
    print(f"   Final state:   |ψ(σ_f)⟩ = [{psi_f[0]:.3f}, {psi_f[1]:.3f}]")
    print(f"   Norm preservation: ||ψ_f|| = {np.linalg.norm(psi_f):.6f}")
    print(f"   Unitarity check: ||U†U - I|| = {np.linalg.norm(U.conj().T @ U - np.eye(2)):.2e}")
    
    # Demo 4: Adiabatic parameter analysis
    print("\n4. Adiabatic Approximation Analysis:")
    
    for sigma in [-2, 0, 2]:
        eigenvals, eigenvecs, max_adiabatic = eff_ham.adiabatic_approximation_sigma(
            sigma, H_tls, epsilon=1e-4)
        print(f"   σ = {sigma:2d}: max adiabatic parameter = {max_adiabatic:.2e}")
        print(f"              eigenvalues = [{eigenvals[0]:.3f}, {eigenvals[1]:.3f}]")


def plot_effective_hamiltonian_evolution():
    """
    Plot evolution of effective Hamiltonian properties vs σ.
    """
    eff_ham = EffectiveHamiltonian()
    
    # Two-level system
    H_tls = HamiltonianExamples.two_level_system(omega_0=1.0, coupling=0.5)
    
    sigma_range = np.linspace(-4, 3, 100)
    spectral_norms = []
    silence_factors = []
    eigenval_gaps = []
    
    for sigma in sigma_range:
        props = eff_ham.analyze_effective_hamiltonian(sigma, H_tls, apply_silence=True)
        spectral_norms.append(props.spectral_norm)
        silence_factors.append(props.asymptotic_silence_factor)
        eigenval_gaps.append(abs(props.eigenvalues[1] - props.eigenvalues[0]))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Spectral norm
    axes[0,0].semilogy(sigma_range, spectral_norms, 'b-', linewidth=2)
    axes[0,0].set_xlabel('σ-time')
    axes[0,0].set_ylabel('||H_eff(σ)||')
    axes[0,0].set_title('Effective Hamiltonian Norm')
    axes[0,0].grid(True, alpha=0.3)
    
    # Silence factor
    axes[0,1].plot(sigma_range, silence_factors, 'r-', linewidth=2)
    axes[0,1].set_xlabel('σ-time')
    axes[0,1].set_ylabel('Silence Factor')
    axes[0,1].set_title('Asymptotic Silence Envelope')
    axes[0,1].grid(True, alpha=0.3)
    
    # Eigenvalue gap
    axes[1,0].semilogy(sigma_range, eigenval_gaps, 'g-', linewidth=2)
    axes[1,0].set_xlabel('σ-time')
    axes[1,0].set_ylabel('|E₁ - E₀|')
    axes[1,0].set_title('Energy Gap')
    axes[1,0].grid(True, alpha=0.3)
    
    # Combined view
    ax_combined = axes[1,1]
    ax_combined.semilogy(sigma_range, spectral_norms, 'b-', linewidth=2, label='||H_eff||')
    ax_combined.semilogy(sigma_range, eigenval_gaps, 'g-', linewidth=2, label='Energy Gap')
    ax_combined.set_xlabel('σ-time')
    ax_combined.set_ylabel('Energy Scale')
    ax_combined.set_title('Energy Scales vs σ-time')
    ax_combined.legend()
    ax_combined.grid(True, alpha=0.3)
    
    # Highlight asymptotic silence region
    silence_region = sigma_range < -2
    for ax in axes.flat:
        ax.axvspan(sigma_range[silence_region][0], -2, alpha=0.2, color='orange', 
                  label='Asymptotic Silence')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    demo_effective_hamiltonian()
    
    # Uncomment to generate plots
    # fig = plot_effective_hamiltonian_evolution()
    # plt.show()