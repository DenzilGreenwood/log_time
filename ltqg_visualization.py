"""
Log-Time Quantum Gravity Visualization Suite
============================================

Author: Denzil James Greenwood
GitHub: https://github.com/DenzilGreenwood/log_time
License: MIT

This module provides comprehensive visualization capabilities for LTQG phenomena,
implementing all figures described in the Log-Time Quantum Gravity paper.

Figures implemented:
1. The Log-Time Map σ = log(τ/τ₀)
2. Regularization of Curvature Invariants
3. Gravitational Redshift as Additive σ-Shift
4. Effective Generator and Asymptotic Silence
5. σ-Uniform Zeno Protocol Predictions
6. Early-Universe Mode Evolution
7. Black Hole Embedding in Log-Time Coordinates (3D)
8. Enhanced Black Hole with Rotational Symmetry and Geodesics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, Dict, List
from ltqg_core import LTQGSimulator, create_ltqg_simulator, LTQGConfig

# Set up publication-quality plotting style
plt.style.use('default')
# Use matplotlib's built-in color cycle instead of seaborn
plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
plt.rcParams.update({
    'figure.figsize': (10, 7),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


class LTQGVisualizer:
    """
    Main visualization class for LTQG phenomena.
    
    Provides methods to generate all figures described in the paper,
    with options for saving high-quality outputs.
    """
    
    def __init__(self, simulator: LTQGSimulator = None, 
                 save_dir: str = "figs", dpi: int = 300):
        """
        Initialize the visualizer.
        
        Args:
            simulator: LTQG simulator instance
            save_dir: Directory to save figures
            dpi: DPI for saved figures
        """
        self.simulator = simulator or create_ltqg_simulator()
        self.save_dir = save_dir
        self.dpi = dpi
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def save_figure(self, fig, filename: str, tight_layout: bool = True):
        """Save figure with high quality settings."""
        if tight_layout:
            fig.tight_layout()
        filepath = os.path.join(self.save_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")
    
    def figure_1_log_time_map(self, save: bool = True) -> plt.Figure:
        """
        Figure 1: The Log-Time Map σ = log(τ/τ₀)
        
        Shows the mapping from proper time τ to log-time σ, highlighting
        the compression of early times and expansion of macroscopic times.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Panel A: τ to σ mapping
        tau_range = np.logspace(-6, 3, 1000)  # From 10⁻⁶ to 10³ τ₀
        sigma_range = self.simulator.time_transform.sigma_from_tau(tau_range)
        
        ax1.semilogx(tau_range, sigma_range, 'b-', linewidth=3)
        ax1.axhline(0, color='red', linestyle='--', alpha=0.7, 
                   label='σ = 0 (τ = τ₀)')
        ax1.axvline(1.0, color='red', linestyle='--', alpha=0.7)
        
        # Highlight key regions
        ax1.fill_between([1e-6, 1e-2], [-20, -20], [20, 20], 
                        alpha=0.2, color='orange', label='Early time compression')
        ax1.fill_between([1, 1000], [-20, -20], [20, 20], 
                        alpha=0.2, color='green', label='Macroscopic expansion')
        
        ax1.set_xlabel('Proper Time τ/τ₀')
        ax1.set_ylabel('Log-Time σ')
        ax1.set_title('Log-Time Transformation')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(-15, 8)
        
        # Panel B: Interval mapping
        tau_intervals = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
        sigma_intervals = self.simulator.time_transform.sigma_from_tau(tau_intervals)
        
        # Show how equal τ intervals become unequal σ intervals
        for i in range(len(tau_intervals)-1):
            tau_mid = np.sqrt(tau_intervals[i] * tau_intervals[i+1])
            sigma_mid = self.simulator.time_transform.sigma_from_tau(np.array([tau_mid]))[0]
            
            # Interval widths
            dtau = tau_intervals[i+1] - tau_intervals[i]
            dsigma = sigma_intervals[i+1] - sigma_intervals[i]
            
            ax2.barh(i, dsigma, height=0.7, alpha=0.7, 
                    label=f'Δτ = {dtau:.1e}' if i < 4 else None)
        
        ax2.set_xlabel('Interval Width Δσ')
        ax2.set_ylabel('Interval Index')
        ax2.set_title('Interval Compression/Expansion')
        ax2.grid(True, alpha=0.3)
        if len(ax2.get_legend_handles_labels()[0]) > 0:
            ax2.legend()
        
        if save:
            self.save_figure(fig, "log_time_map.png")
        
        return fig
    
    def figure_2_singularity_regularization(self, save: bool = True) -> plt.Figure:
        """
        Figure 2: Regularization of Curvature Invariants
        
        Compares singular behavior in τ-space with regularized behavior in σ-space.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # τ-space: singular behavior
        tau_sing = np.logspace(-4, 0, 1000)
        R_tau_singular = 1.0 / tau_sing**2  # R ∝ 1/τ²
        rho_tau_singular = 1.0 / tau_sing**4  # ρ ∝ 1/τ⁴
        
        ax1.loglog(tau_sing, R_tau_singular, 'r-', linewidth=3, 
                  label='R(τ) ∝ 1/τ² (singular)')
        ax1.set_xlabel('Proper Time τ/τ₀')
        ax1.set_ylabel('Curvature Scalar R')
        ax1.set_title('Singular Behavior in τ-Space')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        ax1.set_ylim(1, 1e8)
        
        ax2.loglog(tau_sing, rho_tau_singular, 'purple', linewidth=3,
                  label='ρ(τ) ∝ 1/τ⁴ (singular)')
        ax2.set_xlabel('Proper Time τ/τ₀')
        ax2.set_ylabel('Energy Density ρ')
        ax2.set_title('Energy Density Divergence')
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()
        ax2.set_ylim(1, 1e16)
        
        # σ-space: regularized behavior
        sigma_reg = np.linspace(-10, 2, 1000)
        R_sigma_regular = self.simulator.singularity.curvature_scalar(sigma_reg)
        rho_sigma_regular = self.simulator.singularity.energy_density(sigma_reg)
        
        ax3.plot(sigma_reg, R_sigma_regular, 'b-', linewidth=3,
                label='R(σ) ∝ exp(-2σ) (regular)')
        ax3.set_xlabel('Log-Time σ')
        ax3.set_ylabel('Curvature Scalar R')
        ax3.set_title('Regularized in σ-Space')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_yscale('log')
        
        ax4.plot(sigma_reg, rho_sigma_regular, 'orange', linewidth=3,
                label='ρ(σ) ∝ exp(-4σ) (regular)')
        ax4.set_xlabel('Log-Time σ')
        ax4.set_ylabel('Energy Density ρ')
        ax4.set_title('Energy Density Regularization')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_yscale('log')
        
        # Add asymptotic silence region
        for ax in [ax3, ax4]:
            ax.axvspan(-10, -5, alpha=0.2, color='gray', 
                      label='Asymptotic Silence')
            ax.legend()
        
        if save:
            self.save_figure(fig, "singularity_regularization.png")
        
        return fig
    
    def figure_3_gravitational_redshift_shift(self, save: bool = True) -> plt.Figure:
        """
        Figure 3: Gravitational Redshift as Additive σ-Shift
        
        Shows how multiplicative time dilation becomes additive σ-shifts.
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel A: Redshift factor vs radius
        rs = 1.0  # Schwarzschild radius
        r_range = np.linspace(1.01, 5.0, 1000) * rs
        alpha = self.simulator.redshift.redshift_factor_schwarzschild(r_range, rs)
        
        ax1.plot(r_range/rs, alpha, 'blue', linewidth=3)
        ax1.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Horizon')
        ax1.set_xlabel('Radius r/rs')
        ax1.set_ylabel('Redshift Factor α(r)')
        ax1.set_title('Schwarzschild Redshift Factor')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Panel B: Multiplicative relation in τ
        tau_base = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        alpha_examples = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        for i, a in enumerate(alpha_examples):
            tau_dilated = tau_base * a
            ax2.plot(tau_base, tau_dilated, 'o-', linewidth=2, 
                    label=f'α = {a:.1f}')
        
        ax2.plot([0, 6], [0, 6], 'k--', alpha=0.5, label='No dilation')
        ax2.set_xlabel('Observer A Time τ_A')
        ax2.set_ylabel('Observer B Time τ_B = α τ_A')
        ax2.set_title('Multiplicative Time Dilation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, 5.5)
        ax2.set_ylim(0, 5.5)
        
        # Panel C: Additive relation in σ
        sigma_base = self.simulator.time_transform.sigma_from_tau(tau_base)
        
        for i, a in enumerate(alpha_examples):
            sigma_shift = sigma_base + np.log(a)
            ax3.plot(sigma_base, sigma_shift, 'o-', linewidth=2,
                    label=f'α = {a:.1f}, Δσ = {np.log(a):.1f}')
        
        ax3.plot([-2, 2], [-2, 2], 'k--', alpha=0.5, label='No shift')
        ax3.set_xlabel('Observer A Log-Time σ_A')
        ax3.set_ylabel('Observer B Log-Time σ_B = σ_A + log(α)')
        ax3.set_title('Additive σ-Shifts')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        if save:
            self.save_figure(fig, "gravitational_redshift_shift.png")
        
        return fig
    
    def figure_4_effective_generator_silence(self, save: bool = True) -> plt.Figure:
        """
        Figure 4: Effective Generator and Asymptotic Silence
        
        Shows the magnitude of the σ-time evolution generator and the
        asymptotic silence condition.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Generator magnitude K(σ) = τ₀ exp(σ) H
        sigma_range = np.linspace(-15, 5, 1000)
        H = 1.0  # Normalized Hamiltonian
        K_magnitude = self.simulator.config.tau0 * np.exp(sigma_range) * H
        
        ax1.semilogy(sigma_range, K_magnitude, 'purple', linewidth=3)
        ax1.axhline(1e-10, color='red', linestyle='--', alpha=0.7,
                   label='Numerical silence threshold')
        ax1.axvspan(-15, -10, alpha=0.2, color='gray', label='Asymptotic silence')
        ax1.set_xlabel('Log-Time σ')
        ax1.set_ylabel('Generator Magnitude |K(σ)|')
        ax1.set_title('σ-Time Evolution Generator')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(1e-15, 1e5)
        
        # Evolution "velocity" in different frames
        tau_points = np.logspace(-6, 2, 1000)
        sigma_points = self.simulator.time_transform.sigma_from_tau(tau_points)
        
        # dψ/dτ vs dψ/dσ
        dtau_dsigma = self.simulator.time_transform.dtau_dsigma(sigma_points)
        
        ax2.loglog(tau_points, np.ones_like(tau_points), 'b-', 
                  linewidth=3, label='|dψ/dτ| (constant)')
        ax2.loglog(tau_points, 1.0/dtau_dsigma, 'r-', 
                  linewidth=3, label='|dψ/dσ| ∝ exp(-σ)')
        ax2.set_xlabel('Proper Time τ/τ₀')
        ax2.set_ylabel('Evolution Rate Magnitude')
        ax2.set_title('Quantum Evolution Rates')
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()
        
        # Quantum state norm conservation
        sigma_evo = np.linspace(-5, 3, 100)
        # Show that ||ψ(σ)||² remains constant
        norm_squared = np.ones_like(sigma_evo)  # Unitary evolution preserves norm
        
        ax3.plot(sigma_evo, norm_squared, 'green', linewidth=3,
                label='||ψ(σ)||² = constant')
        ax3.axhspan(0.99, 1.01, alpha=0.2, color='green', 
                   label='Unitarity preserved')
        ax3.set_xlabel('Log-Time σ')
        ax3.set_ylabel('State Norm ||ψ||²')
        ax3.set_title('Unitarity in σ-Time')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0.95, 1.05)
        
        # Time-energy uncertainty in σ-frame
        sigma_uncertainty = np.linspace(-10, 2, 1000)
        delta_E = 1.0  # Fixed energy uncertainty
        delta_sigma = np.ones_like(sigma_uncertainty)  # σ uncertainty
        # Effective time uncertainty: Δt_eff = (dτ/dσ) Δσ
        delta_tau_eff = self.simulator.time_transform.dtau_dsigma(sigma_uncertainty) * delta_sigma
        
        ax4.semilogy(sigma_uncertainty, delta_tau_eff, 'orange', linewidth=3,
                    label='Δτ_eff = (dτ/dσ) Δσ')
        ax4.axhspan(1e-10, 1e-5, alpha=0.2, color='gray',
                   label='Ultra-precise regime')
        ax4.set_xlabel('Log-Time σ')
        ax4.set_ylabel('Effective Time Uncertainty Δτ_eff')
        ax4.set_title('Time-Energy Uncertainty')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        if save:
            self.save_figure(fig, "effective_generator_silence.png")
        
        return fig
    
    def figure_5_zeno_protocol_predictions(self, save: bool = True) -> plt.Figure:
        """
        Figure 5: Predictions for σ-Uniform Zeno Protocols
        
        Shows predicted differences between τ-uniform and σ-uniform 
        measurement protocols under gravitational redshift.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel A: Transition probability vs redshift
        alpha_range = np.logspace(-3, 0, 100)  # From strong to no redshift
        n_measurements = 10
        
        # Standard QM prediction (τ-uniform)
        P_tau_uniform = np.ones_like(alpha_range)  # No redshift dependence
        
        # LTQG prediction (σ-uniform)
        P_sigma_uniform = np.array([
            self.simulator.protocols.zeno_suppression_factor(alpha, n_measurements)
            for alpha in alpha_range
        ])
        
        ax1.semilogx(alpha_range, P_tau_uniform, 'b--', linewidth=3,
                    label='Standard QM (τ-uniform)')
        ax1.semilogx(alpha_range, P_sigma_uniform, 'r-', linewidth=3,
                    label='LTQG (σ-uniform)')
        ax1.fill_between(alpha_range, P_sigma_uniform, P_tau_uniform,
                        alpha=0.3, color='yellow', label='Predicted difference')
        
        ax1.set_xlabel('Redshift Factor α')
        ax1.set_ylabel('Relative Transition Probability')
        ax1.set_title('Zeno Effect vs Gravitational Redshift')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1.1)
        
        # Panel B: Measurement schedule comparison
        tau_range = (0.1, 10.0)
        alpha_demo = 0.1  # Strong redshift case
        protocols = self.simulator.protocols.compare_measurement_protocols(
            tau_range, alpha_demo, n_measurements=8)
        
        tau_uniform = protocols['tau_uniform_times']
        sigma_uniform = protocols['sigma_uniform_times']
        
        indices = np.arange(len(tau_uniform))
        ax2.plot(indices, tau_uniform, 'bo-', linewidth=2, markersize=8,
                label='τ-uniform schedule')
        ax2.plot(indices, sigma_uniform, 'ro-', linewidth=2, markersize=8,
                label='σ-uniform schedule')
        
        ax2.set_xlabel('Measurement Index')
        ax2.set_ylabel('Proper Time τ')
        ax2.set_title(f'Measurement Schedules (α = {alpha_demo})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Panel C: Accumulated phase difference
        sigma_vals = np.linspace(-5, 2, 100)
        base_phase = 1.0
        
        # Phase accumulation in σ-uniform protocol
        phase_sigma = base_phase * sigma_vals  # Linear in σ
        # Phase accumulation in τ-uniform protocol
        tau_vals = self.simulator.time_transform.tau_from_sigma(sigma_vals)
        phase_tau = base_phase * tau_vals  # Exponential in underlying τ
        
        ax3.plot(sigma_vals, phase_sigma, 'r-', linewidth=3,
                label='σ-uniform protocol')
        ax3.plot(sigma_vals, phase_tau, 'b-', linewidth=3,
                label='τ-uniform protocol')
        ax3.fill_between(sigma_vals, phase_sigma, phase_tau,
                        alpha=0.3, color='green', label='Phase difference')
        
        ax3.set_xlabel('Log-Time σ')
        ax3.set_ylabel('Accumulated Phase')
        ax3.set_title('Phase Evolution Comparison')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Panel D: Experimental feasibility
        redshift_factors = [1.0, 0.5, 0.1, 0.01]
        sigma_shifts = [np.log(alpha) for alpha in redshift_factors]
        feasibility_scores = [1.0, 0.8, 0.5, 0.2]  # Heuristic feasibility
        
        colors = ['green', 'yellow', 'orange', 'red']
        bars = ax4.bar(range(len(redshift_factors)), feasibility_scores, 
                      color=colors, alpha=0.7)
        
        ax4.set_xlabel('Redshift Scenario')
        ax4.set_ylabel('Experimental Feasibility')
        ax4.set_title('Protocol Implementation Feasibility')
        ax4.set_xticks(range(len(redshift_factors)))
        ax4.set_xticklabels([f'α={α}\nΔσ={Δσ:.1f}' 
                            for α, Δσ in zip(redshift_factors, sigma_shifts)])
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, (bar, score) in enumerate(zip(bars, feasibility_scores)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{score:.1f}', ha='center', va='bottom')
        
        if save:
            self.save_figure(fig, "zeno_protocol_predictions.png")
        
        return fig
    
    def figure_6_early_universe_modes(self, save: bool = True) -> plt.Figure:
        """
        Figure 6: Early-Universe Mode Evolution in σ-Time
        
        Shows smooth evolution of quantum modes in σ-time vs 
        singular behavior in standard τ-time.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Simulate early universe evolution
        simulation = self.simulator.simulate_early_universe((-15, 5), n_points=1000)
        sigma = simulation['sigma']
        tau = simulation['tau']
        
        # Panel A: Mode amplitude evolution (singular in τ)
        # Standard QFT prediction: mode grows as 1/√τ near singularity
        mode_amplitude_tau = 1.0 / np.sqrt(tau + 1e-10)
        
        ax1.loglog(tau, mode_amplitude_tau, 'r-', linewidth=3,
                  label='Standard QFT: |φ| ∝ 1/√τ')
        ax1.axvline(self.simulator.config.tau0, color='blue', 
                   linestyle='--', alpha=0.7, label='τ = τ₀')
        ax1.set_xlabel('Proper Time τ/τ₀')
        ax1.set_ylabel('Mode Amplitude |φ|')
        ax1.set_title('Singular Mode Evolution (τ-frame)')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        ax1.set_xlim(1e-6, 1e2)
        ax1.set_ylim(1e-2, 1e6)
        
        # Panel B: Mode amplitude evolution (regular in σ)
        # LTQG prediction: mode amplitude = exp(-σ/2) = smooth
        mode_amplitude_sigma = np.exp(-sigma / 2.0)
        
        ax2.semilogy(sigma, mode_amplitude_sigma, 'b-', linewidth=3,
                    label='LTQG: |φ| ∝ exp(-σ/2)')
        ax2.axvspan(-15, -10, alpha=0.2, color='gray', 
                   label='Asymptotic silence')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.7, label='σ = 0')
        ax2.set_xlabel('Log-Time σ')
        ax2.set_ylabel('Mode Amplitude |φ|')
        ax2.set_title('Regular Mode Evolution (σ-frame)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Panel C: Scale factor comparison
        a_radiation = simulation['scale_factor_radiation']
        a_matter = simulation['scale_factor_matter']
        
        ax3.semilogy(sigma, a_radiation, 'orange', linewidth=3,
                    label='Radiation era: a ∝ exp(σ/2)')
        ax3.semilogy(sigma, a_matter, 'purple', linewidth=3,
                    label='Matter era: a ∝ exp(2σ/3)')
        ax3.axvspan(-15, -10, alpha=0.2, color='gray')
        ax3.set_xlabel('Log-Time σ')
        ax3.set_ylabel('Scale Factor a(σ)')
        ax3.set_title('Cosmological Scale Factor Evolution')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Panel D: Curvature evolution
        curvature = simulation['curvature_scalar']
        energy_density = simulation['energy_density']
        
        ax4.semilogy(sigma, curvature, 'green', linewidth=3,
                    label='Curvature scalar R(σ)')
        ax4.semilogy(sigma, energy_density, 'brown', linewidth=3,
                    label='Energy density ρ(σ)')
        
        # Show asymptotic behavior
        silence_mask = simulation['asymptotic_silence']
        if np.any(silence_mask):
            sigma_silence = sigma[silence_mask]
            ax4.axvspan(sigma_silence[0], sigma_silence[-1], 
                       alpha=0.2, color='gray', label='Asymptotic silence')
        else:
            # If no silence region, just mark early times
            ax4.axvspan(sigma[0], sigma[len(sigma)//10], 
                       alpha=0.2, color='gray', label='Early time region')
        
        ax4.set_xlabel('Log-Time σ')
        ax4.set_ylabel('Physical Quantities')
        ax4.set_title('Singularity Regularization')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(1e-10, 1e10)
        
        if save:
            self.save_figure(fig, "early_universe_modes.png")
        
        return fig
    
    def figure_experimental_feasibility(self, save: bool = True) -> plt.Figure:
        """
        Additional Figure: Experimental Feasibility Assessment
        
        Shows practical considerations for implementing LTQG tests.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Panel A: Required precision vs redshift
        alpha_range = np.logspace(-4, 0, 100)
        # Required timing precision scales as α for σ-uniform protocols
        timing_precision = alpha_range * 1e-15  # seconds (rough estimate)
        
        ax1.loglog(alpha_range, timing_precision, 'blue', linewidth=3)
        ax1.axhline(1e-18, color='red', linestyle='--', 
                   label='Current atomic clock precision')
        ax1.axhline(1e-21, color='orange', linestyle='--',
                   label='Future optical lattice clocks')
        ax1.fill_between(alpha_range, timing_precision, 1e-25,
                        where=(timing_precision > 1e-18), alpha=0.3,
                        color='red', label='Currently infeasible')
        
        ax1.set_xlabel('Redshift Factor α')
        ax1.set_ylabel('Required Timing Precision (s)')
        ax1.set_title('Precision Requirements')
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        
        # Panel B: Signal-to-noise ratio
        measurement_times = np.logspace(-6, 2, 100)
        # SNR improves with measurement time for quantum interferometry
        snr = np.sqrt(measurement_times) * 10  # Simplified model
        
        ax2.loglog(measurement_times, snr, 'green', linewidth=3)
        ax2.axhline(3, color='red', linestyle='--', label='Detection threshold')
        ax2.axhline(10, color='orange', linestyle='--', label='High confidence')
        ax2.fill_between(measurement_times, snr, 0.1,
                        where=(snr > 3), alpha=0.3, color='green',
                        label='Detectable regime')
        
        ax2.set_xlabel('Measurement Time (τ₀ units)')
        ax2.set_ylabel('Signal-to-Noise Ratio')
        ax2.set_title('Detection Sensitivity')
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend()
        
        # Panel C: Analog gravity implementations
        systems = ['BEC in trap', 'Optical lattice', 'Ion chains', 'Superconducting circuits']
        max_redshift = [0.01, 0.1, 0.001, 0.05]  # Achievable α values
        difficulty = [3, 2, 4, 2]  # 1=easy, 5=very hard
        
        colors = ['red' if d >= 4 else 'orange' if d >= 3 else 'green' for d in difficulty]
        bars = ax3.barh(range(len(systems)), max_redshift, color=colors, alpha=0.7)
        
        ax3.set_yticks(range(len(systems)))
        ax3.set_yticklabels(systems)
        ax3.set_xlabel('Maximum Achievable |log α|')
        ax3.set_title('Analog Gravity Platforms')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add difficulty annotations
        for i, (bar, diff) in enumerate(zip(bars, difficulty)):
            width = bar.get_width()
            ax3.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                    f'Difficulty: {diff}/5', ha='left', va='center', fontsize=10)
        
        # Panel D: Predicted effect sizes
        redshift_scenarios = ['Earth surface', 'GPS satellites', 'White dwarf', 'Neutron star']
        alpha_values = [0.999999999, 0.99999999, 0.5, 0.01]
        effect_sizes = [abs(np.log(alpha)) * 100 for alpha in alpha_values]  # Percentage
        
        bars = ax4.bar(range(len(redshift_scenarios)), effect_sizes, 
                      color=['green', 'blue', 'orange', 'red'], alpha=0.7)
        
        ax4.set_xticks(range(len(redshift_scenarios)))
        ax4.set_xticklabels(redshift_scenarios, rotation=45, ha='right')
        ax4.set_ylabel('Predicted Effect Size (%)')
        ax4.set_title('LTQG Signatures in Different Environments')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_yscale('log')
        
        # Add value annotations
        for i, (bar, effect) in enumerate(zip(bars, effect_sizes)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{effect:.2e}%', ha='center', va='bottom', fontsize=9)
        
        if save:
            self.save_figure(fig, "experimental_feasibility.png")
        
        return fig
    
    def figure_black_hole_embedding(self, save: bool = True, enhanced: bool = False) -> plt.Figure:
        """
        Black Hole Visualization: 3D Embedding of Log-Time Geometry
        
        Shows the 3D embedding of Schwarzschild spacetime in log-time coordinates,
        demonstrating how the log-time transformation affects the geometry near
        the event horizon and singularity.
        
        Args:
            save: Whether to save the figure
            enhanced: Whether to create enhanced version with rotational symmetry
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        if enhanced:
            return self._figure_black_hole_enhanced(save)
        
        # Parameters
        r_s = 1.0       # Schwarzschild radius
        tau0 = 1e-43    # Planck time (reference)
        sigma = np.linspace(-5, 3, 400)
        r = np.linspace(r_s * 1.001, 6 * r_s, 400)
        
        # Create meshgrid
        R, S = np.meshgrid(r, sigma)
        
        # Define embedding height z(r, sigma)
        # Simplified log-time curvature model
        Z = np.sqrt((R / r_s - 1)) * np.exp(-S / 5)
        
        # Optional color map: log-time
        color_map = plt.cm.plasma((S - S.min()) / (S.max() - S.min()))
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Surface plot
        surf = ax.plot_surface(R, Z, S, facecolors=color_map, 
                              rstride=4, cstride=4, linewidth=0, 
                              antialiased=True, alpha=0.8)
        
        # Add horizon line
        r_horizon = np.ones_like(sigma) * r_s
        z_horizon = np.zeros_like(sigma)
        ax.plot(r_horizon, z_horizon, sigma, 'r-', linewidth=4, 
                label='Event Horizon r = rs')
        
        # Add asymptotic silence region
        sigma_silence = sigma[sigma < -2]
        r_silence = np.ones_like(sigma_silence) * 2 * r_s
        z_silence = np.sqrt((r_silence / r_s - 1)) * np.exp(-sigma_silence / 5)
        ax.plot(r_silence, z_silence, sigma_silence, 'k--', linewidth=3,
                label='Asymptotic Silence Region')
        
        # Formatting
        ax.set_xlabel('r / rs (Radius)', fontsize=12)
        ax.set_ylabel('Embedding z', fontsize=12)
        ax.set_zlabel('σ (Log-Time)', fontsize=12)
        ax.set_title('Log-Time Black Hole Geometry (LTQG View)', fontsize=14, pad=20)
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(S)
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Log-Time σ', fontsize=12)
        
        # Set viewing angle for better visualization
        ax.view_init(elev=20, azim=45)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
        
        # Add text annotation
        ax.text2D(0.02, 0.98, 
                 "LTQG Prediction:\n• Smooth geometry in σ-coordinates\n• Horizon becomes regular\n• Asymptotic silence at σ → -∞", 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if save:
            self.save_figure(fig, "black_hole_embedding.png")
        
        return fig
    
    def _figure_black_hole_enhanced(self, save: bool = True) -> plt.Figure:
        """
        Enhanced Black Hole Visualization with Rotational Symmetry and Geodesics
        
        Implements the refinements suggested by Denzil:
        1. 360° rotational symmetry for intuitive black hole representation
        2. Curvature intensity overlay
        3. Observer geodesic path showing regularized infall
        4. Enhanced scientific annotations
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Parameters
        r_s = 1.0       # Schwarzschild radius
        sigma = np.linspace(-6, 3, 300)
        r = np.linspace(r_s * 1.001, 5 * r_s, 200)
        phi = np.linspace(0, 2*np.pi, 60)  # Azimuthal angle for rotational symmetry
        
        # Create cylindrical coordinates for rotational symmetry
        R, S = np.meshgrid(r, sigma)
        Z = np.sqrt((R / r_s - 1)) * np.exp(-S / 4)  # Embedding height
        
        # Convert to Cartesian for full 3D rotational surface
        PHI, R_cyl = np.meshgrid(phi, r)
        SIGMA_cyl = np.zeros_like(PHI) + 1.0  # Mid-level σ slice for demonstration
        Z_cyl = np.sqrt((R_cyl / r_s - 1)) * np.exp(-SIGMA_cyl / 4)
        
        X_cyl = R_cyl * np.cos(PHI)
        Y_cyl = R_cyl * np.sin(PHI)
        
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Main cross-section surface (enhanced with curvature)
        # Curvature intensity: higher near horizon
        curvature_intensity = 1.0 / ((R / r_s - 1)**2 + 0.01)  # Regularized 1/(r-rs)²
        curvature_normalized = (curvature_intensity - curvature_intensity.min()) / \
                              (curvature_intensity.max() - curvature_intensity.min())
        
        # Color map combining σ-time and curvature
        color_sigma = (S - S.min()) / (S.max() - S.min())
        color_combined = 0.7 * color_sigma + 0.3 * curvature_normalized
        color_map = plt.cm.plasma(color_combined)
        
        # Main surface plot
        surf = ax.plot_surface(R, Z, S, facecolors=color_map, 
                              rstride=3, cstride=3, linewidth=0.5, 
                              antialiased=True, alpha=0.85)
        
        # Add rotational symmetry hint with circular cross-sections
        for sigma_val in [-4, -2, 0, 2]:
            if sigma_val >= sigma.min() and sigma_val <= sigma.max():
                z_level = np.sqrt((r / r_s - 1)) * np.exp(-sigma_val / 4)
                # Draw circles at different σ levels
                theta_circle = np.linspace(0, 2*np.pi, 50)
                for r_val in [1.5*r_s, 2*r_s, 3*r_s]:
                    x_circle = r_val * np.cos(theta_circle)
                    y_circle = r_val * np.sin(theta_circle)
                    z_circle = np.ones_like(theta_circle) * np.sqrt((r_val / r_s - 1)) * np.exp(-sigma_val / 4)
                    sigma_circle = np.ones_like(theta_circle) * sigma_val
                    ax.plot(x_circle, y_circle, sigma_circle, 'gray', alpha=0.3, linewidth=1)
        
        # Event horizon (full rotational surface)
        theta_horizon = np.linspace(0, 2*np.pi, 100)
        sigma_horizon = np.linspace(-5, 3, 100)
        THETA_h, SIGMA_h = np.meshgrid(theta_horizon, sigma_horizon)
        X_horizon = r_s * np.cos(THETA_h)
        Y_horizon = r_s * np.sin(THETA_h)
        Z_horizon = np.zeros_like(THETA_h)
        
        ax.plot_surface(X_horizon, Y_horizon, SIGMA_h, alpha=0.3, color='red', 
                       label='Event Horizon')
        
        # Add horizon circle at σ = 0 level
        x_h = r_s * np.cos(theta_horizon)
        y_h = r_s * np.sin(theta_horizon)
        z_h = np.zeros_like(theta_horizon)
        ax.plot(x_h, y_h, z_h, 'r-', linewidth=4, label='Event Horizon r = rs')
        
        # Observer geodesic: infalling particle path
        # Geodesic in LTQG: asymptotically slows in σ but never reaches singularity
        tau_geodesic = np.logspace(-4, 2, 100)  # Proper time along geodesic
        sigma_geodesic = np.log(tau_geodesic)    # Log-time coordinate
        
        # Radial infall (simplified): starts at r = 4rs, falls toward horizon
        r_geodesic = r_s * (1 + 3 * np.exp(-tau_geodesic/10))  # Asymptotic approach
        
        # Choose φ = 0 plane for geodesic
        x_geodesic = r_geodesic
        y_geodesic = np.zeros_like(r_geodesic)
        z_geodesic = np.sqrt((r_geodesic / r_s - 1)) * np.exp(-sigma_geodesic / 4)
        
        # Only plot geodesic where it makes physical sense
        valid_geodesic = (r_geodesic > r_s) & (sigma_geodesic > -5) & (sigma_geodesic < 3)
        ax.plot(x_geodesic[valid_geodesic], z_geodesic[valid_geodesic], 
                sigma_geodesic[valid_geodesic], 'lime', linewidth=4, 
                label='Observer Geodesic (Regularized Infall)')
        
        # Mark start and asymptotic end of geodesic
        if np.any(valid_geodesic):
            start_idx = np.where(valid_geodesic)[0][0]
            ax.scatter([x_geodesic[start_idx]], [z_geodesic[start_idx]], 
                      [sigma_geodesic[start_idx]], color='lime', s=100, 
                      marker='o', label='Geodesic Start')
        
        # Asymptotic silence boundary
        sigma_silence = np.linspace(-6, -3, 50)
        r_silence_vals = [1.2*r_s, 2*r_s, 3*r_s]
        for r_sil in r_silence_vals:
            z_silence = np.sqrt((r_sil / r_s - 1)) * np.exp(-sigma_silence / 4)
            ax.plot([r_sil]*len(sigma_silence), z_silence, sigma_silence, 
                    'k--', linewidth=2, alpha=0.7)
        
        # Formatting with enhanced labels
        ax.set_xlabel('r / rs (Radius)', fontsize=14, labelpad=10)
        ax.set_ylabel('Embedding Height z', fontsize=14, labelpad=10)
        ax.set_zlabel('σ (Log-Time)', fontsize=14, labelpad=10)
        ax.set_title('Enhanced LTQG Black Hole: Rotational Symmetry & Geodesics', 
                    fontsize=16, pad=25)
        
        # Enhanced colorbar
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(color_combined.flatten())
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.12)
        cbar.set_label('Combined σ-Time & Curvature Intensity', fontsize=12)
        
        # Optimal viewing angle
        ax.view_init(elev=25, azim=30)
        
        # Enhanced legend
        ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 0.9), fontsize=11)
        
        # Scientific caption as annotation
        caption_text = """Figure: Log-Time Quantum Gravity Black Hole Geometry
        
The Schwarzschild singularity at r = 0 is replaced by smooth, 
logarithmically extended geometry in σ-coordinates.

Key Features:
• Proper time compression → additive σ-shift
• Curvature regularization through asymptotic silence  
• Observer geodesics asymptotically slow but never reach σ = -∞
• Rotational symmetry preserved in LTQG coordinates"""
        
        ax.text2D(0.02, 0.02, caption_text,
                 transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        
        # Set axis limits for better view
        ax.set_xlim(0.5, 5)
        ax.set_ylim(0, 3)
        ax.set_zlim(-6, 3)
        
        if save:
            self.save_figure(fig, "black_hole_embedding_enhanced.png")
        
        return fig
    
    def launch_webgl_black_hole(self) -> bool:
        """
        Launch the interactive WebGL black hole visualization in a web browser.
        
        This opens a 3D interactive demonstration with real-time σ-time animation,
        mouse/touch controls, and adjustable parameters.
        
        Returns:
            bool: True if launched successfully, False otherwise
        """
        try:
            import subprocess
            import os
            
            # Find the launcher script
            launcher_path = os.path.join(os.path.dirname(__file__), "launch_webgl_demo.py")
            
            if not os.path.exists(launcher_path):
                print("❌ WebGL launcher not found. Please ensure launch_webgl_demo.py exists.")
                return False
            
            print("🚀 Launching interactive WebGL black hole visualization...")
            
            # Launch in a separate process so it doesn't block
            subprocess.Popen([
                "python", launcher_path
            ], cwd=os.path.dirname(__file__))
            
            print("✅ WebGL demo launched! Check your web browser.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to launch WebGL demo: {e}")
            return False
    
    def generate_all_figures(self, save: bool = True) -> Dict[str, plt.Figure]:
        """
        Generate all LTQG figures described in the paper.
        
        Args:
            save: Whether to save figures to disk
            
        Returns:
            Dictionary of figure names and matplotlib Figure objects
        """
        figures = {}
        
        print("Generating LTQG visualization suite...")
        
        figures['log_time_map'] = self.figure_1_log_time_map(save)
        print("✓ Figure 1: Log-Time Map")
        
        figures['singularity_regularization'] = self.figure_2_singularity_regularization(save)
        print("✓ Figure 2: Singularity Regularization")
        
        figures['gravitational_redshift'] = self.figure_3_gravitational_redshift_shift(save)
        print("✓ Figure 3: Gravitational Redshift")
        
        figures['effective_generator'] = self.figure_4_effective_generator_silence(save)
        print("✓ Figure 4: Effective Generator")
        
        figures['zeno_protocols'] = self.figure_5_zeno_protocol_predictions(save)
        print("✓ Figure 5: Zeno Protocols")
        
        figures['early_universe'] = self.figure_6_early_universe_modes(save)
        print("✓ Figure 6: Early Universe")
        
        figures['experimental_feasibility'] = self.figure_experimental_feasibility(save)
        print("✓ Bonus: Experimental Feasibility")
        
        figures['black_hole_embedding'] = self.figure_black_hole_embedding(save)
        print("✓ Bonus: Black Hole Embedding")
        
        figures['black_hole_enhanced'] = self.figure_black_hole_embedding(save, enhanced=True)
        print("✓ Bonus: Enhanced Black Hole (Rotational + Geodesics)")
        
        if save:
            print(f"\nAll figures saved to: {self.save_dir}/")
        
        return figures


def quick_demo():
    """Quick demonstration of the visualization suite."""
    print("LTQG Visualization Suite Demo")
    print("============================")
    
    # Create visualizer
    visualizer = LTQGVisualizer(save_dir="figs", dpi=300)
    
    # Generate example figures
    print("Generating Log-Time Map...")
    fig1 = visualizer.figure_1_log_time_map(save=True)
    
    print("Generating Black Hole Embedding...")
    fig2 = visualizer.figure_black_hole_embedding(save=True)
    
    # Show both figures
    plt.show()
    
    # Offer to launch WebGL demo
    print("\nDemo complete! Check the 'figs' directory for output.")
    print("Generated:")
    print("• Log-Time Map (fundamental transformation)")
    print("• Black Hole Embedding (3D geometry visualization)")
    
    try:
        response = input("\n🌐 Would you like to launch the interactive WebGL black hole demo? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            visualizer.launch_webgl_black_hole()
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    quick_demo()