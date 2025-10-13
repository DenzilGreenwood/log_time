"""
FLRW Cosmology in σ-Time: Worked Example

This module implements FLRW cosmological models using LTQG's σ-time framework,
demonstrating how standard cosmological evolution appears in logarithmic time
coordinates and showcasing smooth early-σ asymptotics.

Key features:
- FLRW metrics in σ-coordinates
- Radiation, matter, and Λ-dominated epochs
- Curvature scalars vs σ analysis
- Asymptotic silence in early universe
- Phase transitions between epochs
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.special import hyp2f1, gamma
from typing import Callable, Optional, Tuple, Union, Dict, List
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Import from our core modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_concepts.ltqg_core import LTQGFramework
from core_concepts.sigma_transformation import SigmaTransformation
from core_concepts.asymptotic_silence import AsymptoticSilence


@dataclass
class FLRWCosmologyResults:
    """Results from FLRW cosmological analysis in σ-time."""
    sigma_array: np.ndarray
    tau_array: np.ndarray
    scale_factor: np.ndarray
    hubble_parameter: np.ndarray
    curvature_scalars: Dict[str, np.ndarray]
    density_parameters: Dict[str, np.ndarray]
    phase_transitions: List[Dict]
    asymptotic_behavior: Dict[str, np.ndarray]


class FLRWExample:
    """
    FLRW cosmological models in LTQG σ-time framework.
    """
    
    def __init__(self, ltqg_framework: Optional[LTQGFramework] = None):
        """
        Initialize FLRW cosmology example.
        
        Args:
            ltqg_framework: LTQG framework instance
        """
        self.ltqg = ltqg_framework or LTQGFramework()
        self.sigma_transform = SigmaTransformation(self.ltqg.tau_0)
        self.silence = AsymptoticSilence(self.ltqg.tau_0, self.ltqg.hbar)
        
        # Physical constants (natural units)
        self.c = 1.0  # Speed of light
        self.G = 1.0  # Newton's constant
        self.H_0 = 1.0  # Present-day Hubble parameter
        
    def scale_factor_analytic(self, tau: Union[float, np.ndarray], 
                            epoch: str, params: Optional[Dict] = None) -> Union[float, np.ndarray]:
        """
        Analytic scale factor solutions for different cosmological epochs.
        
        Args:
            tau: Proper time (cosmic time for FLRW)
            epoch: Cosmological epoch ('radiation', 'matter', 'lambda', 'inflation')
            params: Additional parameters
            
        Returns:
            Scale factor a(τ)
        """
        if params is None:
            params = {}
            
        tau = np.asarray(tau)
        tau = np.maximum(tau, 1e-10)  # Avoid τ = 0
        
        if epoch == 'radiation':
            # a ∝ t^(1/2) for radiation domination
            a_0 = params.get('a_0', 1.0)
            return a_0 * np.sqrt(tau / self.ltqg.tau_0)
            
        elif epoch == 'matter':
            # a ∝ t^(2/3) for matter domination
            a_0 = params.get('a_0', 1.0)
            return a_0 * (tau / self.ltqg.tau_0)**(2/3)
            
        elif epoch == 'lambda':
            # a ∝ exp(H_Λ t) for cosmological constant domination
            H_lambda = params.get('H_lambda', self.H_0)
            a_0 = params.get('a_0', 1.0)
            return a_0 * np.exp(H_lambda * (tau - self.ltqg.tau_0))
            
        elif epoch == 'inflation':
            # a ∝ exp(H_inf t) for inflation
            H_inf = params.get('H_inf', 10.0 * self.H_0)
            a_0 = params.get('a_0', 1e-30)
            return a_0 * np.exp(H_inf * (tau - self.ltqg.tau_0))
            
        elif epoch == 'stiff':
            # a ∝ t^(1/3) for stiff matter (p = ρ)
            a_0 = params.get('a_0', 1.0)
            return a_0 * (tau / self.ltqg.tau_0)**(1/3)
            
        else:
            raise ValueError(f"Unknown epoch: {epoch}")
    
    def scale_factor_sigma(self, sigma: Union[float, np.ndarray], 
                          epoch: str, params: Optional[Dict] = None) -> Union[float, np.ndarray]:
        """
        Scale factor in σ-time coordinates.
        
        a(σ) = a(τ(σ)) where τ = τ₀ exp(σ)
        
        Args:
            sigma: σ-time coordinate
            epoch: Cosmological epoch
            params: Additional parameters
            
        Returns:
            Scale factor a(σ)
        """
        tau = self.ltqg.tau_from_sigma(sigma)
        return self.scale_factor_analytic(tau, epoch, params)
    
    def hubble_parameter_sigma(self, sigma: Union[float, np.ndarray], 
                             epoch: str, params: Optional[Dict] = None) -> Union[float, np.ndarray]:
        """
        Hubble parameter H(σ) = (1/a)(da/dσ) in σ-time.
        
        Chain rule: da/dσ = (da/dτ)(dτ/dσ) = (da/dτ)τ
        So H(σ) = (1/a)(da/dτ)τ = H(τ)τ
        
        Args:
            sigma: σ-time coordinate
            epoch: Cosmological epoch
            params: Additional parameters
            
        Returns:
            Hubble parameter H(σ)
        """
        sigma = np.asarray(sigma)
        tau = self.ltqg.tau_from_sigma(sigma)
        a_sigma = self.scale_factor_sigma(sigma, epoch, params)
        
        # Analytic H(τ) for each epoch
        if epoch == 'radiation':
            H_tau = 1.0 / (2.0 * tau)
        elif epoch == 'matter':
            H_tau = 2.0 / (3.0 * tau)
        elif epoch == 'lambda':
            H_lambda = params.get('H_lambda', self.H_0)
            H_tau = H_lambda
        elif epoch == 'inflation':
            H_inf = params.get('H_inf', 10.0 * self.H_0)
            H_tau = H_inf
        elif epoch == 'stiff':
            H_tau = 1.0 / (3.0 * tau)
        else:
            raise ValueError(f"Unknown epoch: {epoch}")
        
        # H(σ) = H(τ) × τ
        return H_tau * tau
    
    def curvature_scalars_sigma(self, sigma: Union[float, np.ndarray], 
                              epoch: str, K: float = 0.0, 
                              params: Optional[Dict] = None) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute curvature scalars in σ-time for FLRW spacetime.
        
        Key scalars:
        - Ricci scalar R
        - Ricci squared R_μν R^μν  
        - Weyl squared C_μνρσ C^μνρσ (= 0 for FLRW)
        - Kretschmann scalar R_μνρσ R^μνρσ
        
        Args:
            sigma: σ-time coordinate
            epoch: Cosmological epoch
            K: Spatial curvature parameter
            params: Additional parameters
            
        Returns:
            Dictionary of curvature scalars
        """
        sigma = np.asarray(sigma)
        a_sigma = self.scale_factor_sigma(sigma, epoch, params)
        H_sigma = self.hubble_parameter_sigma(sigma, epoch, params)
        tau = self.ltqg.tau_from_sigma(sigma)
        
        # For FLRW metric: ds² = -dt² + a²(t)[dr²/(1-Kr²) + r²dΩ²]
        # In σ-coordinates: ds² = -τ²dσ² + a²(σ)[dr²/(1-Kr²) + r²dΩ²]
        
        # Hubble parameter in τ-coordinates
        H_tau = H_sigma / tau
        
        # Deceleration parameter
        if np.any(sigma > -10):  # Avoid numerical issues
            # q = -(1/H²)(d²a/dt²)(1/a) for cosmic time
            if epoch == 'radiation':
                q = 1.0
            elif epoch == 'matter':
                q = 0.5
            elif epoch == 'lambda':
                q = -1.0
            elif epoch == 'inflation':
                q = -1.0
            elif epoch == 'stiff':
                q = 2.0
            else:
                q = 0.0
        else:
            q = 0.0
        
        # Ricci scalar: R = 6[(d²a/dt²)/a + ((da/dt)/a)² + K/a²]
        # In τ-coordinates: R = 6[H'τ + H²τ² + K/a²]
        R_ricci = 6.0 * (H_tau * tau * (1 - q) + H_tau**2 * tau**2 + K / a_sigma**2)
        
        # Ricci tensor squared: R_μν R^μν
        # For FLRW: R_μν R^μν = 12H²(H'τ + H²τ² + K/a²)
        R_ricci_squared = 12.0 * H_tau**2 * tau**2 * (H_tau * tau * (1 - q) + H_tau**2 * tau**2 + K / a_sigma**2)
        
        # Weyl tensor (= 0 for FLRW due to maximal symmetry)
        C_weyl_squared = np.zeros_like(sigma)
        
        # Kretschmann scalar: R_μνρσ R^μνρσ
        # For FLRW: R_μνρσ R^μνρσ = 48H⁴ + 24H²K/a² + 12K²/a⁴
        R_kretschmann = (48.0 * H_tau**4 * tau**4 + 
                        24.0 * H_tau**2 * tau**2 * K / a_sigma**2 + 
                        12.0 * K**2 / a_sigma**4)
        
        return {
            'ricci_scalar': R_ricci,
            'ricci_squared': R_ricci_squared,
            'weyl_squared': C_weyl_squared,
            'kretschmann': R_kretschmann,
            'hubble_parameter': H_sigma,
            'deceleration_parameter': q * np.ones_like(sigma)
        }
    
    def density_parameters_sigma(self, sigma: Union[float, np.ndarray], 
                               cosmology_params: Dict) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute density parameters Ω_i(σ) for different matter components.
        
        Ω_i = ρ_i / ρ_critical where ρ_critical = 3H²/(8πG)
        
        Args:
            sigma: σ-time coordinate
            cosmology_params: Cosmological parameters
            
        Returns:
            Dictionary of density parameters
        """
        sigma = np.asarray(sigma)
        tau = self.ltqg.tau_from_sigma(sigma)
        
        # Present-day density parameters
        Omega_r0 = cosmology_params.get('Omega_r0', 5e-5)  # Radiation
        Omega_m0 = cosmology_params.get('Omega_m0', 0.3)   # Matter
        Omega_lambda0 = cosmology_params.get('Omega_lambda0', 0.7)  # Dark energy
        
        # Scale factor evolution
        a_sigma = self.scale_factor_sigma(sigma, 'matter')  # Use matter epoch as reference
        a_0 = 1.0  # Present-day scale factor
        
        # Density evolution: ρ ∝ a^(-3(1+w))
        # Radiation: w = 1/3 → ρ_r ∝ a^(-4)
        # Matter: w = 0 → ρ_m ∝ a^(-3)
        # Lambda: w = -1 → ρ_Λ = constant
        
        Omega_r = Omega_r0 * (a_0 / a_sigma)**4
        Omega_m = Omega_m0 * (a_0 / a_sigma)**3
        Omega_lambda = Omega_lambda0 * np.ones_like(sigma)
        
        # Total density parameter
        Omega_total = Omega_r + Omega_m + Omega_lambda
        
        return {
            'Omega_radiation': Omega_r,
            'Omega_matter': Omega_m,
            'Omega_lambda': Omega_lambda,
            'Omega_total': Omega_total
        }
    
    def phase_transitions_analysis(self, sigma_range: Tuple[float, float],
                                 cosmology_params: Dict) -> List[Dict]:
        """
        Analyze cosmological phase transitions in σ-time.
        
        Key transitions:
        1. Inflation → Radiation
        2. Radiation → Matter equality
        3. Matter → Dark energy domination
        
        Args:
            sigma_range: Range of σ-time to analyze
            cosmology_params: Cosmological parameters
            
        Returns:
            List of phase transition information
        """
        sigma_array = np.linspace(*sigma_range, 1000)
        density_params = self.density_parameters_sigma(sigma_array, cosmology_params)
        
        transitions = []
        
        # Find radiation-matter equality
        Omega_r = density_params['Omega_radiation']
        Omega_m = density_params['Omega_matter']
        
        # Find where Omega_r = Omega_m
        ratio = Omega_r / Omega_m
        equality_indices = np.where(np.diff(np.sign(ratio - 1.0)))[0]
        
        for idx in equality_indices:
            if idx < len(sigma_array) - 1:
                sigma_eq = sigma_array[idx]
                tau_eq = self.ltqg.tau_from_sigma(sigma_eq)
                a_eq = self.scale_factor_sigma(sigma_eq, 'matter')
                
                transitions.append({
                    'name': 'Radiation-Matter Equality',
                    'sigma_transition': sigma_eq,
                    'tau_transition': tau_eq,
                    'scale_factor': a_eq,
                    'redshift': (1.0 / a_eq) - 1.0
                })
        
        # Find matter-lambda equality
        Omega_lambda = density_params['Omega_lambda']
        lambda_ratio = Omega_m / Omega_lambda
        lambda_indices = np.where(np.diff(np.sign(lambda_ratio - 1.0)))[0]
        
        for idx in lambda_indices:
            if idx < len(sigma_array) - 1:
                sigma_lambda = sigma_array[idx]
                tau_lambda = self.ltqg.tau_from_sigma(sigma_lambda)
                a_lambda = self.scale_factor_sigma(sigma_lambda, 'matter')
                
                transitions.append({
                    'name': 'Matter-Lambda Equality',
                    'sigma_transition': sigma_lambda,
                    'tau_transition': tau_lambda,
                    'scale_factor': a_lambda,
                    'redshift': (1.0 / a_lambda) - 1.0
                })
        
        return transitions
    
    def asymptotic_analysis_sigma(self, sigma_range: Tuple[float, float],
                                epoch: str) -> Dict[str, np.ndarray]:
        """
        Analyze asymptotic behavior in σ-time coordinates.
        
        Key aspects:
        1. Early-σ asymptotics (σ → -∞, τ → 0⁺)
        2. Late-σ behavior (σ → +∞, τ → +∞)
        3. Curvature regularization
        4. Asymptotic silence effects
        
        Args:
            sigma_range: Range of σ-time
            epoch: Cosmological epoch
            
        Returns:
            Asymptotic analysis results
        """
        sigma_array = np.linspace(*sigma_range, 1000)
        tau_array = self.ltqg.tau_from_sigma(sigma_array)
        
        # Scale factor and Hubble parameter
        a_sigma = self.scale_factor_sigma(sigma_array, epoch)
        H_sigma = self.hubble_parameter_sigma(sigma_array, epoch)
        
        # Curvature scalars
        curvature = self.curvature_scalars_sigma(sigma_array, epoch)
        
        # Asymptotic silence envelope
        silence_factors = self.silence.silence_envelope(sigma_array)
        
        # Regularized curvature (with silence)
        R_regularized = curvature['ricci_scalar'] * silence_factors
        Kret_regularized = curvature['kretschmann'] * silence_factors**2
        
        return {
            'sigma_array': sigma_array,
            'tau_array': tau_array,
            'scale_factor': a_sigma,
            'hubble_parameter': H_sigma,
            'ricci_scalar': curvature['ricci_scalar'],
            'kretschmann': curvature['kretschmann'],
            'silence_factors': silence_factors,
            'ricci_regularized': R_regularized,
            'kretschmann_regularized': Kret_regularized
        }
    
    def complete_flrw_analysis(self, sigma_range: Tuple[float, float],
                             cosmology_params: Optional[Dict] = None) -> FLRWCosmologyResults:
        """
        Complete FLRW cosmological analysis in σ-time.
        
        Args:
            sigma_range: Range of σ-time for analysis
            cosmology_params: Cosmological parameters
            
        Returns:
            Complete FLRW analysis results
        """
        if cosmology_params is None:
            cosmology_params = {
                'Omega_r0': 5e-5,
                'Omega_m0': 0.3,
                'Omega_lambda0': 0.7,
                'h': 0.7
            }
        
        sigma_array = np.linspace(*sigma_range, 1000)
        tau_array = self.ltqg.tau_from_sigma(sigma_array)
        
        # Determine dominant epoch for each σ
        density_params = self.density_parameters_sigma(sigma_array, cosmology_params)
        
        # Use matter epoch as reference (can be improved with full evolution)
        scale_factor = self.scale_factor_sigma(sigma_array, 'matter')
        hubble_parameter = self.hubble_parameter_sigma(sigma_array, 'matter')
        
        # Curvature scalars
        curvature_scalars = self.curvature_scalars_sigma(sigma_array, 'matter')
        
        # Phase transitions
        transitions = self.phase_transitions_analysis(sigma_range, cosmology_params)
        
        # Asymptotic behavior
        asymptotic_behavior = self.asymptotic_analysis_sigma(sigma_range, 'matter')
        
        return FLRWCosmologyResults(
            sigma_array=sigma_array,
            tau_array=tau_array,
            scale_factor=scale_factor,
            hubble_parameter=hubble_parameter,
            curvature_scalars=curvature_scalars,
            density_parameters=density_params,
            phase_transitions=transitions,
            asymptotic_behavior=asymptotic_behavior
        )


def demo_flrw_cosmology():
    """
    Demonstrate FLRW cosmology in σ-time framework.
    """
    print("=== FLRW Cosmology in σ-Time Demo ===\n")
    
    # Initialize framework
    ltqg = LTQGFramework(tau_0=1.0)
    flrw = FLRWExample(ltqg)
    
    # Demo 1: Scale factor evolution
    print("1. Scale Factor Evolution in Different Epochs:")
    
    epochs = ['radiation', 'matter', 'lambda', 'inflation']
    sigma_test = np.array([-2, 0, 2])
    
    for epoch in epochs:
        print(f"   {epoch:10s}:")
        for sigma in sigma_test:
            tau = ltqg.tau_from_sigma(sigma)
            a_sigma = flrw.scale_factor_sigma(sigma, epoch)
            H_sigma = flrw.hubble_parameter_sigma(sigma, epoch)
            print(f"     σ = {sigma:2d}: τ = {tau:.3f}, a = {a_sigma:.3f}, H = {H_sigma:.3f}")
    
    # Demo 2: Curvature scalars
    print("\n2. Curvature Scalars vs σ-Time:")
    
    sigma_range = (-4, 2)
    sigma_array = np.linspace(*sigma_range, 100)
    
    for epoch in ['radiation', 'matter']:
        print(f"   {epoch.title()} epoch:")
        curvature = flrw.curvature_scalars_sigma(sigma_array, epoch)
        
        # Find maximum values
        max_ricci = np.max(np.abs(curvature['ricci_scalar']))
        max_kretsch = np.max(np.abs(curvature['kretschmann']))
        
        print(f"     Max |R|: {max_ricci:.2e}")
        print(f"     Max |Kretschmann|: {max_kretsch:.2e}")
        
        # Check early-σ behavior
        early_indices = sigma_array < -2
        if np.any(early_indices):
            early_ricci = curvature['ricci_scalar'][early_indices]
            print(f"     Early-σ Ricci range: [{np.min(early_ricci):.2e}, {np.max(early_ricci):.2e}]")
    
    # Demo 3: Phase transitions
    print("\n3. Cosmological Phase Transitions:")
    
    cosmology_params = {
        'Omega_r0': 5e-5,
        'Omega_m0': 0.3,
        'Omega_lambda0': 0.7
    }
    
    transitions = flrw.phase_transitions_analysis((-10, 5), cosmology_params)
    
    for transition in transitions:
        print(f"   {transition['name']}:")
        print(f"     σ = {transition['sigma_transition']:.3f}")
        print(f"     τ = {transition['tau_transition']:.3f}")
        print(f"     a = {transition['scale_factor']:.3f}")
        print(f"     z = {transition['redshift']:.1f}")
    
    # Demo 4: Asymptotic analysis
    print("\n4. Asymptotic Behavior Analysis:")
    
    asymptotic = flrw.asymptotic_analysis_sigma((-5, 3), 'matter')
    
    # Early-σ behavior
    early_mask = asymptotic['sigma_array'] < -3
    if np.any(early_mask):
        early_silence = asymptotic['silence_factors'][early_mask]
        early_curvature = asymptotic['ricci_regularized'][early_mask]
        
        print(f"   Early-σ (σ < -3):")
        print(f"     Silence factor range: [{np.min(early_silence):.3f}, {np.max(early_silence):.3f}]")
        print(f"     Regularized curvature max: {np.max(np.abs(early_curvature)):.2e}")
    
    # Late-σ behavior
    late_mask = asymptotic['sigma_array'] > 2
    if np.any(late_mask):
        late_hubble = asymptotic['hubble_parameter'][late_mask]
        late_scale = asymptotic['scale_factor'][late_mask]
        
        print(f"   Late-σ (σ > 2):")
        print(f"     Hubble parameter range: [{np.min(late_hubble):.2e}, {np.max(late_hubble):.2e}]")
        print(f"     Scale factor range: [{np.min(late_scale):.2e}, {np.max(late_scale):.2e}]")
    
    # Demo 5: Complete analysis
    print("\n5. Complete FLRW Analysis:")
    
    complete_results = flrw.complete_flrw_analysis((-6, 4), cosmology_params)
    
    print(f"   σ-range: [{complete_results.sigma_array[0]:.1f}, {complete_results.sigma_array[-1]:.1f}]")
    print(f"   τ-range: [{complete_results.tau_array[0]:.3f}, {complete_results.tau_array[-1]:.3f}]")
    print(f"   Scale factor range: [{np.min(complete_results.scale_factor):.2e}, {np.max(complete_results.scale_factor):.2e}]")
    print(f"   Number of phase transitions: {len(complete_results.phase_transitions)}")
    
    # Curvature analysis
    max_ricci = np.max(np.abs(complete_results.curvature_scalars['ricci_scalar']))
    max_kret = np.max(np.abs(complete_results.curvature_scalars['kretschmann']))
    
    print(f"   Max curvature scales:")
    print(f"     Ricci: {max_ricci:.2e}")
    print(f"     Kretschmann: {max_kret:.2e}")


def plot_flrw_cosmology_analysis():
    """
    Plot comprehensive FLRW cosmology analysis in σ-time.
    """
    ltqg = LTQGFramework()
    flrw = FLRWExample(ltqg)
    
    # Complete analysis
    cosmology_params = {
        'Omega_r0': 5e-5,
        'Omega_m0': 0.3,
        'Omega_lambda0': 0.7
    }
    
    results = flrw.complete_flrw_analysis((-6, 4), cosmology_params)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    sigma_array = results.sigma_array
    tau_array = results.tau_array
    
    # 1. σ vs τ relationship
    axes[0, 0].loglog(tau_array, np.abs(sigma_array), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Proper Time τ')
    axes[0, 0].set_ylabel('|σ-Time|')
    axes[0, 0].set_title('σ-Time vs Proper Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Scale factor evolution
    axes[0, 1].semilogy(sigma_array, results.scale_factor, 'r-', linewidth=2)
    axes[0, 1].set_xlabel('σ-Time')
    axes[0, 1].set_ylabel('Scale Factor a(σ)')
    axes[0, 1].set_title('Scale Factor Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Hubble parameter
    axes[0, 2].semilogy(sigma_array, np.abs(results.hubble_parameter), 'g-', linewidth=2)
    axes[0, 2].set_xlabel('σ-Time')
    axes[0, 2].set_ylabel('|H(σ)|')
    axes[0, 2].set_title('Hubble Parameter')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Curvature scalars
    axes[1, 0].semilogy(sigma_array, np.abs(results.curvature_scalars['ricci_scalar']), 
                       'purple', linewidth=2, label='Ricci')
    axes[1, 0].semilogy(sigma_array, np.abs(results.curvature_scalars['kretschmann']), 
                       'orange', linewidth=2, label='Kretschmann')
    axes[1, 0].set_xlabel('σ-Time')
    axes[1, 0].set_ylabel('Curvature Scalars')
    axes[1, 0].set_title('Curvature vs σ-Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Density parameters
    axes[1, 1].plot(sigma_array, results.density_parameters['Omega_radiation'], 
                   'red', linewidth=2, label='Radiation')
    axes[1, 1].plot(sigma_array, results.density_parameters['Omega_matter'], 
                   'blue', linewidth=2, label='Matter')
    axes[1, 1].plot(sigma_array, results.density_parameters['Omega_lambda'], 
                   'green', linewidth=2, label='Dark Energy')
    axes[1, 1].set_xlabel('σ-Time')
    axes[1, 1].set_ylabel('Density Parameters Ω_i')
    axes[1, 1].set_title('Density Parameter Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # 6. Asymptotic silence effects
    silence_factors = results.asymptotic_behavior['silence_factors']
    ricci_regularized = results.asymptotic_behavior['ricci_regularized']
    
    axes[1, 2].plot(sigma_array, silence_factors, 'cyan', linewidth=2, label='Silence Factor')
    ax_twin = axes[1, 2].twinx()
    ax_twin.semilogy(sigma_array, np.abs(ricci_regularized), 'magenta', linewidth=2, label='Regularized R')
    
    axes[1, 2].set_xlabel('σ-Time')
    axes[1, 2].set_ylabel('Silence Factor', color='cyan')
    ax_twin.set_ylabel('|Regularized Ricci|', color='magenta')
    axes[1, 2].set_title('Asymptotic Silence Effects')
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Different epochs comparison
    epochs = ['radiation', 'matter', 'lambda']
    colors = ['red', 'blue', 'green']
    
    for epoch, color in zip(epochs, colors):
        a_epoch = flrw.scale_factor_sigma(sigma_array, epoch)
        axes[2, 0].semilogy(sigma_array, a_epoch, color=color, linewidth=2, label=epoch.title())
    
    axes[2, 0].set_xlabel('σ-Time')
    axes[2, 0].set_ylabel('Scale Factor')
    axes[2, 0].set_title('Different Epochs Comparison')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Phase space (H vs a)
    axes[2, 1].loglog(results.scale_factor, np.abs(results.hubble_parameter), 
                     'black', linewidth=2)
    axes[2, 1].set_xlabel('Scale Factor a')
    axes[2, 1].set_ylabel('|Hubble Parameter H|')
    axes[2, 1].set_title('Cosmological Phase Space')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Early universe zoom
    early_mask = sigma_array < -2
    if np.any(early_mask):
        sigma_early = sigma_array[early_mask]
        curvature_early = np.abs(results.curvature_scalars['ricci_scalar'][early_mask])
        silence_early = silence_factors[early_mask]
        
        axes[2, 2].semilogy(sigma_early, curvature_early, 'red', linewidth=2, label='Ricci Scalar')
        ax_twin2 = axes[2, 2].twinx()
        ax_twin2.plot(sigma_early, silence_early, 'blue', linewidth=2, label='Silence')
        
        axes[2, 2].set_xlabel('σ-Time (Early)')
        axes[2, 2].set_ylabel('|Curvature|', color='red')
        ax_twin2.set_ylabel('Silence Factor', color='blue')
        axes[2, 2].set_title('Early Universe (σ < -2)')
        axes[2, 2].grid(True, alpha=0.3)
    
    # Mark phase transitions
    for transition in results.phase_transitions:
        sigma_trans = transition['sigma_transition']
        for ax in axes.flat:
            ax.axvline(sigma_trans, color='orange', linestyle='--', alpha=0.7)
    
    # Highlight asymptotic silence region
    silence_region = sigma_array < -3
    for ax in axes.flat:
        if np.any(silence_region):
            ax.axvspan(sigma_array[silence_region][0], -3, alpha=0.2, 
                      color='lightblue', label='Asymptotic Silence')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    demo_flrw_cosmology()
    
    # Uncomment to generate plots
    # fig = plot_flrw_cosmology_analysis()
    # plt.show()