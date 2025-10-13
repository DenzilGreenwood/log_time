"""
σ-Time Transformation Utilities

This module provides comprehensive utilities for working with the logarithmic
time transformation σ ≡ log(τ/τ₀) and its applications in LTQG.

Key transformations:
- Proper time ↔ σ-time conversions
- Coordinate transformations in various spacetimes
- Differential operators in σ-coordinates
- Metric components and connection coefficients
"""

import numpy as np
from typing import Callable, Tuple, Union, Optional
from scipy.integrate import quad
from scipy.special import gamma, hyp2f1
import matplotlib.pyplot as plt


class SigmaTransformation:
    """
    Utilities for σ-time transformations and coordinate conversions.
    """
    
    def __init__(self, tau_0: float = 1.0, units: str = "geometrized"):
        """
        Initialize σ-transformation utilities.
        
        Args:
            tau_0: Reference time scale
            units: Unit system ("geometrized" for G=c=1, "SI" for SI units, "custom")
        """
        self.tau_0 = tau_0
        self.units = units
    
    def d_sigma_d_tau(self, tau: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute dσ/dτ = 1/τ.
        
        Args:
            tau: Proper time (must be positive)
            
        Returns:
            Derivative dσ/dτ
        """
        tau = np.asarray(tau)
        if np.any(tau <= 0):
            raise ValueError("Proper time τ must be positive")
        return 1.0 / tau
    
    def d_tau_d_sigma(self, sigma: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Compute dτ/dσ = τ = τ₀ exp(σ).
        
        Args:
            sigma: σ-time coordinate
            
        Returns:
            Derivative dτ/dσ
        """
        sigma = np.asarray(sigma)
        return self.tau_0 * np.exp(sigma)
    
    def transform_differential_operator(self, d_d_tau: Callable, 
                                      sigma: Union[float, np.ndarray]) -> Callable:
        """
        Transform differential operator from τ to σ coordinates.
        
        d/dτ → (dσ/dτ) d/dσ = (1/τ) d/dσ
        
        Args:
            d_d_tau: Differential operator in τ-coordinates
            sigma: σ-time coordinate(s)
            
        Returns:
            Transformed operator in σ-coordinates
        """
        tau = self.tau_0 * np.exp(sigma)
        return lambda f: d_d_tau(f) / tau
    
    def integrate_sigma_to_tau(self, f_sigma: Callable[[float], float], 
                              sigma_range: Tuple[float, float]) -> float:
        """
        Transform integral from σ to τ coordinates.
        
        ∫ f(σ) dσ = ∫ f(log(τ/τ₀)) dτ/τ
        
        Args:
            f_sigma: Function in σ-coordinates
            sigma_range: Integration range in σ
            
        Returns:
            Integral value
        """
        def integrand_tau(tau):
            sigma = np.log(tau / self.tau_0)
            return f_sigma(sigma) / tau
        
        tau_min = self.tau_0 * np.exp(sigma_range[0])
        tau_max = self.tau_0 * np.exp(sigma_range[1])
        
        result, _ = quad(integrand_tau, tau_min, tau_max)
        return result
    
    def cosmological_time_to_sigma(self, t_cosmic: Union[float, np.ndarray], 
                                  a_of_t: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
                                  H_0: float = 1.0) -> Union[float, np.ndarray]:
        """
        Convert cosmological time to σ-time for FLRW metrics.
        
        For FLRW: dτ = dt, so σ = log(t/t₀) for comoving observers.
        For proper time along worldlines: dτ = √(g₀₀) dt = dt/a(t) in conformal time.
        
        Args:
            t_cosmic: Cosmological time coordinate
            a_of_t: Scale factor as function of cosmic time
            H_0: Hubble constant normalization
            
        Returns:
            σ-time coordinate
        """
        t_cosmic = np.asarray(t_cosmic)
        
        # For comoving observers in FLRW, proper time equals cosmic time
        # σ = log(t/t₀) where t₀ is some reference time
        return np.log(t_cosmic / self.tau_0)
    
    def conformal_time_to_sigma(self, eta: Union[float, np.ndarray],
                               a_of_eta: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]) -> Union[float, np.ndarray]:
        """
        Convert conformal time η to σ-time.
        
        In conformal coordinates: ds² = a²(η)(-dη² + dx²)
        Proper time for comoving observers: dτ = a(η) dη
        
        Args:
            eta: Conformal time
            a_of_eta: Scale factor as function of conformal time
            
        Returns:
            σ-time coordinate
        """
        eta = np.asarray(eta)
        a_eta = a_of_eta(eta)
        
        # Proper time element: dτ = a(η) dη
        # Need to integrate to get total proper time
        def integrand(eta_prime):
            return a_of_eta(eta_prime)
        
        if np.isscalar(eta):
            tau, _ = quad(integrand, 0, eta)
            return np.log(tau / self.tau_0)
        else:
            # For array input, integrate for each point
            tau_array = np.zeros_like(eta)
            for i, eta_val in enumerate(eta):
                tau_array[i], _ = quad(integrand, 0, eta_val)
            return np.log(tau_array / self.tau_0)
    
    def schwarzschild_proper_time_to_sigma(self, t_coord: Union[float, np.ndarray], 
                                          r: float, M: float = 1.0) -> Union[float, np.ndarray]:
        """
        Convert Schwarzschild coordinate time to σ-time for static observers.
        
        For static observer at radius r:
        dτ = √(1 - 2M/r) dt (in geometrized units with G=c=1)
        
        Units: This method assumes geometrized units (G=c=1) by default.
        Check self.units field for the actual unit system in use.
        
        Args:
            t_coord: Schwarzschild coordinate time
            r: Radial coordinate (must be > 2M)
            M: Mass parameter (in geometrized units)
            
        Returns:
            σ-time coordinate
        """
        if r <= 2*M:
            raise ValueError("Radial coordinate must be outside event horizon (r > 2M)")
        
        # Units check
        if self.units != "geometrized":
            print(f"Warning: Using {self.units} units but method assumes geometrized (G=c=1)")
        
        t_coord = np.asarray(t_coord)
        redshift_factor = np.sqrt(1 - 2*M/r)  # geometrized units (G=c=1)
        
        # Proper time: τ = ∫ √(1 - 2M/r) dt = √(1 - 2M/r) × t
        tau = redshift_factor * t_coord
        
        return np.log(np.abs(tau) / self.tau_0)
    
    def kerr_proper_time_to_sigma(self, t_coord: Union[float, np.ndarray], 
                                 r: float, theta: float, 
                                 M: float = 1.0, a: float = 0.0) -> Union[float, np.ndarray]:
        """
        Convert Boyer-Lindquist time to σ-time for static observers in Kerr spacetime.
        
        Units: This method assumes geometrized units (G=c=1) by default.
        Check self.units field for the actual unit system in use.
        
        Args:
            t_coord: Boyer-Lindquist time coordinate
            r: Radial coordinate
            theta: Polar angle
            M: Mass parameter (geometrized units)
            a: Spin parameter (geometrized units)
            
        Returns:
            σ-time coordinate
        """
        # Units check
        if self.units != "geometrized":
            print(f"Warning: Using {self.units} units but method assumes geometrized (G=c=1)")
        
        t_coord = np.asarray(t_coord)
        
        # Kerr metric components
        rho_sq = r*r + a*a * np.cos(theta)**2
        Delta = r*r - 2*M*r + a*a
        
        if Delta <= 0:
            raise ValueError("Point inside (or at) horizon - no radial escape")
        
        # g_tt component for static observer
        g_tt = -(1 - 2*M*r/rho_sq)
        
        if g_tt >= 0:
            raise ValueError("Point inside ergosphere - no static observers (g_tt ≥ 0)")
        
        # Proper time element: dτ = √(-g_tt) dt
        redshift_factor = np.sqrt(-g_tt)
        tau = redshift_factor * t_coord
        
        return np.log(np.abs(tau) / self.tau_0)


class SigmaCoordinateSystem:
    """
    Coordinate system utilities for σ-time parametrized spacetimes.
    
    Note: We mix σ-time with standard spatial coordinates, setting g_σσ = -τ² 
    for proper-time parametrization. This assumes static observers at fixed 
    spatial coordinates following timelike worldlines parametrized by σ.
    """
    
    def __init__(self, transformation: SigmaTransformation):
        self.transform = transformation
    
    def flrw_metric_sigma(self, sigma: float, spatial_coords: np.ndarray,
                         a_of_sigma: Callable[[float], float],
                         K: float = 0) -> np.ndarray:
        """
        FLRW metric in σ-time coordinates.
        
        ds² = -dτ² + a²(τ)[dr²/(1-Kr²) + r²(dθ² + sin²θ dφ²)]
        → ds² = -τ²dσ² + a²(σ)[dr²/(1-Kr²) + r²(dθ² + sin²θ dφ²)]
        
        Args:
            sigma: σ-time coordinate
            spatial_coords: [r, θ, φ] spatial coordinates
            a_of_sigma: Scale factor as function of σ
            K: Spatial curvature parameter
            
        Returns:
            4x4 metric tensor
        """
        tau = self.transform.tau_0 * np.exp(sigma)
        a_sigma = a_of_sigma(sigma)
        r, theta, phi = spatial_coords
        
        # Metric components in σ-coordinates
        g = np.zeros((4, 4))
        
        # g_σσ = -τ²
        g[0, 0] = -tau*tau
        
        # g_rr = a²/(1-Kr²)
        g[1, 1] = a_sigma*a_sigma / (1 - K*r*r)
        
        # g_θθ = a²r²
        g[2, 2] = a_sigma*a_sigma * r*r
        
        # g_φφ = a²r²sin²θ
        g[3, 3] = a_sigma*a_sigma * r*r * np.sin(theta)**2
        
        return g
    
    def schwarzschild_metric_sigma(self, sigma: float, spatial_coords: np.ndarray,
                                  M: float = 1.0) -> np.ndarray:
        """
        Schwarzschild metric in σ-time coordinates for static observers.
        
        For static observer: dτ = √(1-2M/r) dt
        So dt = dτ/√(1-2M/r) = τ dσ/√(1-2M/r)
        
        Args:
            sigma: σ-time coordinate
            spatial_coords: [r, θ, φ] spatial coordinates
            M: Mass parameter
            
        Returns:
            4x4 metric tensor in mixed coordinates
        """
        tau = self.transform.tau_0 * np.exp(sigma)
        r, theta, phi = spatial_coords
        
        if r <= 2*M:
            raise ValueError("Radial coordinate inside horizon")
        
        f = 1 - 2*M/r
        
        g = np.zeros((4, 4))
        
        # For static observer: g_σσ = -τ² (proper time parametrization)
        g[0, 0] = -tau*tau
        
        # Spatial parts unchanged
        g[1, 1] = 1/f
        g[2, 2] = r*r
        g[3, 3] = r*r * np.sin(theta)**2
        
        return g
    
    def christoffel_symbols_sigma(self, sigma: float, metric_func: Callable,
                                 coords: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """
        Compute Christoffel symbols in σ-coordinate system.
        
        Γᵅ_μν = ½ gᵅᵝ (∂_μ g_βν + ∂_ν g_βμ - ∂_β g_μν)
        
        Args:
            sigma: σ-time coordinate
            metric_func: Function returning metric tensor
            coords: Coordinate values [σ, x¹, x², x³]
            delta: Finite difference step size
            
        Returns:
            4x4x4 array of Christoffel symbols
        """
        # Compute metric and its inverse
        g = metric_func(sigma, coords[1:])
        g_inv = np.linalg.inv(g)
        
        # Initialize Christoffel symbols
        Gamma = np.zeros((4, 4, 4))
        
        # Finite difference derivatives of metric
        for mu in range(4):
            for nu in range(4):
                for alpha in range(4):
                    # Compute derivatives
                    coords_plus = coords.copy()
                    coords_minus = coords.copy()
                    
                    if alpha == 0:  # σ-derivative
                        g_plus = metric_func(sigma + delta, coords[1:])
                        g_minus = metric_func(sigma - delta, coords[1:])
                    else:  # Spatial derivatives
                        coords_plus[alpha] += delta
                        coords_minus[alpha] -= delta
                        g_plus = metric_func(sigma, coords_plus[1:])
                        g_minus = metric_func(sigma, coords_minus[1:])
                    
                    dg_dalpha = (g_plus - g_minus) / (2 * delta)
                    
                    # Christoffel symbol formula
                    for beta in range(4):
                        Gamma[alpha, mu, nu] += 0.5 * g_inv[alpha, beta] * (
                            dg_dalpha[beta, nu] + dg_dalpha[beta, mu] - dg_dalpha[mu, nu]
                        )
        
        return Gamma


def demo_sigma_transformations():
    """
    Demonstrate σ-time transformations in various coordinate systems.
    """
    print("=== σ-Time Transformation Demo ===\n")
    
    transform = SigmaTransformation(tau_0=1.0)
    coord_system = SigmaCoordinateSystem(transform)
    
    # Demo 1: Basic transformation properties
    print("1. Basic Transformation Properties:")
    tau_test = np.array([0.1, 1.0, 10.0])
    sigma_test = transform.d_sigma_d_tau(tau_test) * 0.1  # Small increment
    
    for i, tau in enumerate(tau_test):
        dsigma_dtau = transform.d_sigma_d_tau(tau)
        dtau_dsigma = transform.d_tau_d_sigma(np.log(tau))
        print(f"   τ = {tau:.1f}: dσ/dτ = {dsigma_dtau:.3f}, dτ/dσ = {dtau_dsigma:.3f}")
    
    # Demo 2: Cosmological time conversion
    print("\n2. FLRW Cosmological Time:")
    
    def scale_factor_radiation(t):
        """Scale factor for radiation-dominated universe."""
        return np.sqrt(t)
    
    t_cosmic = np.array([0.1, 1.0, 10.0])
    sigma_cosmic = transform.cosmological_time_to_sigma(t_cosmic, scale_factor_radiation)
    
    for i, t in enumerate(t_cosmic):
        print(f"   t = {t:.1f} → σ = {sigma_cosmic[i]:.3f}")
    
    # Demo 3: Schwarzschild redshift
    print("\n3. Schwarzschild Redshift as σ-offset:")
    M = 1.0
    radii = [3*M, 10*M, 100*M]  # Different distances from black hole
    t_coord = 1.0
    
    for r in radii:
        try:
            sigma_r = transform.schwarzschild_proper_time_to_sigma(t_coord, r, M)
            redshift_factor = np.sqrt(1 - 2*M/r)
            print(f"   r = {r:.1f}M: redshift = {redshift_factor:.3f}, σ = {sigma_r:.3f}")
        except ValueError as e:
            print(f"   r = {r:.1f}M: {e}")
    
    # Demo 4: FLRW metric in σ-coordinates
    print("\n4. FLRW Metric in σ-coordinates:")
    
    def a_of_sigma_matter(sigma):
        """Scale factor in σ-time for matter-dominated universe."""
        tau = transform.tau_0 * np.exp(sigma)
        return tau**(2/3)  # a ∝ t^(2/3) for matter domination
    
    sigma_vals = [-1, 0, 1]
    spatial_coords = [1.0, np.pi/4, 0]  # r, θ, φ
    
    for sigma in sigma_vals:
        g = coord_system.flrw_metric_sigma(sigma, spatial_coords, a_of_sigma_matter)
        det_g = np.linalg.det(g)
        print(f"   σ = {sigma:2d}: det(g) = {det_g:.3e}, g₀₀ = {g[0,0]:.3f}")


if __name__ == "__main__":
    demo_sigma_transformations()