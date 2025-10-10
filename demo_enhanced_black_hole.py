#!/usr/bin/env python3
"""
Enhanced Black Hole Visualization Demo for LTQG
==============================================

Demonstrates the enhanced 3D embedding visualization of black hole geometry
in log-time coordinates with rotational symmetry, curvature intensity mapping,
and observer geodesics showing regularized infall.

Features implemented based on Denzil's feedback:
1. 360° rotational symmetry for intuitive black hole representation
2. Curvature intensity overlay combining σ-time and curvature
3. Observer geodesic path showing regularized particle infall
4. Enhanced scientific annotations and professional formatting

Usage:
    python demo_enhanced_black_hole.py

Author: Denzil James Greenwood
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def demo_comparison():
    """
    Create side-by-side comparison of original and enhanced versions.
    """
    from ltqg_visualization import LTQGVisualizer
    
    print("Creating Enhanced LTQG Black Hole Visualization...")
    print("=" * 55)
    
    # Create visualizer
    visualizer = LTQGVisualizer(save_dir="figs", dpi=300)
    
    # Generate both versions
    print("Generating original version...")
    fig1 = visualizer.figure_black_hole_embedding(save=True, enhanced=False)
    
    print("Generating enhanced version with rotational symmetry...")
    fig2 = visualizer.figure_black_hole_embedding(save=True, enhanced=True)
    
    print("\n✓ Both visualizations created and saved!")
    
    # Display information about the enhancements
    print("\nEnhancements in the new version:")
    print("• 360° rotational symmetry for intuitive black hole geometry")
    print("• Curvature intensity overlay (1/(r-rs)² regularized)")
    print("• Observer geodesic showing regularized infall trajectory")
    print("• Enhanced scientific caption with detailed physics")
    print("• Circular cross-sections showing σ-time layers")
    print("• Professional formatting for publication quality")
    
    return fig1, fig2


def standalone_enhanced_demo():
    """
    Standalone enhanced black hole visualization.
    """
    # Parameters
    r_s = 1.0
    sigma = np.linspace(-6, 3, 200)
    r = np.linspace(r_s * 1.001, 5 * r_s, 150)
    
    # Create meshgrid
    R, S = np.meshgrid(r, sigma)
    Z = np.sqrt((R / r_s - 1)) * np.exp(-S / 4)
    
    # Curvature intensity
    curvature_intensity = 1.0 / ((R / r_s - 1)**2 + 0.01)
    curvature_normalized = (curvature_intensity - curvature_intensity.min()) / \
                          (curvature_intensity.max() - curvature_intensity.min())
    
    # Combined color mapping
    color_sigma = (S - S.min()) / (S.max() - S.min())
    color_combined = 0.7 * color_sigma + 0.3 * curvature_normalized
    color_map = plt.cm.plasma(color_combined)
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Main surface
    surf = ax.plot_surface(R, Z, S, facecolors=color_map, 
                          rstride=3, cstride=3, linewidth=0.5, 
                          antialiased=True, alpha=0.85)
    
    # Event horizon
    theta = np.linspace(0, 2*np.pi, 100)
    x_h = r_s * np.cos(theta)
    y_h = r_s * np.sin(theta)
    z_h = np.zeros_like(theta)
    ax.plot(x_h, y_h, z_h, 'r-', linewidth=4, label='Event Horizon')
    
    # Observer geodesic
    tau_geo = np.logspace(-4, 2, 100)
    sigma_geo = np.log(tau_geo)
    r_geo = r_s * (1 + 3 * np.exp(-tau_geo/10))
    x_geo = r_geo
    z_geo = np.sqrt((r_geo / r_s - 1)) * np.exp(-sigma_geo / 4)
    
    valid = (r_geo > r_s) & (sigma_geo > -6) & (sigma_geo < 3)
    ax.plot(x_geo[valid], z_geo[valid], sigma_geo[valid], 
            'lime', linewidth=4, label='Regularized Infall')
    
    # Formatting
    ax.set_xlabel('r / rs', fontsize=14)
    ax.set_ylabel('Embedding z', fontsize=14)
    ax.set_zlabel('σ (Log-Time)', fontsize=14)
    ax.set_title('Enhanced LTQG Black Hole Geometry', fontsize=16)
    ax.view_init(elev=25, azim=30)
    ax.legend()
    
    return fig


def scientific_explanation():
    """
    Print detailed scientific explanation of the visualization.
    """
    print("\nScientific Explanation of Enhanced LTQG Black Hole Visualization")
    print("=" * 70)
    
    explanation = """
🔬 PHYSICAL INTERPRETATION:

1. GEOMETRIC EMBEDDING:
   • The surface represents spacetime geometry in (r, z, σ) coordinates
   • r/rs: radial distance in Schwarzschild radius units
   • z: embedding height ~ √(r/rs - 1) × exp(-σ/4)
   • σ: log-time coordinate = log(τ/τ₀)

2. SINGULARITY REGULARIZATION:
   • Classical singularity at r = 0 → asymptotic boundary at σ → -∞
   • Curvature invariants: R ∝ 1/(r-rs)² → R ∝ exp(-2σ) (smooth)
   • "Asymptotic silence": quantum evolution halts as σ → -∞

3. GEODESIC REGULARIZATION:
   • Green curve: observer falling toward black hole
   • In τ-time: geodesic reaches singularity in finite proper time
   • In σ-time: geodesic asymptotically approaches σ = -∞ (never reaches)

4. ROTATIONAL SYMMETRY:
   • Full 3D structure would be surface of revolution around z-axis
   • Circular cross-sections show σ-time "layers"
   • Preserves spherical symmetry of original Schwarzschild solution

5. COLOR CODING:
   • Plasma colormap: σ-time coordinate (blue = early, yellow = late)
   • Intensity weighting: curvature magnitude 1/(r-rs)²
   • Shows how curvature smoothly regularizes in LTQG

🎯 KEY LTQG INSIGHTS:

✓ Horizon Regularity: Event horizon becomes coordinate-regular surface
✓ Singularity Resolution: Physical singularity replaced by asymptotic boundary
✓ Time Unification: Multiplicative dilation → additive σ-shifts
✓ Quantum Consistency: Smooth evolution in σ-time frame
✓ Geometric Continuity: No breakdown of spacetime structure

This visualization demonstrates LTQG's central claim: the log-time transformation
σ = log(τ/τ₀) naturally resolves spacetime singularities while preserving all
physical content of General Relativity in the regular regions.
    """
    
    print(explanation)


def main():
    """Main demo function with options."""
    print("Enhanced LTQG Black Hole Visualization Demo")
    print("=" * 45)
    
    # Choose demo type
    demo_type = input("\nChoose demo type:\n1. Framework comparison (both versions)\n2. Standalone enhanced\n3. Scientific explanation\nEnter choice (1-3): ").strip()
    
    if demo_type == "1":
        fig1, fig2 = demo_comparison()
        plt.show()
    elif demo_type == "2":
        fig = standalone_enhanced_demo()
        plt.show()
    elif demo_type == "3":
        scientific_explanation()
    else:
        print("Running all demos...")
        scientific_explanation()
        fig1, fig2 = demo_comparison()
        plt.show()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()