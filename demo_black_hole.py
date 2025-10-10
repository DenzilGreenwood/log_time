#!/usr/bin/env python3
"""
Black Hole Visualization Demo for LTQG
======================================

Demonstrates the 3D embedding visualization of black hole geometry
in log-time coordinates, showing how LTQG regularizes the singularity
and provides a smooth description near the event horizon.

Usage:
    python demo_black_hole.py

Author: Denzil James Greenwood
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ltqg_visualization import LTQGVisualizer


def standalone_black_hole_viz():
    """
    Standalone version of the black hole visualization.
    This can be run independently of the full LTQG framework.
    """
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

    return fig


def demo_with_framework():
    """
    Demo using the full LTQG visualization framework.
    """
    print("Creating LTQG Black Hole Visualization...")
    
    # Create visualizer
    visualizer = LTQGVisualizer(save_dir="figs", dpi=300)
    
    # Generate black hole visualization
    fig = visualizer.figure_black_hole_embedding(save=True)
    
    print("✓ Black hole visualization created and saved!")
    return fig


def main():
    """Main demo function."""
    print("LTQG Black Hole Visualization Demo")
    print("=" * 40)
    
    # Choose which demo to run
    use_framework = True  # Set to False for standalone version
    
    if use_framework:
        try:
            fig = demo_with_framework()
        except ImportError as e:
            print(f"Framework import failed: {e}")
            print("Falling back to standalone version...")
            fig = standalone_black_hole_viz()
    else:
        fig = standalone_black_hole_viz()
    
    # Display the plot
    plt.show()
    
    print("\nVisualization complete!")
    print("This shows how black hole geometry appears in LTQG log-time coordinates.")
    print("Key features:")
    print("• The embedding surface represents spacetime curvature")
    print("• The red line marks the event horizon")
    print("• The color map shows the log-time coordinate σ")
    print("• Early times (negative σ) show asymptotic silence")


if __name__ == "__main__":
    main()