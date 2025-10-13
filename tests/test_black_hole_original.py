#!/usr/bin/env python3
"""
Simple test script for the black hole visualization code.
This runs the exact code provided by the user.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def test_original_code():
    """Test the original black hole visualization code."""
    
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

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    ax.plot_surface(R, Z, S, facecolors=color_map, rstride=4, cstride=4, linewidth=0, antialiased=False)
    ax.set_xlabel('r / r_s')
    ax.set_ylabel('Embedding z')
    ax.set_zlabel('σ (log time)')
    ax.set_title('Log-Time Black Hole Geometry (LTQG View)')
    
    # Save the figure
    plt.savefig('figs/black_hole_original_test.png', dpi=300, bbox_inches='tight')
    print("✓ Original black hole visualization code tested successfully!")
    print("✓ Saved as: figs/black_hole_original_test.png")
    
    return fig

if __name__ == "__main__":
    print("Testing original black hole visualization code...")
    fig = test_original_code()
    plt.show()