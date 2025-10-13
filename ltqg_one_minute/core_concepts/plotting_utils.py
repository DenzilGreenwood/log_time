#!/usr/bin/env python3
"""
Plotting Utilities for LTQG Core Concepts

Shared utilities for organizing and saving plots in the proper results folder structure.
Each Python module gets its own subfolder under core_concepts/results/

Includes centralized config access to ensure plots reflect the exact same parameters
used in evolution computations.
"""

import os
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any

# Default LTQG configuration for plotting consistency
DEFAULT_PLOT_CONFIG = {
    'tau_0': 1.0,
    'envelope_type': 'tanh',
    'envelope_params': {
        'sigma_0': 2.0,
        'width': 1.0,
        'envelope_floor': 1e-8
    }
}

def get_shared_config() -> Dict[str, Any]:
    """
    Get shared LTQG configuration for plotting consistency.
    
    This ensures that all plotting routines use the same τ₀, envelope type,
    and envelope parameters as the evolution computations.
    
    Returns:
        Dictionary with shared configuration parameters
    """
    return DEFAULT_PLOT_CONFIG.copy()

def update_shared_config(config_updates: Dict[str, Any]) -> None:
    """
    Update the shared plotting configuration.
    
    Args:
        config_updates: Dictionary with configuration updates
    """
    global DEFAULT_PLOT_CONFIG
    DEFAULT_PLOT_CONFIG.update(config_updates)

def setup_results_directory(module_name: str) -> str:
    """
    Create the results directory structure for a given module.
    
    Args:
        module_name: Name of the Python module (without .py extension)
        
    Returns:
        Path to the results directory for this module
        
    Example:
        results_dir = setup_results_directory('asymptotic_silence')
        # Creates: core_concepts/results/asymptotic_silence/
    """
    results_dir = os.path.join('results', module_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def save_plot(fig_or_plt, filename: str, module_name: str, **kwargs) -> str:
    """
    Save a plot to the appropriate results directory.
    
    Args:
        fig_or_plt: Either matplotlib.pyplot or a Figure object
        filename: Name of the output file (with extension)
        module_name: Name of the Python module
        **kwargs: Additional arguments passed to savefig()
        
    Returns:
        Full path to the saved file
    """
    results_dir = setup_results_directory(module_name)
    output_path = os.path.join(results_dir, filename)
    
    # Set default kwargs for high-quality output
    save_kwargs = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'facecolor': 'white',
        'edgecolor': 'none'
    }
    save_kwargs.update(kwargs)
    
    if hasattr(fig_or_plt, 'savefig'):
        # It's pyplot or a figure
        fig_or_plt.savefig(output_path, **save_kwargs)
    else:
        # Assume it's pyplot
        plt.savefig(output_path, **save_kwargs)
    
    return output_path

def configure_matplotlib_for_export():
    """
    Configure matplotlib for non-interactive, high-quality export.
    Call this at the beginning of plotting scripts.
    """
    plt.switch_backend('Agg')  # Non-interactive backend
    
    # Set default style for publication-quality figures
    plt.rcParams.update({
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

def list_results_structure():
    """
    Display the current results folder structure.
    """
    results_base = 'results'
    if not os.path.exists(results_base):
        print("No results directory found.")
        return
    
    print("Results folder structure:")
    print(f"{results_base}/")
    
    for module_dir in sorted(os.listdir(results_base)):
        module_path = os.path.join(results_base, module_dir)
        if os.path.isdir(module_path):
            print(f"  {module_dir}/")
            
            # List files in each module directory
            files = [f for f in os.listdir(module_path) if f.endswith(('.png', '.pdf', '.svg', '.jpg'))]
            for file in sorted(files):
                print(f"    {file}")

if __name__ == "__main__":
    """
    Demo/test the plotting utilities.
    """
    print("=== LTQG Plotting Utilities Demo ===\n")
    
    # Configure matplotlib
    configure_matplotlib_for_export()
    
    # Create a test plot
    import numpy as np
    
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Test Plot')
    plt.legend()
    plt.grid(True)
    
    # Save using the utility
    output_path = save_plot(plt, 'test_plot.png', 'plotting_utils')
    plt.close()
    
    print(f"Test plot saved to: {output_path}")
    print()
    
    # Show results structure
    list_results_structure()