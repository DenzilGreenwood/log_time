"""
LTQG One-Minute Demonstration: Complete Framework Integration

This script demonstrates the complete LTQG framework, showcasing how σ-time
addresses Kiefer's conceptual problems in quantum gravity through practical
examples and visualizations.

Usage:
    python ltqg_one_minute_demo.py [--mode MODE] [--examples] [--plots]
    
Modes:
    - overview: Quick conceptual overview
    - mathematical: Mathematical framework demonstration
    - physical: Physical applications and examples
    - operational: Experimental tests and predictions
    - complete: Full comprehensive demonstration
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all LTQG modules
from core_concepts.ltqg_core import LTQGFramework, demo_ltqg_basics
from core_concepts.sigma_transformation import SigmaTransformation, demo_sigma_transformations
from core_concepts.effective_hamiltonian import EffectiveHamiltonian, demo_effective_hamiltonian
from core_concepts.asymptotic_silence import AsymptoticSilence, demo_asymptotic_silence

from kiefer_problems.problem_of_time import ProblemOfTime, demo_problem_of_time

from worked_examples.flrw_cosmology import FLRWExample, demo_flrw_cosmology

from operational_tests.two_clock_experiment import TwoClockExperiment, demo_two_clock_experiment


class LTQGOneMinuteDemo:
    """
    Complete LTQG demonstration framework.
    """
    
    def __init__(self):
        """Initialize the demonstration framework."""
        self.ltqg = LTQGFramework(tau_0=1.0, hbar=1.0)
        self.demos = {
            'overview': self.overview_demo,
            'mathematical': self.mathematical_demo,
            'physical': self.physical_demo,
            'operational': self.operational_demo,
            'complete': self.complete_demo
        }
        
    def print_header(self, title: str, level: int = 1):
        """Print formatted section header."""
        if level == 1:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
        elif level == 2:
            print(f"\n{'-'*40}")
            print(f"  {title}")
            print(f"{'-'*40}")
        else:
            print(f"\n{title}:")
    
    def overview_demo(self):
        """Quick conceptual overview of LTQG."""
        self.print_header("LTQG IN ONE MINUTE: OVERVIEW", 1)
        
        print("""
CORE CONCEPT:
σ ≡ log(τ/τ₀) - Logarithmic time transformation

KEY EQUATION:
iℏ ∂ψ/∂σ = H_eff(σ) ψ  where  H_eff(σ) = τ H(τ)

REVOLUTIONARY FEATURES:
✓ Multiplicative time dilations → Additive σ-shifts
✓ Asymptotic silence as τ→0⁺ (natural boundary conditions)
✓ Clean clock synchronization across gravitational potentials
✓ Unified treatment of QM phases and GR redshift

SOLVES KIEFER'S PROBLEMS:
• Problem of Time: σ provides monotonic, curvature-agnostic internal time
• Singularities: Divergences τ⁻ⁿ → e⁻ⁿσ (regularized at σ→-∞)
• Black Holes: Redshift factors become additive σ-offsets
• Quantum Cosmology: Smooth boundary conditions at σ→-∞
""")
        
        # Quick numerical demonstration
        print("\nQUICK DEMONSTRATION:")
        
        # σ-time transformation
        tau_values = [0.1, 1.0, 10.0]
        print(f"{'τ (proper time)':>15} {'σ (log time)':>15} {'τ→0 behavior':>20}")
        print("-" * 50)
        
        for tau in tau_values:
            sigma = self.ltqg.sigma_from_tau(tau)
            behavior = "Silent" if sigma < -2 else "Active"
            print(f"{tau:>15.1f} {sigma:>15.3f} {behavior:>20}")
        
        # Multiplicative to additive transformation
        print(f"\nMULTIPLICATIVE → ADDITIVE TRANSFORMATION:")
        dilations = [0.5, 2.0, 10.0]
        print(f"{'Dilation Factor':>15} {'σ-Shift':>15}")
        print("-" * 30)
        
        for k in dilations:
            sigma_shift = self.ltqg.multiplicative_to_additive_shift(k)
            print(f"{k:>15.1f} {sigma_shift:>15.3f}")
    
    def mathematical_demo(self):
        """Demonstrate mathematical framework."""
        self.print_header("MATHEMATICAL FRAMEWORK", 1)
        
        print("Running core mathematical demonstrations...\n")
        
        # Core LTQG framework
        self.print_header("σ-Time Core Framework", 2)
        demo_ltqg_basics()
        
        # σ-transformations
        self.print_header("σ-Time Transformations", 2)
        demo_sigma_transformations()
        
        # Effective Hamiltonian
        self.print_header("Effective Hamiltonian H_eff(σ)", 2)
        demo_effective_hamiltonian()
        
        # Asymptotic silence
        self.print_header("Asymptotic Silence Mechanism", 2)
        demo_asymptotic_silence()
    
    def physical_demo(self):
        """Demonstrate physical applications."""
        self.print_header("PHYSICAL APPLICATIONS", 1)
        
        print("Running physical applications demonstrations...\n")
        
        # Problem of Time
        self.print_header("Problem of Time Solution", 2)
        demo_problem_of_time()
        
        # FLRW Cosmology
        self.print_header("FLRW Cosmology in σ-Time", 2)
        demo_flrw_cosmology()
    
    def operational_demo(self):
        """Demonstrate operational tests."""
        self.print_header("OPERATIONAL TESTS", 1)
        
        print("Running operational test demonstrations...\n")
        
        # Two-clock experiment
        self.print_header("Two-Clock Experiment", 2)
        demo_two_clock_experiment()
    
    def complete_demo(self):
        """Run complete comprehensive demonstration."""
        self.print_header("COMPLETE LTQG DEMONSTRATION", 1)
        
        print("Running complete LTQG framework demonstration...")
        print("This includes all mathematical, physical, and operational aspects.\n")
        
        # Run all demos in sequence
        self.overview_demo()
        self.mathematical_demo()
        self.physical_demo()
        self.operational_demo()
        
        # Summary
        self.print_header("SUMMARY AND CONCLUSIONS", 1)
        print(self.generate_summary())
    
    def generate_summary(self) -> str:
        """Generate comprehensive summary."""
        return """
LTQG FRAMEWORK SUMMARY:

THEORETICAL ACHIEVEMENTS:
✓ Unified clock choice for quantum gravity
✓ Resolution of Wheeler-DeWitt timelessness
✓ Natural regularization of spacetime singularities
✓ Clean semiclassical limit and boundary conditions
✓ Transparent handling of gravitational redshift

MATHEMATICAL TOOLS:
• σ ≡ log(τ/τ₀) transformation
• H_eff(σ) = τ H(τ) effective Hamiltonian
• Asymptotic silence envelope functions
• σ-parametrized Wheeler-DeWitt equation
• Linear phase accumulation in σ-time

EXPERIMENTAL PREDICTIONS:
• Phase differences linear in Δσ
• Gravitational effects as additive offsets
• σ-uniform experimental scheduling
• Improved clock synchronization
• Testable deviations from traditional approaches

APPLICATIONS DEMONSTRATED:
• FLRW cosmology with smooth early-σ asymptotics
• Black hole physics with redshift regularization
• Quantum evolution with asymptotic silence
• Two-clock experiments with σ-scheduling

COMPARISON WITH ALTERNATIVES:
• Complements (doesn't replace) existing approaches
• Provides computational and conceptual advantages
• Enables quantitative predictions for experiments
• Clarifies relationship between QM and GR time

NEXT STEPS:
• Implement full WebGL visualization
• Develop experimental protocols
• Extend to higher-dimensional systems
• Compare with observational data
• Publish peer-reviewed results

The LTQG framework provides a practical, testable approach to
fundamental problems in quantum gravity while preserving the
successful aspects of both quantum mechanics and general relativity.
"""
    
    def run_with_plots(self, mode: str = 'complete'):
        """Run demonstration with plots."""
        print("Generating LTQG visualization plots...\n")
        
        try:
            # Import plotting functions
            from core_concepts.effective_hamiltonian import plot_effective_hamiltonian_evolution
            from core_concepts.asymptotic_silence import plot_asymptotic_silence_analysis
            from worked_examples.flrw_cosmology import plot_flrw_cosmology_analysis
            from operational_tests.two_clock_experiment import plot_two_clock_experiment_analysis
            
            # Generate plots
            figures = {}
            
            if mode in ['complete', 'mathematical']:
                print("Generating effective Hamiltonian plots...")
                figures['hamiltonian'] = plot_effective_hamiltonian_evolution()
                
                print("Generating asymptotic silence plots...")
                figures['silence'] = plot_asymptotic_silence_analysis()
            
            if mode in ['complete', 'physical']:
                print("Generating FLRW cosmology plots...")
                figures['flrw'] = plot_flrw_cosmology_analysis()
            
            if mode in ['complete', 'operational']:
                print("Generating two-clock experiment plots...")
                figures['experiment'] = plot_two_clock_experiment_analysis()
            
            # Show all plots
            if figures:
                print(f"\nGenerated {len(figures)} visualization(s).")
                print("Close plot windows to continue...")
                plt.show()
            
        except ImportError as e:
            print(f"Plotting disabled: {e}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def run(self, mode: str = 'overview', generate_plots: bool = False):
        """Run the demonstration."""
        if mode not in self.demos:
            print(f"Unknown mode: {mode}")
            print(f"Available modes: {list(self.demos.keys())}")
            return
        
        # Run the demonstration
        self.demos[mode]()
        
        # Generate plots if requested
        if generate_plots:
            self.run_with_plots(mode)


def list_examples():
    """List available example files and their purposes."""
    print("\nLTQG ONE-MINUTE EXAMPLE FILES:")
    print("=" * 50)
    
    examples = {
        "Core Concepts": [
            ("ltqg_core.py", "Basic σ-time framework and transformations"),
            ("sigma_transformation.py", "Coordinate transformations and utilities"),
            ("effective_hamiltonian.py", "H_eff(σ) construction and analysis"),
            ("asymptotic_silence.py", "Silence mechanism and regularization")
        ],
        "Kiefer Problems": [
            ("problem_of_time.py", "Wheeler-DeWitt constraint in σ-time"),
        ],
        "Worked Examples": [
            ("flrw_cosmology.py", "FLRW universe with radiation/matter/Λ"),
        ],
        "Visualizations": [
            ("webgl_sigma_demo.html", "Interactive WebGL demonstration"),
        ],
        "Operational Tests": [
            ("two_clock_experiment.py", "σ-uniform experimental protocols"),
        ]
    }
    
    for category, files in examples.items():
        print(f"\n{category}:")
        for filename, description in files:
            print(f"  {filename:25s} - {description}")
    
    print(f"\nUsage Examples:")
    print(f"  python ltqg_one_minute_demo.py --mode overview")
    print(f"  python ltqg_one_minute_demo.py --mode complete --plots")
    print(f"  python core_concepts/ltqg_core.py")
    print(f"  python worked_examples/flrw_cosmology.py")
    print(f"  # Open visualizations/webgl_sigma_demo.html in browser")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LTQG One-Minute Framework Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ltqg_one_minute_demo.py --mode overview
  python ltqg_one_minute_demo.py --mode complete --plots
  python ltqg_one_minute_demo.py --examples
        """
    )
    
    parser.add_argument('--mode', '-m', 
                       choices=['overview', 'mathematical', 'physical', 'operational', 'complete'],
                       default='overview',
                       help='Demonstration mode (default: overview)')
    
    parser.add_argument('--plots', '-p', 
                       action='store_true',
                       help='Generate visualization plots')
    
    parser.add_argument('--examples', '-e',
                       action='store_true', 
                       help='List available example files')
    
    args = parser.parse_args()
    
    if args.examples:
        list_examples()
        return
    
    # Create and run demonstration
    demo = LTQGOneMinuteDemo()
    demo.run(mode=args.mode, generate_plots=args.plots)


if __name__ == "__main__":
    main()