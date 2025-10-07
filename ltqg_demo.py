"""
Log-Time Quantum Gravity: Complete Demonstration
================================================

Author: Denzil James Greenwood
GitHub: https://github.com/DenzilGreenwood/log_time
License: MIT

This script provides a comprehensive demonstration of the LTQG framework,
generating all figures from the paper and showcasing the experimental
predictions described in the theory.

Usage:
    python ltqg_demo.py --generate-all
    python ltqg_demo.py --figures-only
    python ltqg_demo.py --experiments-only
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Import LTQG modules
try:
    from ltqg_core import create_ltqg_simulator, LTQGConfig
    from ltqg_visualization import LTQGVisualizer
    from ltqg_experiments import ExperimentalSuite
except ImportError as e:
    print(f"Error importing LTQG modules: {e}")
    print("Make sure all LTQG Python files are in the same directory.")
    sys.exit(1)


def print_header():
    """Print the main header for the demonstration."""
    print("=" * 60)
    print("LOG-TIME QUANTUM GRAVITY (LTQG) DEMONSTRATION")
    print("=" * 60)
    print("A reparameterization approach to temporal unification")
    print("in General Relativity and Quantum Mechanics")
    print()
    print("σ = log(τ/τ₀) - Converting multiplicative time dilation")
    print("into additive phase shifts")
    print("=" * 60)
    print()


def demonstrate_core_concepts():
    """Demonstrate the core mathematical concepts of LTQG."""
    print("🔬 CORE LTQG CONCEPTS")
    print("-" * 30)
    
    # Create simulator
    simulator = create_ltqg_simulator(tau0=1.0)
    
    # 1. Time transformation
    print("1. Time Transformation σ = log(τ/τ₀)")
    tau_examples = np.array([1e-6, 1e-3, 1.0, 1e3, 1e6])
    sigma_examples = simulator.time_transform.sigma_from_tau(tau_examples)
    
    print("   τ/τ₀        →    σ")
    for tau, sigma in zip(tau_examples, sigma_examples):
        print(f"   {tau:8.1e}   →   {sigma:8.2f}")
    print()
    
    # 2. Singularity regularization
    print("2. Singularity Regularization")
    sigma_sing = np.array([-10, -5, 0, 5])
    curvature = simulator.singularity.curvature_scalar(sigma_sing)
    
    print("   σ        R(σ) ∝ exp(-2σ)")
    for s, r in zip(sigma_sing, curvature):
        print(f"   {s:5.1f}   →   {r:8.2e}")
    print()
    
    # 3. Redshift as additive shift
    print("3. Gravitational Redshift as Additive σ-Shift")
    alpha_examples = np.array([1.0, 0.5, 0.1, 0.01])
    sigma_shifts = simulator.redshift.sigma_shift_gravitational(alpha_examples)
    
    print("   α       Δσ = log(α)")
    for alpha, delta_sigma in zip(alpha_examples, sigma_shifts):
        print(f"   {alpha:4.2f}   →   {delta_sigma:8.2f}")
    print()
    
    # 4. Asymptotic silence
    print("4. Asymptotic Silence Condition")
    print("   As σ → -∞: K(σ) = τ₀ exp(σ) H → 0")
    print("   Generator vanishes, quantum evolution halts")
    print()


def generate_paper_figures(save_dir: str = "figs"):
    """Generate all figures from the LTQG paper."""
    print("📊 GENERATING PAPER FIGURES")
    print("-" * 30)
    
    # Create visualizer
    visualizer = LTQGVisualizer(save_dir=save_dir, dpi=300)
    
    print("Creating publication-quality figures...")
    start_time = time.time()
    
    # Generate all figures
    figures = visualizer.generate_all_figures(save=True)
    
    end_time = time.time()
    print(f"✅ Generated {len(figures)} figures in {end_time - start_time:.1f} seconds")
    print(f"📁 Figures saved to: {os.path.abspath(save_dir)}/")
    print()
    
    # List generated figures
    print("Generated figures:")
    for i, (name, fig) in enumerate(figures.items(), 1):
        filename = name.replace('_', ' ').title()
        print(f"   {i}. {filename}")
    print()
    
    return figures


def run_experimental_analysis():
    """Run comprehensive experimental analysis."""
    print("🧪 EXPERIMENTAL PREDICTIONS")
    print("-" * 30)
    
    # Create experimental suite
    suite = ExperimentalSuite()
    
    print("Analyzing LTQG experimental signatures...")
    
    # Generate comprehensive analysis
    start_time = time.time()
    results = suite.run_comprehensive_analysis()
    end_time = time.time()
    
    print(f"✅ Analysis completed in {end_time - start_time:.2f} seconds")
    print()
    
    # Print summary
    summary = suite.generate_experimental_summary()
    print(summary)
    
    # Detailed analysis of most promising experiment
    print("🎯 MOST PROMISING EXPERIMENT DETAILS")
    print("-" * 40)
    
    # Find experiment with highest distinguishability
    best_experiment = max(results.items(), 
                         key=lambda x: x[1]['distinguishability'])
    exp_name, exp_result = best_experiment
    
    print(f"Experiment: {exp_name.upper()}")
    setup = exp_result['setup']
    ltqg_pred = exp_result['ltqg_prediction']
    qm_pred = exp_result['qm_prediction']
    distinguishability = exp_result['distinguishability']
    
    print(f"Environment: {setup.environment}")
    print(f"Feasibility Score: {setup.feasibility_score:.2f}/1.0")
    print(f"Required Precision: {setup.precision:.2e}")
    print(f"Experiment Duration: {setup.duration:.2e} seconds")
    print(f"Statistical Significance: {distinguishability:.2f}σ")
    print()
    
    # Show prediction comparison
    print("Prediction Comparison:")
    for key in ltqg_pred.keys():
        if isinstance(ltqg_pred[key], (int, float)) and key != 'protocol':
            ltqg_val = ltqg_pred[key]
            qm_val = qm_pred.get(key, 0)
            if qm_val != 0:
                diff_percent = abs(ltqg_val - qm_val) / abs(qm_val) * 100
                print(f"  {key}: LTQG={ltqg_val:.6f}, QM={qm_val:.6f} ({diff_percent:.2f}% diff)")
            else:
                print(f"  {key}: LTQG={ltqg_val:.6f}, QM={qm_val:.6f}")
    print()
    
    return results


def create_theoretical_overview():
    """Create a theoretical overview of LTQG."""
    print("📚 THEORETICAL OVERVIEW")
    print("-" * 30)
    
    overview = """
Key Theoretical Insights:

1. TEMPORAL UNIFICATION
   • GR: Time dilation τ_B = α τ_A (multiplicative)
   • QM: Phase evolution e^{iφ} (additive)
   • LTQG: σ_B = σ_A + log(α) (unified additive structure)

2. SINGULARITY RESOLUTION
   • Classical: Q(τ) ∝ 1/τ^n → ∞ as τ → 0
   • LTQG: Q(σ) ∝ exp(-nσ) → 0 as σ → -∞
   • No more spacetime singularities!

3. MODIFIED QUANTUM EVOLUTION
   • Standard: iℏ ∂|ψ⟩/∂τ = H|ψ⟩
   • LTQG: iℏ ∂|ψ⟩/∂σ = K(σ)|ψ⟩ = τ₀ exp(σ) H|ψ⟩
   • Generator vanishes as σ → -∞ (asymptotic silence)

4. PHYSICAL PREDICTIONS
   • σ-uniform measurement protocols show redshift dependence
   • Gravitational interferometry with additive phase shifts
   • Early universe modes regularized without trans-Planckian problems
   • Clock transport experiments reveal path-dependent phases

5. GAUGE INVARIANCE
   • Physical predictions independent of τ₀ choice
   • Total proper time evolution: ∫K(σ)dσ = H∫dτ = H(τf - τi)
   • τ₀ acts like "units constant" for comparing σ-protocols
"""
    
    print(overview)


def demonstrate_numerical_examples():
    """Show concrete numerical examples of LTQG effects."""
    print("🔢 NUMERICAL EXAMPLES")
    print("-" * 30)
    
    simulator = create_ltqg_simulator()
    
    print("Example 1: Black Hole Approach")
    print("   At r = 2rs (outside horizon):")
    rs = 1.0
    r = 2.0 * rs
    alpha = simulator.redshift.redshift_factor_schwarzschild(np.array([r]), rs)[0]
    sigma_shift = simulator.redshift.sigma_shift_gravitational(np.array([alpha]))[0]
    print(f"   • Redshift factor α = {alpha:.6f}")
    print(f"   • σ-shift = log(α) = {sigma_shift:.6f}")
    print(f"   • Clock rate suppression: {alpha:.1%}")
    print()
    
    print("Example 2: Early Universe Regularization")
    print("   At τ = 10^(-6) τ₀ (early time):")
    tau_early = 1e-6
    sigma_early = simulator.time_transform.sigma_from_tau(np.array([tau_early]))[0]
    curvature_early = simulator.singularity.curvature_scalar(np.array([sigma_early]))[0]
    print(f"   • Log-time σ = {sigma_early:.2f}")
    print(f"   • Curvature R(σ) = {curvature_early:.2e}")
    print(f"   • Classical R(τ) ∝ 1/τ² = {1/tau_early**2:.2e} (divergent!)")
    print()
    
    print("Example 3: Generator Suppression")
    print("   Evolution generator magnitude |K(σ)|:")
    sigma_values = [-10, -5, 0, 5]
    H_magnitude = 1.0
    for sigma in sigma_values:
        K_mag = simulator.config.tau0 * np.exp(sigma) * H_magnitude
        print(f"   • σ = {sigma:3d}: |K| = {K_mag:.2e}")
    print()


def performance_benchmark():
    """Benchmark the computational performance of LTQG calculations."""
    print("⚡ PERFORMANCE BENCHMARK")
    print("-" * 30)
    
    simulator = create_ltqg_simulator()
    
    # Time various operations
    operations = {
        "Time transformation (1M points)": lambda: simulator.time_transform.sigma_from_tau(
            np.logspace(-6, 6, 1000000)),
        "Singularity regularization (100K points)": lambda: simulator.singularity.curvature_scalar(
            np.linspace(-20, 10, 100000)),
        "Redshift calculation (10K points)": lambda: simulator.redshift.redshift_factor_schwarzschild(
            np.linspace(1.01, 100, 10000), 1.0),
        "Early universe simulation": lambda: simulator.simulate_early_universe((-15, 5), 10000),
        "Black hole approach": lambda: simulator.simulate_black_hole_approach((1.01, 10), 1.0, 10000)
    }
    
    print("Operation timing:")
    for name, operation in operations.items():
        start_time = time.time()
        operation()
        end_time = time.time()
        print(f"   {name}: {end_time - start_time:.4f} seconds")
    print()


def create_comparison_table():
    """Create a comparison table between LTQG and other quantum gravity approaches."""
    print("📋 COMPARISON WITH OTHER QUANTUM GRAVITY THEORIES")
    print("-" * 50)
    
    comparison = """
| Aspect              | LTQG              | Loop QG           | String Theory     | CDT              |
|---------------------|-------------------|-------------------|-------------------|------------------|
| Background Indep.   | ✓ (σ-coord)      | ✓                 | ✗                 | ✓                |
| Singularity Res.    | ✓ (asymptotic)   | ✓ (bounce)        | ✓ (extended obj)  | ✓ (discrete)     |
| Unitarity           | ✓ (preserved)    | ✓                 | ? (black holes)   | ✓                |
| Testable Predict.   | ✓ (σ-protocols)  | ? (Planck scale)  | ? (extra dims)    | ? (discrete)     |
| Mathematical Simp.  | ✓ (coordinate)   | ✗ (complex)       | ✗ (very complex)  | ✗ (computational)|
| Classical Limit     | ✓ (exact GR)     | ✓ (approximate)   | ✓ (low energy)    | ✓ (large scale)  |
| QFT Compatibility   | ✓ (natural)      | ? (difficult)     | ✓ (string field)  | ? (emergence)    |
| Current Status      | Proposal         | Developed         | Mature            | Active research  |

Key LTQG Advantages:
• Minimal modification (just time coordinate)
• Exact preservation of GR and QM in respective limits
• Clear experimental predictions accessible with current/near-future technology
• No additional assumptions about space quantization or extra dimensions
• Natural unification of temporal structures
"""
    
    print(comparison)


def run_interferometry_parameter_sweep():
    """Run parameter sweep analysis for interferometry experiment."""
    print("\n" + "="*80)
    print("INTERFEROMETRY PARAMETER SWEEP ANALYSIS")
    print("="*80)
    
    # Create simulator and experimental suite
    simulator = create_ltqg_simulator(tau0=1.0)  # Use standard tau0 value
    exp_suite = ExperimentalSuite(simulator)
    
    # Access the interferometry protocol
    interferometry = exp_suite.interferometry_protocol
    
    # Create base experimental setup (same as before)
    from ltqg_experiments import InterferometryExperiment
    base_setup = InterferometryExperiment(
        duration=1000.0,           # 1000 seconds
        precision=1e-18,           # LIGO-class precision
        environment="vacuum",      # Ultra-high vacuum
        feasibility_score=0.8,     # Challenging but feasible
        path_length=4000.0,        # 4 km arm length (LIGO-scale)
        wavelength=1064e-9,        # Nd:YAG laser (1064 nm)
        redshift_gradient=1e-15,   # Weak gravitational gradient
        beam_splitter_efficiency=0.99
    )
    
    print(f"Base experimental setup:")
    print(f"  Path length: {base_setup.path_length/1000:.1f} km")
    print(f"  Wavelength: {base_setup.wavelength*1e9:.0f} nm")
    print(f"  Redshift gradient: {base_setup.redshift_gradient:.1e}")
    print(f"  Duration: {base_setup.duration} s")
    
    # Get baseline predictions
    ltqg_base = interferometry.predict_signal(base_setup)
    qm_base = interferometry.predict_standard_qm(base_setup)
    base_distinguishability = interferometry.distinguishability(base_setup)
    
    print(f"\nBaseline Results:")
    print(f"  LTQG phase difference: {ltqg_base['phase_difference']:.6f} rad")
    print(f"  QM phase difference: {qm_base['phase_difference']:.6f} rad")
    print(f"  Relative difference: {abs(ltqg_base['phase_difference'] - qm_base['phase_difference'])/abs(qm_base['phase_difference'])*100:.1f}%")
    print(f"  Distinguishability: {base_distinguishability:.2e} σ")
    
    # Run parameter sweeps
    sweep_configs = [
        {
            'name': 'τ₀ Scaling',
            'type': 'tau0',
            'range': (0.1, 10.0),
            'points': 15,
            'description': 'Test if differences scale with fundamental time constant'
        },
        {
            'name': 'Redshift Gradient',
            'type': 'redshift_gradient', 
            'range': (1e-16, 1e-14),
            'points': 15,
            'description': 'Test scaling with gravitational field strength'
        },
        {
            'name': 'Path Length',
            'type': 'path_length',
            'range': (1000.0, 10000.0),
            'points': 15,
            'description': 'Test scaling with interferometer arm length'
        },
        {
            'name': 'Wavelength',
            'type': 'wavelength',
            'range': (500e-9, 2000e-9),
            'points': 15,
            'description': 'Test scaling with optical wavelength'
        }
    ]
    
    all_results = {}
    
    for config in sweep_configs:
        print(f"\n{'-'*60}")
        print(f"Running {config['name']} Sweep")
        print(f"Description: {config['description']}")
        print(f"Range: {config['range'][0]:.2e} to {config['range'][1]:.2e}")
        
        # Run the sweep
        results = interferometry.parameter_sweep_analysis(
            base_setup=base_setup,
            sweep_type=config['type'],
            sweep_range=config['range'],
            n_points=config['points']
        )
        
        all_results[config['name']] = results
        
        # Print summary
        summary = results['summary']
        scaling = results['scaling_analysis']
        
        print(f"\nResults Summary:")
        print(f"  Mean relative difference: {summary['mean_relative_difference']:.6f}")
        print(f"  Std relative difference: {summary['std_relative_difference']:.6f}")
        print(f"  Range: {summary['min_relative_difference']:.6f} to {summary['max_relative_difference']:.6f}")
        print(f"  Mean distinguishability: {summary['mean_distinguishability']:.2e} σ")
        print(f"  Effect stability: {'STABLE' if summary['stable_effect'] else 'UNSTABLE'}")
        
        # Show actual phase differences to understand the scale
        phase_diffs = results['ltqg_phase_differences'] - results['qm_phase_differences']
        print(f"  Absolute phase differences: {np.min(phase_diffs):.2e} to {np.max(phase_diffs):.2e} rad")
        print(f"  LTQG phase range: {np.min(results['ltqg_phase_differences']):.6f} to {np.max(results['ltqg_phase_differences']):.6f} rad")
        print(f"  QM phase range: {np.min(results['qm_phase_differences']):.6f} to {np.max(results['qm_phase_differences']):.6f} rad")
        
        print(f"\nScaling Analysis:")
        print(f"  Power law exponent: {scaling['power_law_exponent']:.3f}")
        print(f"  Correlation coefficient: {scaling['correlation_coefficient']:.3f}")
        print(f"  Scaling type: {scaling['scaling_type']}")
        print(f"  Interpretation: {scaling['interpretation']}")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    
    stable_count = sum(1 for result in all_results.values() if result['summary']['stable_effect'])
    total_count = len(all_results)
    
    print(f"Stability Summary: {stable_count}/{total_count} parameter sweeps show stable effects")
    
    # Check for consistent physical scaling
    physical_indicators = []
    artifact_indicators = []
    
    for name, result in all_results.items():
        scaling = result['scaling_analysis']
        if scaling['scaling_type'] in ['scale_invariant_physical', 'power_law_physical']:
            physical_indicators.append(name)
        elif scaling['scaling_type'] in ['exponential_artifact', 'unstable_artifact']:
            artifact_indicators.append(name)
    
    print(f"\nPhysical Effect Indicators: {physical_indicators}")
    print(f"Artifact Indicators: {artifact_indicators}")
    
    if len(physical_indicators) >= len(artifact_indicators):
        print(f"\n🟢 CONCLUSION: The large LTQG vs QM differences appear to be GENUINE PHYSICAL EFFECTS")
        print(f"   The effects show stable, predictable scaling across multiple parameters.")
        print(f"   The distinguishability values >10¹² σ likely represent real theoretical predictions.")
    else:
        print(f"\n🔴 CONCLUSION: The large LTQG vs QM differences may be NUMERICAL ARTIFACTS")
        print(f"   The effects show unstable or pathological scaling behavior.")
        print(f"   The distinguishability values >10¹² σ may be due to scaling issues.")
    
    return all_results


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="LTQG Complete Demonstration")
    parser.add_argument("--generate-all", action="store_true",
                       help="Generate all figures and run all analyses")
    parser.add_argument("--figures-only", action="store_true",
                       help="Generate figures only")
    parser.add_argument("--experiments-only", action="store_true",
                       help="Run experimental analysis only")
    parser.add_argument("--parameter-sweep", action="store_true",
                       help="Run interferometry parameter sweep analysis only")
    parser.add_argument("--save-dir", type=str, default="figs",
                       help="Directory to save figures")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't display plots (useful for batch processing)")
    
    args = parser.parse_args()
    
    # Set matplotlib backend for batch processing
    if args.no_display:
        plt.switch_backend('Agg')
    
    print_header()
    
    # Check if only parameter sweep requested
    if args.parameter_sweep:
        sweep_results = run_interferometry_parameter_sweep()
        return
    
    # Always show core concepts
    demonstrate_core_concepts()
    create_theoretical_overview()
    demonstrate_numerical_examples()
    
    if args.figures_only or args.generate_all:
        figures = generate_paper_figures(args.save_dir)
        if not args.no_display:
            plt.show()
    
    if args.experiments_only or args.generate_all:
        experimental_results = run_experimental_analysis()
    
    if args.generate_all:
        performance_benchmark()
        create_comparison_table()
        # Also run parameter sweep for complete analysis
        sweep_results = run_interferometry_parameter_sweep()
    
    print("🎉 LTQG DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("The Log-Time Quantum Gravity framework offers a minimal,")
    print("operationally distinct route toward unified temporal structure")
    print("underlying gravity and quantum mechanics.")
    print()
    print("Next steps:")
    print("• Examine the generated figures in the '{}' directory".format(args.save_dir))
    print("• Consider implementing the proposed experimental protocols")
    print("• Explore extensions to quantum field theory and cosmology")
    print("• Investigate the mathematical foundations further")
    print()
    print("For more information, see the full LTQG paper and")
    print("the accompanying Python implementation.")
    print("=" * 60)


if __name__ == "__main__":
    main()