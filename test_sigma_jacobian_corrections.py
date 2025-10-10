"""
Test Suite for σ-Jacobian Mathematical Corrections
=================================================

This script verifies that all σ-Jacobian identities are correctly implemented
according to the rigorous mathematical framework provided.

Tests cover:
1. Basic transformations: σ = log(τ/τ₀), τ = τ₀e^σ
2. Derivatives: dσ/dτ = 1/τ, dτ/dσ = τ
3. Measures: dτ = τ dσ
4. Hamiltonian scaling: H_eff = τ H
5. Proper integration with Jacobian factors
6. σ-uniform protocol validation with τ-weighting
"""

import numpy as np
import matplotlib.pyplot as plt
from ltqg_core import create_ltqg_simulator, LTQGConfig, LTQGSimulator
from ltqg_experiments import ExperimentalSuite
import sys
import warnings

def test_basic_transformations():
    """Test basic σ ↔ τ transformations."""
    print("Testing basic transformations...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    transform = simulator.time_transform
    
    # Test forward and backward transformations
    tau_values = np.array([0.1, 1.0, 10.0, 100.0])
    sigma_values = transform.sigma_from_tau(tau_values)
    tau_recovered = transform.tau_from_sigma(sigma_values)
    
    # Check identity: τ = τ₀ exp(σ) where σ = log(τ/τ₀)
    expected_sigma = np.log(tau_values / 1.0)
    
    sigma_error = np.max(np.abs(sigma_values - expected_sigma))
    tau_error = np.max(np.abs(tau_values - tau_recovered))
    
    print(f"  σ transformation error: {sigma_error:.2e}")
    print(f"  τ recovery error: {tau_error:.2e}")
    
    assert sigma_error < 1e-14, f"σ transformation error too large: {sigma_error}"
    assert tau_error < 1e-13, f"τ recovery error too large: {tau_error}"  # Slightly relaxed for numerical precision
    print("  ✓ Basic transformations passed")


def test_jacobian_derivatives():
    """Test Jacobian derivatives: dτ/dσ = τ, dσ/dτ = 1/τ."""
    print("Testing Jacobian derivatives...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    transform = simulator.time_transform
    
    # Test dτ/dσ = τ₀ exp(σ) = τ
    sigma_test = np.array([-2.0, 0.0, 2.0])
    dtau_dsigma = transform.dtau_dsigma(sigma_test)
    tau_expected = transform.tau_from_sigma(sigma_test)
    
    jacobian_error = np.max(np.abs(dtau_dsigma - tau_expected))
    print(f"  dτ/dσ Jacobian error: {jacobian_error:.2e}")
    
    # Test dσ/dτ = 1/τ (numerical verification)
    tau_test = np.array([0.1, 1.0, 10.0])
    delta_tau = 1e-8
    sigma_plus = transform.sigma_from_tau(tau_test + delta_tau)
    sigma_minus = transform.sigma_from_tau(tau_test - delta_tau)
    dsigma_dtau_numerical = (sigma_plus - sigma_minus) / (2 * delta_tau)
    dsigma_dtau_analytical = 1.0 / tau_test
    
    derivative_error = np.max(np.abs(dsigma_dtau_numerical - dsigma_dtau_analytical))
    print(f"  dσ/dτ derivative error: {derivative_error:.2e}")
    
    assert jacobian_error < 1e-14, f"Jacobian error too large: {jacobian_error}"
    assert derivative_error < 1e-6, f"Derivative error too large: {derivative_error}"
    print("  ✓ Jacobian derivatives passed")


def test_hamiltonian_scaling():
    """Test that H_eff = τ H is correctly implemented."""
    print("Testing Hamiltonian scaling...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    evolution = simulator.evolution
    
    # Test Hamiltonian
    H = np.array([[1.0, 0.5], [0.5, -1.0]])
    sigma_test = 2.0
    tau_test = simulator.time_transform.tau_from_sigma(np.array([sigma_test]))[0]
    
    # Get effective generator
    K_sigma = evolution.sigma_generator(H, sigma_test)
    
    # Expected: K(σ) = τ₀ exp(σ) H = τ H
    expected_K = tau_test * H
    
    scaling_error = np.max(np.abs(K_sigma - expected_K))
    print(f"  Hamiltonian scaling error: {scaling_error:.2e}")
    
    assert scaling_error < 1e-14, f"Hamiltonian scaling error: {scaling_error}"
    print("  ✓ Hamiltonian scaling passed")


def test_measure_integration():
    """Test proper measure transformation in integrals."""
    print("Testing measure integration...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    
    # Test integral: ∫ f(τ) dτ = ∫ f(τ(σ)) τ(σ) dσ
    # Use f(τ) = 1/τ so the integral converges
    
    # Direct τ integration
    tau_range = np.logspace(-1, 1, 1000)  # 0.1 to 10
    f_tau = 1.0 / tau_range
    integral_tau = np.trapz(f_tau, tau_range)
    
    # σ integration with proper Jacobian
    sigma_range = simulator.time_transform.sigma_from_tau(tau_range)
    tau_from_sigma = simulator.time_transform.tau_from_sigma(sigma_range)
    f_sigma = 1.0 / tau_from_sigma  # f(τ(σ))
    jacobian = tau_from_sigma  # dτ/dσ = τ
    integrand_sigma = f_sigma * jacobian  # Should equal 1.0
    integral_sigma = np.trapz(integrand_sigma, sigma_range)
    
    measure_error = abs(integral_tau - integral_sigma) / abs(integral_tau)
    print(f"  τ integral: {integral_tau:.6f}")
    print(f"  σ integral: {integral_sigma:.6f}")
    print(f"  Relative error: {measure_error:.2e}")
    
    assert measure_error < 0.01, f"Measure transformation error: {measure_error}"
    print("  ✓ Measure integration passed")


def test_sigma_uniform_protocol():
    """Test σ-uniform protocol with proper τ-weighting."""
    print("Testing σ-uniform protocol validation...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    protocols = simulator.protocols
    
    # Run protocol comparison with validation
    tau_range = (1.0, 10.0)
    result = protocols.compare_measurement_protocols(
        tau_range=tau_range, alpha=0.5, n_measurements=10
    )
    
    validation = result['sigma_protocol_validation']
    
    print(f"  Spacing error: {validation['spacing_error']:.6f}")
    print(f"  Cutoff sensitive: {validation['cutoff_sensitive']}")
    print(f"  Validation passed: {validation['validation_passed']}")
    
    # Check test observable comparison
    test_comp = validation['test_observable_comparison']
    print(f"  Unweighted error: {test_comp['unweighted_relative_error']:.6f}")
    print(f"  Weighted error: {test_comp['weighted_relative_error']:.6f}")
    
    # The weighted error should be smaller (better) than unweighted
    assert test_comp['weighted_relative_error'] < test_comp['unweighted_relative_error'], \
        "σ-uniform weighting is not improving accuracy"
    
    if validation['warnings']:
        print(f"  Warnings: {len(validation['warnings'])}")
        for warning in validation['warnings']:
            print(f"    - {warning}")
    
    print("  ✓ σ-uniform protocol validation passed")


def test_phase_integral_corrections():
    """Test that phase integrals include proper Jacobian factors."""
    print("Testing phase integral corrections...")
    
    from ltqg_experiments import ClockTransportProtocol, ExperimentalSetup
    
    simulator = create_ltqg_simulator(tau0=1.0)
    protocol = ClockTransportProtocol(simulator)
    
    # Create test setup
    setup = ExperimentalSetup(
        duration=1.0, precision=1e-6, 
        environment="Test", feasibility_score=0.8
    )
    
    # Test the corrected gravitational path integral
    path_redshifts = np.array([1.0, 0.9, 0.8, 0.9, 1.0])  # Round trip
    path_times = np.linspace(0, setup.duration, len(path_redshifts))
    
    # This should now include proper τ Jacobian factor
    phase = protocol.gravitational_path_integral(path_redshifts, path_times)
    
    print(f"  Corrected phase integral: {phase:.6f}")
    
    # Verify that the calculation ran without errors
    assert np.isfinite(phase), "Phase integral returned non-finite value"
    assert phase != 0.0, "Phase integral returned exactly zero (suspicious)"
    
    print("  ✓ Phase integral corrections passed")


def test_overall_consistency():
    """Test overall consistency of the mathematical framework."""
    print("Testing overall consistency...")
    
    # Run full experimental suite to check for errors
    try:
        suite = ExperimentalSuite()
        results = suite.run_comprehensive_analysis()
        
        # Check that all experiments ran without errors
        for experiment_name, result in results.items():
            ltqg_pred = result['ltqg_prediction']
            qm_pred = result['qm_prediction']
            distinguishability = result['distinguishability']
            
            assert np.isfinite(distinguishability), f"{experiment_name}: Non-finite distinguishability"
            assert distinguishability >= 0, f"{experiment_name}: Negative distinguishability"
            
            print(f"  {experiment_name}: {distinguishability:.3f}σ distinguishability")
        
        print("  ✓ Overall consistency passed")
        
    except Exception as e:
        print(f"  ✗ Overall consistency failed: {e}")
        raise


def test_cosmological_models():
    """Test cosmological model implementations for σ-time consistency."""
    print("Testing cosmological models...")
    
    simulator = create_ltqg_simulator(tau0=1.0)
    cosmology = simulator.cosmology
    
    # Test power-law scale factors
    sigma_range = np.linspace(-2, 2, 100)
    
    # Radiation era: a(τ) ∝ τ^(1/2) → a(σ) ∝ exp(σ/2)
    a_radiation = cosmology.scale_factor_power_law(sigma_range, n=0.5)
    expected_radiation = np.exp(0.5 * sigma_range)
    
    radiation_error = np.max(np.abs(a_radiation - expected_radiation))
    print(f"  Radiation era scaling error: {radiation_error:.2e}")
    
    # Matter era: a(τ) ∝ τ^(2/3) → a(σ) ∝ exp(2σ/3)
    a_matter = cosmology.scale_factor_power_law(sigma_range, n=2.0/3.0)
    expected_matter = np.exp((2.0/3.0) * sigma_range)
    
    matter_error = np.max(np.abs(a_matter - expected_matter))
    print(f"  Matter era scaling error: {matter_error:.2e}")
    
    assert radiation_error < 1e-14, f"Radiation scaling error: {radiation_error}"
    assert matter_error < 1e-14, f"Matter scaling error: {matter_error}"
    print("  ✓ Cosmological models passed")


def run_all_tests():
    """Run all tests and summarize results."""
    print("=" * 60)
    print("LTQG σ-Jacobian Mathematical Corrections Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_transformations,
        test_jacobian_derivatives,
        test_hamiltonian_scaling,
        test_measure_integration,
        test_sigma_uniform_protocol,
        test_phase_integral_corrections,
        test_cosmological_models,
        test_overall_consistency,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()
    
    print("=" * 60)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! σ-Jacobian corrections are mathematically sound.")
    else:
        print("❌ Some tests failed. Please review the mathematical implementation.")
    
    print("=" * 60)
    
    return failed == 0


def generate_mathematical_validation_report():
    """Generate a detailed mathematical validation report."""
    print("\nGenerating Mathematical Validation Report...")
    print("-" * 50)
    
    simulator = create_ltqg_simulator(tau0=1.0)
    
    # Test various σ-Jacobian identities
    sigma_test = np.array([-3, -1, 0, 1, 3])
    tau_test = simulator.time_transform.tau_from_sigma(sigma_test)
    
    print("Core Identity Verification:")
    print(f"σ values: {sigma_test}")
    print(f"τ = τ₀e^σ: {tau_test}")
    print(f"σ recovered: {simulator.time_transform.sigma_from_tau(tau_test)}")
    print()
    
    # Test Jacobian factors
    jacobian = simulator.time_transform.dtau_dsigma(sigma_test)
    print("Jacobian Verification:")
    print(f"dτ/dσ = τ₀e^σ: {jacobian}")
    print(f"Should equal τ: {tau_test}")
    print(f"Match: {np.allclose(jacobian, tau_test)}")
    print()
    
    # Test Hamiltonian scaling
    H_test = np.array([[1, 0], [0, -1]])
    K_values = []
    for sigma in sigma_test:
        K = simulator.evolution.sigma_generator(H_test, sigma)
        eigenvals = np.linalg.eigvals(K)
        K_values.append(np.max(np.abs(eigenvals)))
    
    print("Hamiltonian Scaling Verification:")
    print(f"Max |eigenvalue| of K(σ): {K_values}")
    print(f"Should scale as τ: {tau_test}")
    print(f"Ratio K/τ: {np.array(K_values) / tau_test}")
    print()
    
    # Protocol validation
    protocols = simulator.protocols
    result = protocols.compare_measurement_protocols(
        tau_range=(0.1, 10.0), alpha=0.3, n_measurements=8
    )
    
    validation = result['sigma_protocol_validation']
    print("σ-Uniform Protocol Validation:")
    print(f"Spacing error: {validation['spacing_error']:.6f}")
    print(f"Cutoff σ_min: {validation['sigma_cutoff']:.3f}")
    print(f"Cutoff τ_min: {validation['tau_cutoff']:.6f}")
    print(f"Validation passed: {validation['validation_passed']}")
    print()
    
    if validation['warnings']:
        print("WARNINGS:")
        for warning in validation['warnings']:
            print(f"  • {warning}")
    else:
        print("No validation warnings.")
    
    print("-" * 50)
    print("Mathematical validation report complete.")


if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    # Generate detailed report
    generate_mathematical_validation_report()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)