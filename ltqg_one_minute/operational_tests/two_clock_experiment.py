"""
Two-Clock Experiment in σ-Time: Operational Tests

This module implements operational tests for LTQG using two-clock experiments
scheduled uniformly in σ-time. These experiments test the key prediction that
phase differences are linear in Δσ, isolating gravitational contributions as offsets.

Key concepts:
- σ-uniform experimental scheduling
- Gravitational redshift as σ-offset
- Clock synchronization across potentials
- Phase accumulation measurements
- Comparison with traditional approaches
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
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
class ClockConfiguration:
    """Configuration for a single clock in the experiment."""
    position: np.ndarray  # Spatial position [x, y, z]
    gravitational_potential: float  # Local gravitational potential φ
    initial_phase: float  # Initial quantum phase
    oscillator_frequency: float  # Local oscillator frequency
    clock_id: str  # Identifier for the clock


@dataclass
class TwoClockExperimentResults:
    """Results from two-clock experiment in σ-time."""
    sigma_array: np.ndarray
    tau_array: np.ndarray
    clock_phases: Dict[str, np.ndarray]
    phase_differences: np.ndarray
    gravitational_offsets: np.ndarray
    synchronization_errors: np.ndarray
    ltqg_predictions: Dict[str, np.ndarray]
    traditional_predictions: Dict[str, np.ndarray]


class TwoClockExperiment:
    """
    Implementation of two-clock experiments using LTQG σ-time scheduling.
    """
    
    def __init__(self, ltqg_framework: Optional[LTQGFramework] = None):
        """
        Initialize two-clock experiment framework.
        
        Args:
            ltqg_framework: LTQG framework instance
        """
        self.ltqg = ltqg_framework or LTQGFramework()
        self.sigma_transform = SigmaTransformation(self.ltqg.tau_0)
        self.silence = AsymptoticSilence(self.ltqg.tau_0, self.ltqg.hbar)
        
        # Physical constants
        self.c = 1.0  # Speed of light (natural units)
        self.G = 1.0  # Newton's constant
        
    def setup_clock_pair(self, clock1_config: ClockConfiguration, 
                         clock2_config: ClockConfiguration) -> Dict[str, ClockConfiguration]:
        """
        Set up a pair of clocks for the experiment.
        
        Args:
            clock1_config: Configuration for first clock
            clock2_config: Configuration for second clock
            
        Returns:
            Dictionary of clock configurations
        """
        return {
            clock1_config.clock_id: clock1_config,
            clock2_config.clock_id: clock2_config
        }
    
    def compute_gravitational_sigma_offset(self, phi1: float, phi2: float) -> float:
        """
        Compute σ-offset due to gravitational potential difference.
        
        In weak field: Δσ ≈ (φ₂ - φ₁)/c²
        
        Args:
            phi1: Gravitational potential at clock 1
            phi2: Gravitational potential at clock 2
            
        Returns:
            σ-offset due to gravitational redshift
        """
        return (phi2 - phi1) / (self.c * self.c)
    
    def clock_evolution_sigma(self, sigma: float, phase_components: np.ndarray,
                            clock_configs: Dict[str, ClockConfiguration]) -> np.ndarray:
        """
        Evolution equation for clock phases in σ-time.
        
        For each clock: iℏ ∂φ/∂σ = H_eff_clock(σ) φ
        where H_eff_clock includes local gravitational effects.
        
        Args:
            sigma: σ-time parameter
            phase_components: Real and imaginary parts of clock phases
            clock_configs: Clock configurations
            
        Returns:
            Time derivatives of phase components
        """
        n_clocks = len(clock_configs)
        n_complex = len(phase_components) // 2
        
        # Extract complex phases
        phases_real = phase_components[:n_complex]
        phases_imag = phase_components[n_complex:]
        phases = phases_real + 1j * phases_imag
        
        # Compute proper time
        tau = self.ltqg.tau_from_sigma(sigma)
        
        # Evolution for each clock
        dphases_dsigma = np.zeros_like(phases, dtype=complex)
        
        clock_list = list(clock_configs.values())
        for i, clock in enumerate(clock_list):
            # Local effective Hamiltonian
            # H_eff = τ × (local_frequency + gravitational_corrections)
            
            # Gravitational redshift factor
            redshift_factor = np.sqrt(1 + 2 * clock.gravitational_potential / (self.c * self.c))
            
            # Local frequency in σ-time
            local_freq_eff = tau * clock.oscillator_frequency * redshift_factor
            
            # Apply asymptotic silence
            silence_factor = self.silence.silence_envelope(sigma)
            local_freq_eff *= silence_factor
            
            # Phase evolution: iℏ ∂φ/∂σ = H_eff φ
            dphases_dsigma[i] = -1j * local_freq_eff * phases[i] / self.ltqg.hbar
        
        # Convert back to real/imaginary representation
        return np.concatenate([dphases_dsigma.real, dphases_dsigma.imag])
    
    def run_sigma_scheduled_experiment(self, clock_configs: Dict[str, ClockConfiguration],
                                     sigma_range: Tuple[float, float],
                                     num_measurements: int = 1000) -> TwoClockExperimentResults:
        """
        Run two-clock experiment with σ-uniform scheduling.
        
        Key prediction: Phase differences linear in Δσ with gravitational offsets.
        
        Args:
            clock_configs: Clock configurations
            sigma_range: Range of σ-time for experiment
            num_measurements: Number of measurement points
            
        Returns:
            Complete experiment results
        """
        sigma_array = np.linspace(*sigma_range, num_measurements)
        tau_array = self.ltqg.tau_from_sigma(sigma_array)
        
        # Initial conditions (complex phases for all clocks)
        clock_list = list(clock_configs.values())
        n_clocks = len(clock_list)
        
        initial_phases = np.array([np.exp(1j * clock.initial_phase) for clock in clock_list], 
                                dtype=complex)
        initial_real_imag = np.concatenate([initial_phases.real, initial_phases.imag])
        
        # Solve phase evolution
        def evolution_rhs(sigma, phase_components):
            return self.clock_evolution_sigma(sigma, phase_components, clock_configs)
        
        # Integrate using solve_ivp for better control
        sol = solve_ivp(evolution_rhs, sigma_range, initial_real_imag, 
                       t_eval=sigma_array, dense_output=True, rtol=1e-8)
        
        if not sol.success:
            raise RuntimeError(f"Clock evolution failed: {sol.message}")
        
        # Extract clock phases
        n_complex = len(initial_phases)
        phases_real = sol.y[:n_complex, :]
        phases_imag = sol.y[n_complex:, :]
        phases_complex = phases_real + 1j * phases_imag
        
        # Store phases by clock ID
        clock_phases = {}
        for i, clock in enumerate(clock_list):
            clock_phases[clock.clock_id] = phases_complex[i, :]
        
        # Compute phase differences
        if len(clock_list) >= 2:
            phase_diff = np.angle(clock_phases[clock_list[1].clock_id] / 
                                clock_phases[clock_list[0].clock_id])
            
            # Unwrap phase to handle 2π jumps
            phase_diff = np.unwrap(phase_diff)
        else:
            phase_diff = np.zeros_like(sigma_array)
        
        # Gravitational offset prediction
        if len(clock_list) >= 2:
            sigma_offset = self.compute_gravitational_sigma_offset(
                clock_list[0].gravitational_potential,
                clock_list[1].gravitational_potential
            )
            gravitational_offsets = sigma_offset * np.ones_like(sigma_array)
        else:
            gravitational_offsets = np.zeros_like(sigma_array)
        
        # LTQG predictions
        ltqg_predictions = self.compute_ltqg_predictions(sigma_array, clock_configs)
        
        # Traditional predictions for comparison
        traditional_predictions = self.compute_traditional_predictions(sigma_array, tau_array, clock_configs)
        
        # Synchronization errors (measure of σ-scheduling accuracy)
        sync_errors = self.compute_synchronization_errors(sigma_array, clock_phases, clock_configs)
        
        return TwoClockExperimentResults(
            sigma_array=sigma_array,
            tau_array=tau_array,
            clock_phases=clock_phases,
            phase_differences=phase_diff,
            gravitational_offsets=gravitational_offsets,
            synchronization_errors=sync_errors,
            ltqg_predictions=ltqg_predictions,
            traditional_predictions=traditional_predictions
        )
    
    def compute_ltqg_predictions(self, sigma_array: np.ndarray, 
                               clock_configs: Dict[str, ClockConfiguration]) -> Dict[str, np.ndarray]:
        """
        Compute LTQG theoretical predictions for the experiment.
        
        Key predictions:
        1. Phase differences linear in Δσ
        2. Gravitational effects as additive σ-offsets
        3. Clock rates proportional to τ = exp(σ)
        
        Args:
            sigma_array: Array of σ-time values
            clock_configs: Clock configurations
            
        Returns:
            Dictionary of LTQG predictions
        """
        clock_list = list(clock_configs.values())
        
        # Linear phase accumulation in σ
        if len(clock_list) >= 2:
            freq_diff = clock_list[1].oscillator_frequency - clock_list[0].oscillator_frequency
            potential_diff = clock_list[1].gravitational_potential - clock_list[0].gravitational_potential
            
            # LTQG prediction: Δφ = ∫ Δω_eff(σ) dσ
            # where Δω_eff includes both frequency and gravitational contributions
            
            tau_array = self.ltqg.tau_from_sigma(sigma_array)
            
            # Frequency difference contribution (scales with τ)
            freq_contribution = np.zeros_like(sigma_array)
            for i, sigma in enumerate(sigma_array):
                tau = tau_array[i]
                silence_factor = self.silence.silence_envelope(sigma)
                freq_contribution[i] = freq_diff * tau * silence_factor * (sigma - sigma_array[0])
            
            # Gravitational contribution (additive offset)
            gravitational_contribution = potential_diff / (self.c * self.c) * np.ones_like(sigma_array)
            
            # Total LTQG prediction
            total_phase_diff = freq_contribution + gravitational_contribution
            
        else:
            total_phase_diff = np.zeros_like(sigma_array)
            freq_contribution = np.zeros_like(sigma_array)
            gravitational_contribution = np.zeros_like(sigma_array)
        
        return {
            'total_phase_difference': total_phase_diff,
            'frequency_contribution': freq_contribution,
            'gravitational_contribution': gravitational_contribution,
            'linearity_measure': self.compute_linearity_measure(sigma_array, total_phase_diff)
        }
    
    def compute_traditional_predictions(self, sigma_array: np.ndarray, tau_array: np.ndarray,
                                      clock_configs: Dict[str, ClockConfiguration]) -> Dict[str, np.ndarray]:
        """
        Compute traditional (non-LTQG) predictions for comparison.
        
        Traditional approach: schedule experiments uniformly in coordinate time,
        then account for gravitational redshift post-hoc.
        
        Args:
            sigma_array: Array of σ-time values
            tau_array: Array of proper time values
            clock_configs: Clock configurations
            
        Returns:
            Dictionary of traditional predictions
        """
        clock_list = list(clock_configs.values())
        
        if len(clock_list) >= 2:
            # Traditional approach: integrate in proper time
            traditional_phase_diff = np.zeros_like(sigma_array)
            
            for i, tau in enumerate(tau_array):
                # Redshift factors
                redshift1 = np.sqrt(1 + 2 * clock_list[0].gravitational_potential / (self.c * self.c))
                redshift2 = np.sqrt(1 + 2 * clock_list[1].gravitational_potential / (self.c * self.c))
                
                # Accumulated phase difference
                if i > 0:
                    dt = tau_array[i] - tau_array[i-1]
                    dphase = (clock_list[1].oscillator_frequency * redshift2 - 
                             clock_list[0].oscillator_frequency * redshift1) * dt
                    traditional_phase_diff[i] = traditional_phase_diff[i-1] + dphase
            
            # Multiplicative redshift effects (traditional treatment)
            redshift_ratio = np.sqrt((1 + 2 * clock_list[1].gravitational_potential / (self.c * self.c)) / 
                                   (1 + 2 * clock_list[0].gravitational_potential / (self.c * self.c)))
            
        else:
            traditional_phase_diff = np.zeros_like(sigma_array)
            redshift_ratio = 1.0
        
        return {
            'phase_difference_traditional': traditional_phase_diff,
            'redshift_ratio': redshift_ratio * np.ones_like(sigma_array),
            'coordinate_time_array': tau_array
        }
    
    def compute_synchronization_errors(self, sigma_array: np.ndarray, 
                                     clock_phases: Dict[str, np.ndarray],
                                     clock_configs: Dict[str, ClockConfiguration]) -> np.ndarray:
        """
        Compute synchronization errors when using σ-uniform scheduling.
        
        Measure how well clocks stay synchronized when scheduled uniformly in σ
        compared to traditional coordinate time scheduling.
        
        Args:
            sigma_array: Array of σ-time values
            clock_phases: Evolved clock phases
            clock_configs: Clock configurations
            
        Returns:
            Array of synchronization error measures
        """
        if len(clock_phases) < 2:
            return np.zeros_like(sigma_array)
        
        # Compare actual phase evolution with ideal synchronization
        clock_ids = list(clock_phases.keys())
        
        # Ideal synchronization: phases should evolve predictably in σ
        sync_errors = np.zeros_like(sigma_array)
        
        for i, sigma in enumerate(sigma_array):
            # Expected phase relationship based on LTQG
            tau = self.ltqg.tau_from_sigma(sigma)
            
            # Compute expected phase difference
            expected_phase_diff = 0.0
            for clock_id in clock_ids:
                clock = clock_configs[clock_id]
                silence_factor = self.silence.silence_envelope(sigma)
                expected_phase_diff += clock.oscillator_frequency * tau * silence_factor * sigma
            
            # Actual phase difference
            actual_phases = [clock_phases[clock_id][i] for clock_id in clock_ids]
            actual_phase_diff = np.sum([np.angle(phase) for phase in actual_phases])
            
            # Synchronization error
            sync_errors[i] = abs(actual_phase_diff - expected_phase_diff)
        
        return sync_errors
    
    def compute_linearity_measure(self, sigma_array: np.ndarray, phase_diff: np.ndarray) -> float:
        """
        Compute measure of linearity in σ for phase differences.
        
        LTQG predicts phase differences should be linear in Δσ.
        
        Args:
            sigma_array: Array of σ-time values
            phase_diff: Array of phase differences
            
        Returns:
            Linearity measure (R² coefficient)
        """
        if len(sigma_array) < 3:
            return 0.0
        
        # Linear fit: phase_diff = a * sigma + b
        A = np.vstack([sigma_array, np.ones(len(sigma_array))]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(A, phase_diff, rcond=None)
        
        # R-squared calculation
        if len(residuals) > 0:
            ss_res = residuals[0]
            ss_tot = np.sum((phase_diff - np.mean(phase_diff))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            # Perfect fit or degenerate case
            r_squared = 1.0 if np.allclose(phase_diff, coeffs[0] * sigma_array + coeffs[1]) else 0.0
        
        return r_squared
    
    def analyze_experimental_precision(self, results: TwoClockExperimentResults,
                                     noise_level: float = 1e-6) -> Dict:
        """
        Analyze experimental precision and error sources.
        
        Args:
            results: Experiment results
            noise_level: Simulated measurement noise level
            
        Returns:
            Precision analysis results
        """
        # Add simulated measurement noise
        noisy_phase_diff = results.phase_differences + noise_level * np.random.randn(len(results.sigma_array))
        
        # Compare LTQG vs traditional predictions
        ltqg_error = np.mean(np.abs(noisy_phase_diff - results.ltqg_predictions['total_phase_difference']))
        traditional_error = np.mean(np.abs(noisy_phase_diff - results.traditional_predictions['phase_difference_traditional']))
        
        # Linearity analysis
        linearity_ltqg = self.compute_linearity_measure(results.sigma_array, 
                                                       results.ltqg_predictions['total_phase_difference'])
        linearity_measured = self.compute_linearity_measure(results.sigma_array, noisy_phase_diff)
        
        # Gravitational offset detection
        measured_offset = np.mean(noisy_phase_diff) - np.mean(results.ltqg_predictions['frequency_contribution'])
        predicted_offset = np.mean(results.gravitational_offsets)
        offset_accuracy = abs(measured_offset - predicted_offset) / abs(predicted_offset) if predicted_offset != 0 else 0
        
        return {
            'ltqg_prediction_error': ltqg_error,
            'traditional_prediction_error': traditional_error,
            'improvement_ratio': traditional_error / ltqg_error if ltqg_error > 0 else np.inf,
            'linearity_ltqg': linearity_ltqg,
            'linearity_measured': linearity_measured,
            'gravitational_offset_accuracy': offset_accuracy,
            'synchronization_quality': np.mean(results.synchronization_errors),
            'noise_level': noise_level
        }


def demo_two_clock_experiment():
    """
    Demonstrate two-clock experiment with σ-time scheduling.
    """
    print("=== Two-Clock Experiment in σ-Time Demo ===\n")
    
    # Initialize framework
    ltqg = LTQGFramework(tau_0=1.0, hbar=1.0)
    experiment = TwoClockExperiment(ltqg)
    
    # Demo 1: Setup clock configurations
    print("1. Clock Configuration Setup:")
    
    # Clock 1: At ground level
    clock1 = ClockConfiguration(
        position=np.array([0, 0, 0]),
        gravitational_potential=0.0,  # Reference potential
        initial_phase=0.0,
        oscillator_frequency=1.0,  # Reference frequency
        clock_id="ground_clock"
    )
    
    # Clock 2: At elevated position (weaker gravitational field)
    clock2 = ClockConfiguration(
        position=np.array([0, 0, 100]),  # 100 units higher
        gravitational_potential=0.001,  # Slightly higher potential
        initial_phase=0.0,
        oscillator_frequency=1.0,
        clock_id="elevated_clock"
    )
    
    clock_configs = experiment.setup_clock_pair(clock1, clock2)
    
    for clock_id, clock in clock_configs.items():
        print(f"   {clock_id}:")
        print(f"     Position: {clock.position}")
        print(f"     Potential: {clock.gravitational_potential:.6f}")
        print(f"     Frequency: {clock.oscillator_frequency:.3f}")
    
    # Demo 2: Gravitational σ-offset calculation
    print("\n2. Gravitational σ-Offset:")
    
    sigma_offset = experiment.compute_gravitational_sigma_offset(
        clock1.gravitational_potential, clock2.gravitational_potential)
    
    print(f"   Potential difference: {clock2.gravitational_potential - clock1.gravitational_potential:.6f}")
    print(f"   σ-offset: {sigma_offset:.6f}")
    print(f"   Traditional redshift factor: {np.sqrt(1 + 2*clock2.gravitational_potential):.6f}")
    
    # Demo 3: Run σ-scheduled experiment
    print("\n3. σ-Uniform Experiment Execution:")
    
    sigma_range = (-2.0, 2.0)
    num_measurements = 500
    
    try:
        results = experiment.run_sigma_scheduled_experiment(
            clock_configs, sigma_range, num_measurements)
        
        print(f"   σ-range: [{sigma_range[0]}, {sigma_range[1]}]")
        print(f"   Measurements: {num_measurements}")
        print(f"   Evolution successful: {results.sigma_array is not None}")
        
        # Analyze phase differences
        final_phase_diff = results.phase_differences[-1]
        initial_phase_diff = results.phase_differences[0]
        total_phase_change = final_phase_diff - initial_phase_diff
        
        print(f"   Total phase difference change: {total_phase_change:.6f} rad")
        print(f"   Average synchronization error: {np.mean(results.synchronization_errors):.2e}")
        
    except Exception as e:
        print(f"   Experiment failed: {e}")
        return
    
    # Demo 4: LTQG vs Traditional predictions
    print("\n4. LTQG vs Traditional Predictions:")
    
    ltqg_pred = results.ltqg_predictions
    trad_pred = results.traditional_predictions
    
    # Linearity analysis
    linearity_ltqg = ltqg_pred['linearity_measure']
    linearity_traditional = experiment.compute_linearity_measure(
        results.sigma_array, trad_pred['phase_difference_traditional'])
    
    print(f"   LTQG linearity (R²): {linearity_ltqg:.6f}")
    print(f"   Traditional linearity (R²): {linearity_traditional:.6f}")
    
    # Final predictions comparison
    ltqg_final = ltqg_pred['total_phase_difference'][-1]
    trad_final = trad_pred['phase_difference_traditional'][-1]
    measured_final = results.phase_differences[-1]
    
    print(f"   Final phase difference:")
    print(f"     Measured: {measured_final:.6f}")
    print(f"     LTQG predicted: {ltqg_final:.6f}")
    print(f"     Traditional predicted: {trad_final:.6f}")
    
    ltqg_error = abs(measured_final - ltqg_final)
    trad_error = abs(measured_final - trad_final)
    
    print(f"   Prediction errors:")
    print(f"     LTQG error: {ltqg_error:.2e}")
    print(f"     Traditional error: {trad_error:.2e}")
    print(f"     Improvement ratio: {trad_error/ltqg_error:.2f}" if ltqg_error > 0 else "∞")
    
    # Demo 5: Precision analysis
    print("\n5. Experimental Precision Analysis:")
    
    noise_levels = [1e-8, 1e-6, 1e-4]
    
    for noise in noise_levels:
        precision = experiment.analyze_experimental_precision(results, noise)
        
        print(f"   Noise level {noise:.0e}:")
        print(f"     LTQG accuracy: {precision['ltqg_prediction_error']:.2e}")
        print(f"     Traditional accuracy: {precision['traditional_prediction_error']:.2e}")
        print(f"     Improvement ratio: {precision['improvement_ratio']:.2f}")
        print(f"     Gravitational offset accuracy: {precision['gravitational_offset_accuracy']:.2%}")


def plot_two_clock_experiment_analysis():
    """
    Plot comprehensive two-clock experiment analysis.
    """
    ltqg = LTQGFramework()
    experiment = TwoClockExperiment(ltqg)
    
    # Setup clock configurations
    clock1 = ClockConfiguration(
        position=np.array([0, 0, 0]),
        gravitational_potential=0.0,
        initial_phase=0.0,
        oscillator_frequency=1.0,
        clock_id="ground_clock"
    )
    
    clock2 = ClockConfiguration(
        position=np.array([0, 0, 100]),
        gravitational_potential=0.005,  # Stronger effect for visualization
        initial_phase=0.0,
        oscillator_frequency=1.01,  # Slight frequency difference
        clock_id="elevated_clock"
    )
    
    clock_configs = experiment.setup_clock_pair(clock1, clock2)
    
    # Run experiment
    results = experiment.run_sigma_scheduled_experiment(clock_configs, (-3, 3), 1000)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    sigma_array = results.sigma_array
    tau_array = results.tau_array
    
    # 1. Clock phases evolution
    for clock_id, phases in results.clock_phases.items():
        axes[0, 0].plot(sigma_array, np.angle(phases), linewidth=2, label=f'{clock_id} phase')
    axes[0, 0].set_xlabel('σ-Time')
    axes[0, 0].set_ylabel('Clock Phase (rad)')
    axes[0, 0].set_title('Individual Clock Phases')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Phase differences
    axes[0, 1].plot(sigma_array, results.phase_differences, 'b-', linewidth=2, label='Measured')
    axes[0, 1].plot(sigma_array, results.ltqg_predictions['total_phase_difference'], 
                   'r--', linewidth=2, label='LTQG Prediction')
    axes[0, 1].plot(sigma_array, results.traditional_predictions['phase_difference_traditional'], 
                   'g:', linewidth=2, label='Traditional')
    axes[0, 1].set_xlabel('σ-Time')
    axes[0, 1].set_ylabel('Phase Difference (rad)')
    axes[0, 1].set_title('Phase Difference Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Linearity analysis
    # Linear fit for LTQG prediction
    A = np.vstack([sigma_array, np.ones(len(sigma_array))]).T
    ltqg_coeffs, _, _, _ = np.linalg.lstsq(A, results.ltqg_predictions['total_phase_difference'], rcond=None)
    ltqg_fit = ltqg_coeffs[0] * sigma_array + ltqg_coeffs[1]
    
    axes[0, 2].plot(sigma_array, results.phase_differences, 'b-', linewidth=2, label='Measured')
    axes[0, 2].plot(sigma_array, ltqg_fit, 'r--', linewidth=2, label='Linear Fit')
    axes[0, 2].set_xlabel('σ-Time')
    axes[0, 2].set_ylabel('Phase Difference (rad)')
    axes[0, 2].set_title('Linearity in σ-Time')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Gravitational contributions
    axes[1, 0].plot(sigma_array, results.ltqg_predictions['frequency_contribution'], 
                   'purple', linewidth=2, label='Frequency Diff')
    axes[1, 0].plot(sigma_array, results.ltqg_predictions['gravitational_contribution'], 
                   'orange', linewidth=2, label='Gravitational Offset')
    axes[1, 0].plot(sigma_array, results.ltqg_predictions['total_phase_difference'], 
                   'black', linewidth=2, label='Total LTQG')
    axes[1, 0].set_xlabel('σ-Time')
    axes[1, 0].set_ylabel('Phase Contribution (rad)')
    axes[1, 0].set_title('LTQG Contribution Breakdown')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Synchronization errors
    axes[1, 1].semilogy(sigma_array, results.synchronization_errors, 'cyan', linewidth=2)
    axes[1, 1].set_xlabel('σ-Time')
    axes[1, 1].set_ylabel('Synchronization Error')
    axes[1, 1].set_title('σ-Scheduling Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. σ vs τ relationship
    axes[1, 2].semilogx(tau_array, sigma_array, 'brown', linewidth=2)
    axes[1, 2].set_xlabel('Proper Time τ')
    axes[1, 2].set_ylabel('σ-Time')
    axes[1, 2].set_title('σ-Time Coordinate Transformation')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add precision analysis as text
    precision = experiment.analyze_experimental_precision(results, 1e-6)
    
    textstr = f"""Precision Analysis (1e-6 noise):
LTQG Error: {precision['ltqg_prediction_error']:.2e}
Traditional Error: {precision['traditional_prediction_error']:.2e}
Improvement: {precision['improvement_ratio']:.2f}×
Linearity (LTQG): {precision['linearity_ltqg']:.4f}
Gravitational Accuracy: {precision['gravitational_offset_accuracy']:.2%}"""
    
    fig.text(0.02, 0.98, textstr, transform=fig.transFigure, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    demo_two_clock_experiment()
    
    # Uncomment to generate plots
    # fig = plot_two_clock_experiment_analysis()
    # plt.show()