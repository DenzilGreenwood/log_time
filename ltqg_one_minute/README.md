# LTQG in One Minute: Logarithmic Time Quantum Gravity

## Overview

This folder contains a comprehensive implementation of Logarithmic Time Quantum Gravity (LTQG) concepts, addressing Kiefer's fundamental problems in quantum gravity through the σ-time reparameterization framework.

## Core Concept

**Clock Change**: Use σ ≡ log(τ/τ₀) with τ = proper time.

**Dynamics**: Standard unitary physics preserved but written as:
```
iℏ ∂ψ/∂σ = H_eff(σ) ψ
```
where H_eff(σ) = τ H(τ) because dσ/dτ = 1/τ.

**Asymptotic Silence**: As τ→0⁺ (early-time/near-horizon), τ→0 ⇒ H_eff→0, so the effective generator shuts off without changing underlying physics.

**Operational Bonus**: Multiplicative time dilations become additive σ-shifts, making clock comparisons and experimental scheduling across gravitational potentials algebraically clean.

## Folder Structure

### `/core_concepts/`
- `ltqg_core.py` - Core LTQG mathematical framework
- `sigma_transformation.py` - σ-time transformation utilities
- `effective_hamiltonian.py` - H_eff(σ) implementations
- `asymptotic_silence.py` - Near-horizon and early-time behavior

### `/kiefer_problems/`
- `problem_of_time.py` - Wheeler-DeWitt timelessness solutions
- `quantization_necessity.py` - Why quantize gravity analysis
- `singularity_resolution.py` - Big bang and black hole singularities
- `black_hole_thermodynamics.py` - Information paradox approaches
- `covariant_canonical.py` - Background independence
- `quantum_cosmology.py` - Boundary conditions and arrow of time
- `asymptotic_safety.py` - Running dimensions and scale dependence
- `decoherence_interpretation.py` - Emergence of classicality

### `/worked_examples/`
- `flrw_cosmology.py` - FLRW with radiation/matter/Λ
- `schwarzschild_horizon.py` - Near-horizon geodesics and redshift
- `wdw_minisuperspace.py` - Wheeler-DeWitt toy models
- `sigma_wdw_rewrite.py` - σ-parametrized WDW equation

### `/visualizations/`
- `sigma_time_plots.py` - Interactive σ vs τ visualizations
- `webgl_sigma_demo.html` - WebGL demonstration with σ-uniform ticks
- `phase_accumulation.py` - Phase differences as Δσ
- `curvature_scalars.py` - Smooth early-σ asymptotics

### `/operational_tests/`
- `two_clock_experiment.py` - σ-uniform scheduling protocols
- `ramsey_interferometer.py` - Atom interferometer with σ-scheduling
- `phase_prediction.py` - Linear phase differences in Δσ
- `gravitational_redshift.py` - Redshift as σ-offset isolation

## Quick Start

1. **Mathematical Foundation**: Start with `core_concepts/ltqg_core.py`
2. **Kiefer Problems**: Explore specific problems in `kiefer_problems/`
3. **Examples**: Run worked examples in `worked_examples/`
4. **Visualizations**: View interactive demos in `visualizations/`
5. **Tests**: Implement operational protocols in `operational_tests/`

## Key Technical Results

### Three Mathematical Lemmas
1. **Continuity**: If H(τ) is τ-regular, then H_eff(σ) is strongly continuous and vanishes as σ→−∞
2. **WKB Matching**: σ-parametrized solutions match WKB limit in minisuperspace
3. **Entropy Growth**: Under generic local couplings, entanglement entropy is non-decreasing in σ

### Operational Predictions
- Phase differences linear in Δσ
- Gravitational contributions as clean offsets
- σ-uniform experimental scheduling
- Multiplicative dilations → additive shifts

## What LTQG Claims (and Doesn't Claim)

### Claims
- Unifying clock choice for thorny problems
- Transparent handling of time, initial data, regularization
- Calculable experimental scheduling across potentials
- Clean semiclassical limit bookkeeping

### Doesn't Claim
- New interaction or replacement for LQG/string theory
- Solution to microstate counting problem
- Proof of information recovery
- Complete theory of quantum gravity

## Usage

Each module can be run independently:

```python
from core_concepts.ltqg_core import LTQGFramework
from kiefer_problems.problem_of_time import ProblemOfTime
from worked_examples.flrw_cosmology import FLRWExample

# Initialize LTQG framework
ltqg = LTQGFramework()

# Address specific Kiefer problem
time_problem = ProblemOfTime(ltqg)
time_problem.solve_wdw_constraint()

# Run cosmology example
flrw = FLRWExample(ltqg)
flrw.plot_curvature_vs_sigma()
```

## Research Integration

This implementation integrates with the broader LTQG research program:
- Complements existing WebGL visualizations
- Provides computational backend for theoretical analysis
- Enables quantitative testing of LTQG predictions
- Facilitates comparison with alternative approaches

## License

MIT License - See main repository LICENSE file.