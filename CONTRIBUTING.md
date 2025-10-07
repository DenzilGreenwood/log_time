# Contributing to LTQG

We welcome contributions to the Log-Time Quantum Gravity project! This document provides guidelines for contributing.

## 🚀 How to Contribute

### Reporting Bugs
- Use the GitHub issue tracker
- Include a clear description of the problem
- Provide minimal code example that reproduces the issue
- Include Python version and dependency versions

### Suggesting Enhancements
- Open an issue with enhancement label
- Clearly describe the proposed feature
- Explain why this enhancement would be useful
- Provide example usage if applicable

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 🧪 Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/log_time.git
cd log_time

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Generate documentation
cd docs
make html
```

## 📋 Code Style

- Follow PEP 8
- Use type hints where appropriate
- Add docstrings to all public functions
- Keep line length under 100 characters
- Use descriptive variable names

### Example Function
```python
def sigma_from_tau(tau: np.ndarray, tau0: float = 1.0) -> np.ndarray:
    """
    Convert proper time to log-time coordinates.
    
    Parameters
    ----------
    tau : np.ndarray
        Proper time values (must be positive)
    tau0 : float, optional
        Reference time scale (default: 1.0)
        
    Returns
    -------
    np.ndarray
        Log-time coordinates σ = log(τ/τ₀)
        
    Raises
    ------
    ValueError
        If tau contains non-positive values
    """
    if np.any(tau <= 0):
        raise ValueError("Proper time τ must be positive")
    return np.log(tau / tau0)
```

## 🧮 Mathematical Contributions

When contributing mathematical formulations:

1. **Provide clear derivations** in docstrings or documentation
2. **Include references** to relevant papers or textbooks  
3. **Add unit tests** with known analytical results
4. **Document assumptions** and domain of validity
5. **Use consistent notation** with existing codebase

### Mathematical Documentation Format
```python
def effective_hamiltonian(sigma: float, H: np.ndarray, tau0: float) -> np.ndarray:
    """
    Compute the effective Hamiltonian K(σ) in log-time coordinates.
    
    Mathematical Background
    -----------------------
    In LTQG, the effective Hamiltonian is given by:
    
        K(σ) = τ₀ exp(σ) H = τ H
        
    where τ = τ₀ exp(σ) is the proper time. This ensures that
    the modified Schrödinger equation:
    
        iℏ ∂|ψ⟩/∂σ = K(σ)|ψ⟩
        
    is equivalent to the standard equation in proper time.
    
    Parameters
    ----------
    sigma : float
        Log-time coordinate
    H : np.ndarray
        Standard Hamiltonian matrix
    tau0 : float
        Reference time scale
        
    Returns
    -------
    np.ndarray
        Effective Hamiltonian K(σ)
    """
    tau = tau0 * np.exp(sigma)
    return tau * H
```

## 🔬 Testing Guidelines

### Unit Tests
- Test edge cases and boundary conditions
- Include tests with known analytical results
- Test error handling and validation
- Use descriptive test names

### Integration Tests  
- Test complete workflows
- Verify figure generation
- Check experimental protocol outputs
- Validate mathematical consistency

### Example Test
```python
import pytest
import numpy as np
from ltqg_core import TimeTransform

class TestTimeTransform:
    def test_sigma_from_tau_basic(self):
        """Test basic σ = log(τ/τ₀) transformation."""
        transform = TimeTransform(tau0=1.0)
        tau = np.array([0.1, 1.0, 10.0])
        expected = np.array([-2.302585, 0.0, 2.302585])
        result = transform.sigma_from_tau(tau)
        np.testing.assert_allclose(result, expected, rtol=1e-6)
        
    def test_roundtrip_consistency(self):
        """Test τ → σ → τ roundtrip consistency."""
        transform = TimeTransform(tau0=2.71828)
        tau_original = np.logspace(-3, 3, 100)
        sigma = transform.sigma_from_tau(tau_original)
        tau_recovered = transform.tau_from_sigma(sigma)
        np.testing.assert_allclose(tau_recovered, tau_original, rtol=1e-12)
        
    def test_invalid_input(self):
        """Test error handling for invalid inputs."""
        transform = TimeTransform()
        with pytest.raises(ValueError, match="Proper time.*must be positive"):
            transform.sigma_from_tau(np.array([-1.0, 0.0, 1.0]))
```

## 📚 Documentation

### Code Documentation
- All public functions must have docstrings
- Use NumPy-style docstrings
- Include mathematical formulations where relevant
- Provide usage examples

### Educational Content
- Add notebook cells explaining new concepts
- Include interactive visualizations
- Provide physical intuition alongside mathematics
- Test all code examples

### Paper/Research Documentation
- Update research paper for theoretical extensions
- Include new experimental predictions
- Add references to relevant literature
- Maintain mathematical rigor

## 🎯 Areas for Contribution

### High Priority
- [ ] Quantum Field Theory extension to σ-time
- [ ] Detailed cosmological model predictions  
- [ ] Advanced experimental protocol optimization
- [ ] Performance optimization for large-scale simulations

### Medium Priority
- [ ] Additional visualization options
- [ ] Interactive web-based demonstrations
- [ ] Integration with experimental data analysis tools
- [ ] Educational video/animation content

### Research Extensions
- [ ] Loop Quantum Gravity comparison
- [ ] String Theory σ-time formulation
- [ ] Causal set theory connections
- [ ] Thermodynamic implications

## 🔍 Review Process

All contributions undergo review focusing on:

1. **Correctness**: Mathematical and physical accuracy
2. **Clarity**: Code readability and documentation quality
3. **Consistency**: Adherence to project style and conventions
4. **Testing**: Adequate test coverage and validation
5. **Performance**: Computational efficiency considerations

## 📞 Getting Help

- **Questions**: Open a GitHub issue with the "question" label
- **Discussions**: Use GitHub Discussions for broader topics
- **Documentation**: Check the [project website](https://denzilgreenwood.github.io/log_time/)
- **Examples**: Refer to the educational notebook and demo scripts

## 🏆 Recognition

Contributors will be acknowledged in:
- README contributor list
- Research paper acknowledgments (for significant theoretical contributions)
- Release notes for each version
- Project website credits

Thank you for contributing to advancing our understanding of quantum gravity! 🌌