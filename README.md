# 🌌 Log-Time Quantum Gravity (LTQG)

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://denzilgreenwood.github.io/log_time/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-orange.svg)](https://github.com/DenzilGreenwood/log_time)

> **Bridging the temporal divide between General Relativity and Quantum Mechanics through logarithmic time coordinates**

A comprehensive Python implementation of the Log-Time Quantum Gravity framework, which reconciles General Relativity's multiplicative time-dilation structure with Quantum Mechanics' additive phase evolution through the elegant transformation: **σ = log(τ/τ₀)**

## 🚀 Quick Links

- 📖 **[Interactive Website](https://denzilgreenwood.github.io/log_time/)** - Explore LTQG concepts visually
- 📄 **[Main Research Paper](docs/LTQG_Research_Paper.pdf)** - Complete theoretical framework
- 📄 **[Problem of Time Paper](docs/LTQG_Problem_of_Time.pdf)** - Canonical quantum gravity applications
- 📓 **[Educational Notebook](LTQG_Educational_Notebook.ipynb)** - Step-by-step tutorial
- 🖼️ **[Figure Gallery](figs/)** - All publication-quality visualizations

## 🎯 What is LTQG?

Log-Time Quantum Gravity addresses the fundamental incompatibility between:

- **General Relativity**: Multiplicative time dilation `τ' = γ(v, Φ) τ`
- **Quantum Mechanics**: Additive phase evolution `φ = Et/ℏ`

### 💡 The Key Insight

The logarithmic transformation **σ = log(τ/τ₀)** converts multiplication to addition:

```
τ_B = α τ_A  ⟹  σ_B = σ_A + log(α)
```

This enables natural unification while providing automatic **singularity regularization** through "asymptotic silence" - the vanishing of quantum evolution generators as σ → -∞.

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|---------|
| 🔬 **Core Framework** | Complete mathematical implementation | ✅ |
| 📊 **Visualizations** | All paper figures with publication quality | ✅ |
| 🧪 **Experiments** | Testable predictions & protocols | ✅ |
| 📚 **Education** | Interactive Jupyter notebook tutorial | ✅ |
| 📖 **Documentation** | Research paper & comprehensive guides | ✅ |
| 🌐 **Website** | Interactive GitHub Pages site | ✅ |

### 🔬 Core Framework
- Time transformations between proper time τ and log-time σ
- Singularity regularization mechanisms
- Modified Schrödinger evolution in σ-time
- Gravitational redshift as additive σ-shifts
- Asymptotic silence condition implementation
- Unitary quantum evolution preservation

### 📊 Visualizations
Implementation of all figures from the LTQG paper:
1. **Log-Time Map**: σ = log(τ/τ₀) transformation
2. **Singularity Regularization**: Smooth behavior in σ-space
3. **Gravitational Redshift**: Multiplicative → additive conversion
4. **Effective Generator**: Asymptotic silence condition
5. **Zeno Protocols**: σ-uniform vs τ-uniform predictions
6. **Early Universe**: Mode evolution without trans-Planckian problems

### 🧪 Experimental Protocols
- **σ-Uniform Zeno/Anti-Zeno Protocols**: Testing quantum measurement under gravitational redshift
- **Analog Gravity Interferometry**: Tabletop tests using optical systems or BECs
- **Early Universe Signatures**: CMB predictions with LTQG modifications
- **Clock Transport Loops**: Path-dependent quantum phase accumulation

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/DenzilGreenwood/log_time.git
cd log_time

# Install dependencies
pip install -r requirements.txt
```

### Generate All Figures
```bash
# Complete demonstration with all outputs
python ltqg_demo.py --generate-all

# Publication figures only
python ltqg_demo.py --figures-only

# Experimental analysis only  
python ltqg_demo.py --experiments-only
```

### Interactive Tutorial
```bash
# Launch Jupyter notebook
jupyter notebook LTQG_Educational_Notebook.ipynb
```

## 📊 Example Usage

```python
from ltqg_core import create_ltqg_simulator
from ltqg_visualization import LTQGVisualizer
import numpy as np

# Create LTQG simulator
simulator = create_ltqg_simulator(tau0=1.0)

# Time transformation
tau = np.logspace(-6, 3, 100)
sigma = simulator.time_transform.sigma_from_tau(tau)

# Generate visualizations
visualizer = LTQGVisualizer(save_dir="figs")
figures = visualizer.generate_all_figures(save=True)

# Run experimental analysis
from ltqg_experiments import ExperimentalSuite
suite = ExperimentalSuite()
results = suite.run_comprehensive_analysis()
print(suite.generate_experimental_summary())
```

## 📁 Project Structure

```
log-time-quantum-gravity/
├── 📚 Core Implementation
│   ├── ltqg_core.py              # Mathematical framework (607 lines)
│   ├── ltqg_visualization.py     # Complete visualization suite (691 lines)
│   ├── ltqg_experiments.py       # Experimental protocols (757+ lines)
│   └── ltqg_demo.py              # Demonstration system (371+ lines)
│
├── 📓 Educational Materials
│   ├── LTQG_Educational_Notebook.ipynb  # Interactive tutorial (27 cells)
│   ├── LTQG_Research_Paper.tex          # Main research paper source
│   └── LTQG_Problem_of_Time.tex         # Problem of time paper source
│
├── 📊 Generated Content
│   ├── figs/                     # Publication-quality figures (16 images)
│   └── docs/                     # GitHub Pages website & PDFs
│
├── 📋 Documentation
│   ├── README.md                 # This file
│   ├── CONTRIBUTING.md           # Contribution guidelines
│   ├── CHANGELOG.md              # Version history
│   ├── DEPLOYMENT.md             # GitHub deployment guide
│   └── LICENSE                   # MIT license
│
└── ⚙️ Configuration
    ├── requirements.txt          # Python dependencies
    ├── .gitignore                # Git ignore rules
    └── .github/workflows/        # GitHub Actions (Pages deployment)
```

## 🎯 Quick Actions

| Action | Command | Description |
|--------|---------|-------------|
| 🎬 **Demo Everything** | `python ltqg_demo.py --generate-all` | Complete analysis with all outputs |
| 🖼️ **Generate Figures** | `python ltqg_demo.py --figures-only` | Publication-quality visualizations |
| 🧪 **Run Experiments** | `python ltqg_demo.py --experiments-only` | Experimental predictions analysis |
| 📚 **Interactive Tutorial** | `jupyter notebook LTQG_Educational_Notebook.ipynb` | Step-by-step learning |
| 🌐 **View Website** | Open `docs/index.html` | GitHub Pages interface |

## Mathematical Foundation

### Core Transformation
The fundamental insight of LTQG is the logarithmic time transformation:

```
σ = log(τ/τ₀)
```

where τ is proper time and τ₀ is a reference time constant (typically Planck time).

### Modified Schrödinger Equation
In σ-time, quantum evolution follows:

```
iℏ ∂|ψ⟩/∂σ = K(σ)|ψ⟩ = τ₀ exp(σ) H|ψ⟩
```

### Singularity Regularization
Classical divergences Q(τ) ∝ τ⁻ⁿ become regularized:

```
Q(σ) = (1/τ₀ⁿ) exp(-nσ) → 0 as σ → -∞
```

### Gravitational Redshift Unification
Multiplicative time dilation τ_B = α τ_A becomes additive σ-shift:

```
σ_B = σ_A + log(α)
```

## Experimental Predictions

LTQG makes several falsifiable predictions:

1. **σ-Uniform Zeno Suppression**: Quantum measurements at uniform σ intervals show redshift-dependent suppression
2. **Additive Interferometric Phases**: Gravitational effects appear as additive phase shifts rather than multiplicative
3. **Early Universe Regularization**: Smooth cosmological evolution without trans-Planckian problems
4. **Path-Dependent Clock Transport**: Quantum phases depend on redshift history in closed loops

## Generated Outputs

Running the demonstration generates:

### Figures (`figs/` directory)
- `log_time_map.png` - Time transformation visualization
- `singularity_regularization.png` - Curvature regularization 
- `gravitational_redshift_shift.png` - Redshift unification
- `effective_generator_silence.png` - Asymptotic silence
- `zeno_protocol_predictions.png` - Experimental protocols
- `early_universe_modes.png` - Cosmological evolution
- `experimental_feasibility.png` - Implementation assessment

### Analysis Reports
- Comprehensive experimental feasibility assessment
- Statistical significance calculations for each protocol
- Theoretical comparison with other quantum gravity approaches
- Performance benchmarks for numerical calculations

## Physics Insights

### Temporal Unification
LTQG resolves the fundamental temporal mismatch between GR and QM:
- **GR**: Local, geometric, multiplicative time dilation
- **QM**: Global, linear, additive phase evolution  
- **LTQG**: Unified additive structure in σ-coordinates

### Singularity Resolution
Classical spacetime singularities become:
- Smooth asymptotic boundaries in σ-time
- Regions of "asymptotic silence" where quantum evolution halts
- Finite-norm quantum states without divergence

### Operational Distinguishability
LTQG is experimentally distinguishable from standard QM through:
- σ-uniform vs τ-uniform measurement protocols
- Gravitational interferometry with analog systems
- Early universe observational signatures
- Precision clock transport experiments

## Performance

The implementation is optimized for:
- **Large-scale simulations**: Efficient NumPy vectorization
- **High precision**: Numerically stable algorithms  
- **Modular design**: Easy to extend and modify
- **Educational use**: Clear, well-documented code

Typical performance on modern hardware:
- Time transformation (1M points): ~0.01 seconds
- Singularity regularization (100K points): ~0.005 seconds  
- Complete early universe simulation: ~0.1 seconds
- Full figure generation suite: ~10 seconds

## 🌐 Interactive Website

The project includes a beautiful GitHub Pages website with:

- **Visual concept exploration** with interactive design
- **Complete figure gallery** with detailed explanations  
- **Mathematical foundations** presented clearly
- **Experimental predictions** with feasibility analysis
- **Quick start guides** and tutorial links
- **Download links** for papers and notebooks

Visit: **[https://denzilgreenwood.github.io/log_time/](https://denzilgreenwood.github.io/log_time/)**

## 📋 Requirements

- **Python 3.8+**
- **NumPy** ≥ 1.21.0  
- **SciPy** ≥ 1.7.0
- **Matplotlib** ≥ 3.5.0
- **Jupyter** (for interactive notebook)

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- 🐛 Reporting bugs and requesting features
- 🔬 Adding experimental protocols  
- 📚 Improving documentation
- 🧮 Extending mathematical frameworks
- 🎨 Enhancing visualizations

## 📄 Citation

If you use this work in research, please cite:

```bibtex
@misc{ltqg2025,
  title={Log-Time Quantum Gravity: A Reparameterization Approach to Temporal Unification},
  author={Denzil James Greenwood},
  year={2025},
  note={Available at: https://github.com/DenzilGreenwood/log_time}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🌌 Log-Time Quantum Gravity 🌌**

*Bridging the temporal divide between General Relativity and Quantum Mechanics*

[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://denzilgreenwood.github.io/log_time/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>