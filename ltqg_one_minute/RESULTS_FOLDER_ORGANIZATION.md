# LTQG Results Folder Organization

This document explains the organized folder structure for exporting images and results from the LTQG core concepts modules.

## Folder Structure

```
ltqg_one_minute/
├── core_concepts/
│   ├── results/                    # Root results directory
│   │   ├── asymptotic_silence/     # Results from asymptotic_silence.py
│   │   │   ├── H_eff_norms_evolution.png
│   │   │   ├── information_preservation_evolution.png
│   │   │   └── redshift_validation_1to1.png
│   │   ├── effective_hamiltonian/  # Results from effective_hamiltonian.py
│   │   ├── ltqg_core/             # Results from ltqg_core.py
│   │   ├── sigma_transformation/   # Results from sigma_transformation.py
│   │   └── plotting_utils/        # Test plots from plotting utilities
│   ├── asymptotic_silence.py
│   ├── asymptotic_silence_plots.py
│   ├── plotting_utils.py          # Shared plotting utilities
│   └── ...
├── kiefer_problems/
│   └── results/                   # Future: kiefer_problems results
└── worked_examples/
    └── results/                   # Future: worked_examples results
```

## Usage Pattern

### For Each Module
Each Python module in `core_concepts/` gets its own results subdirectory:
- `asymptotic_silence.py` → `results/asymptotic_silence/`
- `effective_hamiltonian.py` → `results/effective_hamiltonian/`
- `ltqg_core.py` → `results/ltqg_core/`
- `sigma_transformation.py` → `results/sigma_transformation/`

### Using the Plotting Utilities

Import and use the shared utilities in any plotting script:

```python
from plotting_utils import configure_matplotlib_for_export, save_plot, setup_results_directory

# Configure matplotlib for high-quality export
configure_matplotlib_for_export()

# Create your plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("My Analysis")

# Save to appropriate results folder
output_path = save_plot(plt, 'my_analysis.png', 'module_name')
plt.close()
```

### Directory Creation
- Directories are created automatically when needed
- No manual folder creation required
- Use `setup_results_directory('module_name')` to ensure folder exists

## Key Features

### Automatic Organization
- ✅ Each module has its own results folder
- ✅ No image clutter in main directories
- ✅ Easy to find specific module outputs
- ✅ Consistent naming convention

### High-Quality Output
- ✅ 300 DPI resolution for publications
- ✅ Consistent matplotlib styling
- ✅ White background, clean edges
- ✅ Non-interactive backend for automation

### Shared Utilities
- ✅ `plotting_utils.py` provides common functions
- ✅ Consistent save behavior across modules
- ✅ Publication-ready default settings
- ✅ Easy integration into existing code

## Current Status

### Implemented ✅
- [x] `core_concepts/results/` structure created
- [x] `asymptotic_silence/` subfolder with 3 diagnostic plots
- [x] `plotting_utils.py` shared utilities module
- [x] Automatic directory creation
- [x] High-quality image export settings

### Ready for Implementation 🔄
- [ ] `effective_hamiltonian/` plotting integration
- [ ] `ltqg_core/` visualization exports
- [ ] `sigma_transformation/` analysis plots
- [ ] Other module plotting scripts

## Example: Adding Plots to Another Module

To add plotting capability to `effective_hamiltonian.py`:

1. **Create plotting script**: `effective_hamiltonian_plots.py`
2. **Import utilities**:
   ```python
   from plotting_utils import configure_matplotlib_for_export, save_plot
   configure_matplotlib_for_export()
   ```
3. **Save plots properly**:
   ```python
   output_path = save_plot(plt, 'hamiltonian_evolution.png', 'effective_hamiltonian')
   ```
4. **Run script**: Images automatically saved to `results/effective_hamiltonian/`

## Benefits

### For Development
- **Clean Workspace**: No image files scattered in code directories
- **Easy Cleanup**: Delete entire `results/` folder to clean all outputs
- **Module Isolation**: Each module's outputs are separate and organized

### For Collaboration
- **Clear Organization**: Anyone can quickly find specific module outputs
- **Consistent Structure**: Same pattern across all modules
- **Version Control**: Can easily `.gitignore` the `results/` folder if desired

### For Publication
- **High Quality**: 300 DPI, publication-ready images
- **Batch Processing**: Easy to regenerate all plots with consistent quality
- **Selective Export**: Can export only specific module results as needed

This organization system scales naturally as the LTQG framework grows and new modules are added.