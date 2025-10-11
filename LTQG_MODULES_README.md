# LTQG Comprehensive Visualization Suite

## ⚛️ Complete LTQG Module Collection

This enhanced visualization now includes **6 complete LTQG demonstration modules**, each showing different aspects of Log-Time Quantum Gravity:

### 🔄 Core Clock Drift Demo
- **σ-uniform vs τ-uniform time evolution**
- Interactive black hole horizon with real-time clock synchronization
- Data export for numerical analysis of σ-offset curves

### ⚛️ 1) Big-Bang "Reverse Funnel"
**Physics**: Scale factor evolution a(σ) = e^{nσ}
```javascript
// Radiation epoch (n=1/2) or matter epoch (n=2/3)
const aBB = Math.exp(nBB * sigmaBB);
```
- **Visual**: Flaring cylinder that expands exponentially with σ-time
- **Key insight**: No singular beginning—just a finite throat that grows
- **Includes**: Comoving galaxy points that scale ∝ a(σ)

### 🌠 2) Wormhole / Einstein–Rosen Bridge  
**Physics**: Throat geometry z(r,σ) = √(r²/r₀² - 1) e^{-|σ|/κ}
```javascript
const shrink = Math.exp(-Math.abs(sigmaWH)/kWH);
```
- **Visual**: Two mirrored funnels connected at σ-throat
- **Key insight**: Bridge stays regular—no pinch singularities
- **Behavior**: Both halves contract smoothly as |σ| increases

### 🌌 3) de Sitter Expansion "Bubble"
**Physics**: Exponential expansion a(σ) = exp(H τ₀ e^σ)
```javascript
const a = Math.exp(H * tau0dS * Math.exp(sigmaDeSitter));
bubble.material.opacity = 1.0/Math.sqrt(a); // Surface brightness ∝ 1/a
```
- **Visual**: Transparent sphere growing exponentially
- **Key insight**: Surface dims as 1/a—observational redshift effect
- **Display**: Auto-scaling with opacity modulation

### 🔬 4) 1D Quantum Well in σ (Web Worker)
**Physics**: σ-Schrödinger equation iℏ ∂σψ = τ₀ e^σ H ψ
```javascript
// Worker: sigma_schrodinger.js
const tauEff = tau0 * Math.exp(sigma);
// Crank-Nicolson stepping with exp(-i τ₀ e^σ H dσ / ħ)
```
- **Visual**: Real-time line plot of |ψ(x,σ)|² evolution
- **Key insight**: Freeze-out for σ→-∞, rapid oscillation for σ→+∞  
- **Setup**: Harmonic oscillator with Gaussian wave packet

### 🌊 5) σ-Gravitational Wave Funnel
**Physics**: σ-damped amplitude A(σ) = A₀ e^{-γ e^σ}
```glsl
// Vertex shader displacement
float A = A0 * exp(-gamma * exp(sigma));
float phase = kWave * length(position.xz) - omega * t;
pos.y += A * sin(phase);
```
- **Visual**: Rippling funnel surface with shader-based waves
- **Key insight**: Waves self-quench as σ→-∞ (natural regularization)
- **Implementation**: Custom GLSL vertex/fragment shaders

## 🎮 Interactive Controls

### Module Toggles
- **⚛️ Big Bang**: Toggle reverse funnel + galaxy points
- **🌠 Wormhole**: Toggle Einstein-Rosen bridge pair
- **🌌 de Sitter**: Toggle expansion bubble
- **🔬 Quantum Well**: Toggle σ-QM evolution line
- **🌊 σ-GW Funnel**: Toggle gravitational wave surface

### Enhanced UI
- **5 modules active** status indicator
- **🌀 Auto-orbit**: Automatic camera rotation
- **📊 Export Data**: JSON download with full σ-evolution data
- **Visual presets**: Earth-like vs Near-horizon scenarios

## 🔧 Technical Implementation

### Modular Architecture
Each module follows the pattern:
1. **Setup once**: Geometry + material creation
2. **Update per frame**: Physics-driven parameter evolution
3. **UI integration**: Toggle visibility + status tracking

### Performance Optimizations
- **Background Worker**: Quantum evolution off main thread
- **Shader-based GW**: GPU vertex displacement for waves
- **Smart scaling**: Clamped growth for stable visualization
- **Memory management**: Limited data export arrays

### ES Module Structure
```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
// Clean module imports with importmap resolution
```

## 📊 Data Export Format

```json
{
  "timestamp": "2025-10-11T...",
  "mode": "sigma",
  "parameters": {
    "rs": 1.0,
    "rnear": 1.2, 
    "rfar": 5.0,
    "omega": 1.0,
    "tau0": 1.0
  },
  "deltaValues": [
    {
      "time": 1760179164817,
      "mode": "sigma",
      "deltaSigma": 0.0023,
      "theoreticalDeltaSigma": 0.0024,
      "deltaTheta": 0.157,
      // ... full measurement arrays
    }
  ]
}
```

## 🚀 Quick Start

1. **Launch**: Open `ltqg_clock_drift.html` in modern browser
2. **Explore**: Toggle modules on/off to focus on specific physics
3. **Analyze**: Export data for numerical σ-offset analysis
4. **Learn**: Each module demonstrates different LTQG regularization

## 🔮 What You'll See

- **Big Bang**: Smooth expansion from finite throat (no singularity)
- **Wormhole**: Stable bridge connecting two spacetime regions  
- **de Sitter**: Exponential cosmic expansion with redshift dimming
- **Quantum**: Wave packet evolution showing σ-dependent dynamics
- **GW Funnel**: Self-regularizing gravitational waves

**The key insight**: All LTQG modules show natural regularization—physics stays finite and well-behaved even in extreme regimes where classical theory diverges.

---

### 📚 Physics References
- Log-time coordinates: σ = ln(τ/τ₀)
- LTQG clock synchronization: Equal Δσ → Different Δτ  
- σ-Schrödinger evolution: iℏ ∂σψ = τ₀ e^σ H ψ
- Gravitational redshift: α(r) = √(1 - rs/r)