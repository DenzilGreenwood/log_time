// ⚛️ 4) 1D Quantum Well in σ (numerical core)
// Worker for solving σ-Schrödinger: iℏ ∂σψ = τ₀ e^σ H ψ

self.onmessage = (e) => {
  const {N, dx, dtSigma, steps, tau0, sigma0, Varr, psiRe, psiIm} = e.data;
  const hbar = 1.0, m = 1.0;
  let sigma = sigma0;
  let Re = Float64Array.from(psiRe), Im = Float64Array.from(psiIm);
  
  // Laplacian with periodic boundary conditions
  const lap = (arr, i) => (arr[(i+1)%N] - 2*arr[i] + arr[(i-1+N)%N])/(dx*dx);
  
  for (let n = 0; n < steps; n++) {
    const tauEff = tau0 * Math.exp(sigma);
    
    // 1st-order split-step (OK for demo): exp(-i τ₀ e^σ H dσ / ħ)
    for (let i = 0; i < N; i++) {
      const HRe = -(hbar*hbar/(2*m))*lap(Re,i) + Varr[i]*Re[i];
      const HIm = -(hbar*hbar/(2*m))*lap(Im,i) + Varr[i]*Im[i];
      const dRe = -(tauEff*dtSigma/hbar) * HIm;
      const dIm = +(tauEff*dtSigma/hbar) * HRe;
      Re[i] += dRe; 
      Im[i] += dIm;
    }
    sigma += dtSigma;
    
    // Send progress updates every 8 steps
    if (n % 8 === 0) {
      self.postMessage({sigma, Re: Array.from(Re), Im: Array.from(Im)});
    }
  }
  
  self.postMessage({done: true, sigma, Re: Array.from(Re), Im: Array.from(Im)});
};