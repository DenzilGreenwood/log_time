# log_time_visualizations.py
# Visualizing σ = log(τ/τ0) across singularities, horizons, and FLRW cosmology
# Usage examples:
#   python log_time_visualizations.py
#   python log_time_visualizations.py --tau0 1.0 --H 0.5 --rs 1.0 --rmax 12 --save --outdir figs --dpi 200

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Core transforms
# -----------------------------
def sigma_from_tau(tau, tau0=1.0):
    """σ = log(τ/τ0), with τ, τ0 > 0."""
    return np.log(tau / tau0)

def tau_from_sigma(sigma, tau0=1.0):
    """τ = τ0 * e^σ."""
    return tau0 * np.exp(sigma)

def safe_log(x, floor=1e-300):
    """Numerically safe log with positive clamp."""
    return np.log(np.clip(x, floor, None))

# -----------------------------
# 1) Singularities (τ → 0) & H_eff scaling
# -----------------------------
def visualize_singularity(tau0=1.0, save=False, outdir=".", dpi=150):
    # τ-space: approach 0 from the right (log grid)
    tau = np.logspace(-6, 2, 400)
    Q_tau = 1.0 / tau  # a typical divergent quantity Q ∝ 1/τ

    # σ-space (linear grid)
    sigma = np.linspace(-20, 6, 400)
    Q_sigma = (1.0 / tau0) * np.exp(-sigma)        # Q(σ) ∝ e^{-σ} / τ0
    H_eff   = tau0 * np.exp(sigma) * 1.0           # set H=1 for scale

    # Plot 1: divergence in τ
    fig1 = plt.figure(figsize=(7, 5))
    plt.loglog(tau, Q_tau)
    plt.title("Divergence in τ-space: Q ∝ 1/τ")
    plt.xlabel("τ")
    plt.ylabel("Q(τ)")
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    if save:
        fig1.savefig(os.path.join(outdir, "singularity_Q_tau.png"), dpi=dpi)

    # Plot 2: σ-regularized behavior
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(sigma, Q_sigma)
    plt.title("Regularized in σ-space: Q(σ) ∝ e^{-σ}")
    plt.xlabel("σ = log(τ/τ0)")
    plt.ylabel("Q(σ)")
    plt.grid(True, ls=":")
    plt.tight_layout()
    if save:
        fig2.savefig(os.path.join(outdir, "singularity_Q_sigma.png"), dpi=dpi)

    # Plot 3: effective Hamiltonian scale
    fig3 = plt.figure(figsize=(7, 5))
    plt.plot(sigma, H_eff)
    plt.title("Effective Hamiltonian scale: H_eff(σ) = τ0 e^{σ} H (H=1)")
    plt.xlabel("σ = log(τ/τ0)")
    plt.ylabel("H_eff(σ)")
    plt.axhline(0, lw=0.8)
    plt.grid(True, ls=":")
    plt.tight_layout()
    if save:
        fig3.savefig(os.path.join(outdir, "singularity_H_eff.png"), dpi=dpi)

# -----------------------------
# 2) Near-horizon physics: σ(r) = log α(r) + log(τ/τ0)
# -----------------------------
def alpha_standard(r, rs):
    """Schwarzschild redshift outside the horizon: α(r) = sqrt(1 - rs/r) → 0 as r→rs+."""
    return np.sqrt(1.0 - (rs / r))

def alpha_alt_text(r, rs):
    """
    Heuristic divergent form sometimes used in back-of-envelope derivations:
    α(r) = 2 rs / (r - rs) → ∞ as r→rs+.
    Shown for contrast with the standard form.
    """
    return 2.0 * rs / (r - rs)

def visualize_horizon(rs=1.0, rmax=10.0, tau_ratio_term=0.0, save=False, outdir=".", dpi=150):
    # Sample just outside the horizon; avoid r=rs exactly
    r = np.linspace(rs + 1e-6, rmax, 1000)

    a_std = alpha_standard(r, rs)
    a_alt = alpha_alt_text(r, rs)

    sigma_std = safe_log(a_std) + tau_ratio_term
    sigma_alt = safe_log(a_alt) + tau_ratio_term

    # Plot α(r) for intuition (log y)
    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(r/rs, a_std, label=r"α_std(r) = $\sqrt{1 - r_s/r}$")
    plt.plot(r/rs, a_alt, label=r"α_alt(r) = $2 r_s/(r - r_s)$")
    plt.yscale("log")
    plt.title("Redshift models near a Schwarzschild horizon")
    plt.xlabel("r / r_s")
    plt.ylabel("α(r)")
    plt.grid(True, ls=":")
    plt.legend()
    # annotate the horizon
    plt.axvline(1.0, ls="--")
    plt.tight_layout()
    if save:
        fig1.savefig(os.path.join(outdir, "horizon_alpha.png"), dpi=dpi)

    # Plot σ(r)
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(r/rs, sigma_std, label=r"σ_std(r) = log α_std + log(τ/τ0)")
    plt.plot(r/rs, sigma_alt, label=r"σ_alt(r) = log α_alt + log(τ/τ0)")
    plt.title("σ(r) = log α(r) + log(τ/τ0)")
    plt.xlabel("r / r_s")
    plt.ylabel("σ(r)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.axvline(1.0, ls="--")
    plt.tight_layout()
    if save:
        fig2.savefig(os.path.join(outdir, "horizon_sigma.png"), dpi=dpi)

# -----------------------------
# 3) FLRW cosmology in σ-time
# -----------------------------
def visualize_flrw(tau0=1.0, H=1.0, save=False, outdir=".", dpi=150):
    """
    Compare σ-frame expansion for:
      - Radiation: a ∝ t^{1/2}  → (1/a) da/dσ = 1/2
      - Matter:    a ∝ t^{2/3}  → (1/a) da/dσ = 2/3
      - de Sitter: a = exp(H t) → (1/a) da/dσ = H τ0 e^{σ}
    """
    sigma = np.linspace(-8, 4, 500)
    t = tau_from_sigma(sigma, tau0=tau0)

    # Power-law cosmologies
    n_rad = 0.5
    n_mat = 2.0/3.0
    a_rad = np.exp(n_rad * sigma)         # since a(σ) = e^{nσ}
    a_mat = np.exp(n_mat * sigma)

    # de Sitter (inflation) with t = τ0 e^{σ}
    a_des = np.exp(H * t)

    # σ-frame expansion rates
    rate_rad = np.full_like(sigma, n_rad)
    rate_mat = np.full_like(sigma, n_mat)
    rate_des = H * tau0 * np.exp(sigma)

    # Plot: scale factors in σ
    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(sigma, a_rad, label="Radiation  a(σ)=e^{(1/2)σ}")
    plt.plot(sigma, a_mat, label="Matter     a(σ)=e^{(2/3)σ}")
    plt.plot(sigma, a_des, label="de Sitter  a=exp(H τ0 e^{σ})")
    plt.yscale("log")
    plt.title("Scale factor in σ-time")
    plt.xlabel("σ = log(τ/τ0)")
    plt.ylabel("a(σ)")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    if save:
        fig1.savefig(os.path.join(outdir, "flrw_scale_factor.png"), dpi=dpi)

    # Plot: σ-frame expansion rate
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(sigma, rate_rad, label="Radiation: (1/a) da/dσ = 1/2")
    plt.plot(sigma, rate_mat, label="Matter: (1/a) da/dσ = 2/3")
    plt.plot(sigma, rate_des, label="de Sitter: (1/a) da/dσ = H τ0 e^{σ}")
    plt.title("σ-frame expansion rate")
    plt.xlabel("σ = log(τ/τ0)")
    plt.ylabel("(1/a) · da/dσ")
    plt.grid(True, ls=":")
    plt.legend()
    plt.tight_layout()
    if save:
        fig2.savefig(os.path.join(outdir, "flrw_sigma_rate.png"), dpi=dpi)

# -----------------------------
# Main / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visualize log-time σ = log(τ/τ0) in several GR/QFT contexts.")
    p.add_argument("--tau0", type=float, default=1.0, help="Reference proper time τ0.")
    p.add_argument("--H", type=float, default=0.5, help="Hubble parameter for de Sitter visualization.")
    p.add_argument("--rs", type=float, default=1.0, help="Schwarzschild radius r_s.")
    p.add_argument("--rmax", type=float, default=10.0, help="Max radius for horizon plots (in units of r_s).")
    p.add_argument("--tau_ratio_term", type=float, default=0.0, help="Additive log(τ/τ0) term for σ(r).")
    p.add_argument("--save", action="store_true", help="Save figures as PNGs instead of only showing.")
    p.add_argument("--outdir", type=str, default=".", help="Directory to save figures.")
    p.add_argument("--dpi", type=int, default=150, help="Figure DPI when saving.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.save and not os.path.isdir(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    visualize_singularity(tau0=args.tau0, save=args.save, outdir=args.outdir, dpi=args.dpi)
    visualize_horizon(rs=args.rs, rmax=args.rmax, tau_ratio_term=args.tau_ratio_term,
                      save=args.save, outdir=args.outdir, dpi=args.dpi)
    visualize_flrw(tau0=args.tau0, H=args.H, save=args.save, outdir=args.outdir, dpi=args.dpi)

    plt.show()