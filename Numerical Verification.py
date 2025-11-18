import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# ==========================================
# Configuration & Path Setup
# ==========================================

USER_PATH = "/Users/lichenghao/Downloads/"
if os.path.exists(USER_PATH):
    OUTPUT_DIR = USER_PATH
else:
    OUTPUT_DIR = os.getcwd()
    print(f"Note: User path not found. Saving figures to current directory: {OUTPUT_DIR}")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (6, 4),
    'lines.linewidth': 1.5,
    'savefig.dpi': 300
})

# ==========================================
# Model Primitives (Base Parameters)
# ==========================================
MU = 0.6       # Informed trader intensity
Q = 0.75       # Honest signal accuracy
PHI1 = 1.0     # Manipulator strategy (theta=1)
PHI0 = 0.0     # Manipulator strategy (theta=0)

# ==========================================
# Core Functions
# ==========================================

def odds_to_prob(odds):
    """Converts odds to probability."""
    return odds / (1.0 + odds)


def calculate_spread_metrics(p0, rho, delta, mu=MU, q=Q, phi1=PHI1, phi0=PHI0):
    """
    Computes the Exact Spread, Baseline Spread, and First-Order Approximation.

    Returns:
        S_exact: The spread calculated using full Bayesian updating.
        S_approx: The spread calculated using the first-order Taylor expansion.
        S_base: The baseline spread without coordination manipulation (delta=1).
        Delta_S: The spread expansion component (epsilon * Xi).
    """
    # 1. Baseline (Order Flow Only)
    O_0 = p0 / (1.0 - p0)
    Lambda_y_plus = (1.0 + mu) / (1.0 - mu)
    Lambda_y_minus = (1.0 - mu) / (1.0 + mu)

    p0_plus = odds_to_prob(O_0 * Lambda_y_plus)
    p0_minus = odds_to_prob(O_0 * Lambda_y_minus)
    S_base = p0_plus - p0_minus

    # 2. Manipulation Parameters
    epsilon = (1.0 - rho) * (1.0 - delta)
    kappa = phi1 / (rho * q) - phi0 / (rho * (1.0 - q))

    # 3. Exact Spread Calculation
    Lambda_x_1 = (rho * q + epsilon * phi1) / (rho * (1.0 - q) + epsilon * phi0)
    O_ask = O_0 * Lambda_y_plus * Lambda_x_1
    ask_price_exact = odds_to_prob(O_ask)

    # Bid Price (Sell order + Negative Content)
    O_bid = O_0 * Lambda_y_minus * (1.0 / Lambda_x_1)
    bid_price_exact = odds_to_prob(O_bid)

    S_exact = ask_price_exact - bid_price_exact

    # 4. First-Order Approximation
    h_p_plus = p0_plus * (1.0 - p0_plus)
    h_p_minus = p0_minus * (1.0 - p0_minus)

    # Xi coefficient (Sum of curvatures)
    Xi = kappa * (h_p_plus + h_p_minus)

    Delta_S = epsilon * Xi
    S_approx = S_base + Delta_S

    return S_exact, S_approx, S_base, Delta_S

def calculate_deterrence_threshold(Pi, k, F):
    """
    Calculates the dynamic deterrence threshold delta*.
    Formula: delta* = (Pi - k) / (Pi + F)
    """
    denom = Pi + F
    val = (Pi - k) / denom
    return np.clip(val, 0.0, 1.0)

# ==========================================
# Figure Generation
# ==========================================

def generate_figure_1():
    """
    Figure 1: Accuracy of First-Order Approximation (Plotting Expansion)
    Compares the Exact Spread *Expansion* vs. the 1st-Order *Expansion*.
    """
    print("Generating Figure 1...")
    p0_fixed = 0.3
    rho_values = [0.7, 0.8, 0.9]
    delta_grid = np.linspace(0.0, 1.0, 25)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    for i, rho in enumerate(rho_values):
        exact_spreads, _, _, delta_s_approx = calculate_spread_metrics(p0_fixed, rho, delta_grid)

        s_0_baseline = exact_spreads[-1]

        exact_expansion = exact_spreads - s_0_baseline

        ax = axes[i]
        ax.plot(delta_grid, exact_expansion, 'o-', markersize=4, label='Exact Expansion ($\Delta S$)', color='#1f77b4')
        ax.plot(delta_grid, delta_s_approx, 's--', markersize=4, label='1st-Order Approx ($\epsilon \Xi$)', color='#ff7f0e')

        ax.set_title(f'Honest Fraction $\\rho = {rho}$')
        ax.set_xlabel('Detection Probability $\\delta$')
        if i == 0:
            ax.set_ylabel('Spread Expansion ($\Delta S$)')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'Figure 1: Accuracy of Approximation ($p_0={p0_fixed}$)', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "Figure_1_Approximation.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")


def generate_figure_2():
    """
    Figure 2: Determinants of the Deterrence Threshold (delta*).
    Sensitivity analysis of delta* with respect to Pi, k, and F.
    Base case: Pi=100, k=20, F=50 -> delta* = 0.533
    """
    print("Generating Figure 2...")

    # Base parameters
    Pi_base = 100.0
    k_base = 20.0
    F_base = 50.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Sensitivity to Profit (Pi)
    pi_range = np.linspace(25, 200, 100)
    d_pi = calculate_deterrence_threshold(pi_range, k_base, F_base)
    axes[0].plot(pi_range, d_pi, linewidth=2, color='#2ca02c')
    axes[0].set_xlabel(r'Manipulation Profit ($\Pi$)')
    axes[0].set_ylabel(r'Deterrence Threshold $\delta^*$')
    axes[0].set_title(r'Effect of $\Pi$ on $\delta^*$')

    # 2. Sensitivity to Cost (k)
    k_range = np.linspace(5, 50, 100)
    d_k = calculate_deterrence_threshold(Pi_base, k_range, F_base)
    axes[1].plot(k_range, d_k, linewidth=2, color='#d62728')
    axes[1].set_xlabel(r'Coordination Cost ($k$)')
    axes[1].set_title(r'Effect of $k$ on $\delta^*$')

    # 3. Sensitivity to Penalty (F)
    f_range = np.linspace(10, 150, 100)
    d_f = calculate_deterrence_threshold(Pi_base, k_base, f_range)
    axes[2].plot(f_range, d_f, linewidth=2, color='#9467bd')
    axes[2].set_xlabel(r'Detection Penalty ($F$)')
    axes[2].set_title(r'Effect of $F$ on $\delta^*$')

    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)
        base_delta = calculate_deterrence_threshold(Pi_base, k_base, F_base)
        ax.axhline(base_delta, color='gray', linestyle='--', alpha=0.5, label='Base Case')

    plt.suptitle('Figure 2: Sensitivity of Deterrence Threshold $\delta^*$', y=1.02)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "Figure_2_Deterrence.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")


def generate_figure_3():
    """
    Figure 3: 3D Surface Plot of Spread Change.
    X-axis: Detection Probability (delta)
    Y-axis: Prior Belief (p0)
    Z-axis: Spread Change (Delta S)
    Demonstrates robustness across the full range of priors.
    """
    print("Generating Figure 3 (3D Surface)...")
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    rho_fixed = 0.5

    delta_vals = np.linspace(0.5, 1.0, 50)
    p0_vals = np.linspace(0.1, 0.9, 50)
    DELTA, P0 = np.meshgrid(delta_vals, p0_vals)

    Z = np.zeros_like(DELTA)

    for i in range(DELTA.shape[0]):
        for j in range(DELTA.shape[1]):
            d = DELTA[i, j]
            p = P0[i, j]
            _, _, _, delta_s = calculate_spread_metrics(p, rho_fixed, d)
            Z[i, j] = delta_s

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(DELTA, P0, Z, cmap=cm.viridis,
                           linewidth=0, antialiased=False, alpha=0.9)

    fig.colorbar(surf, shrink=0.5, aspect=10, label=r'Spread Change $\Delta S$')

    ax.set_xlabel(r'Detection Probability $\delta$')
    ax.set_ylabel(r'Prior Belief $p_0$')
    ax.set_zlabel(r'Spread Change $\Delta S$')

    ax.view_init(elev=30, azim=135)

    plt.title(f'Figure 3: Robustness to Prior Skewness ($\\rho={rho_fixed}$)')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "Figure_3_3D_Invariance.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved: {save_path}")

# ==========================================
# Main Execution
# ==========================================

if __name__ == "__main__":
    print("Starting numerical verification for 'Synchronization as Manipulation'...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    generate_figure_1()
    generate_figure_2()
    generate_figure_3()

    print("\nAll figures generated successfully.")
