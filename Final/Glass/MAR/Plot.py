"""
Generate single panel plot for Run 7 showing m-effect
For GECCO paper
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =============================================================================
# PUBLICATION-QUALITY SETTINGS
# =============================================================================

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 12
rcParams['legend.fontsize'] = 10
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.linewidth'] = 0.8
rcParams['grid.linewidth'] = 0.5
rcParams['lines.linewidth'] = 2.0
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.pad_inches'] = 0.05

# Colors
COLOR_M0 = '#0072B2'   # Blue - observed
COLOR_M1 = '#D55E00'   # Orange - missing


# =============================================================================
# ACTIVATION FUNCTION
# =============================================================================

def f_run7(x, m, c):
    """
    Run 7: f(x, m, c) = min(x * m, x) + x

    When m=0 (observed): min(0, x) + x = x (for x>0), 2x (for x<0)
    When m=1 (missing):  min(x, x) + x = 2x
    """
    return np.minimum(x * m, x) + x


# =============================================================================
# PLOTTING
# =============================================================================

def plot_run7_single():
    """
    Single panel showing m-effect for Run 7
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    x = np.linspace(-3, 3, 400)
    m0 = np.zeros_like(x)
    m1 = np.ones_like(x)
    c1 = np.ones_like(x)

    # Compute
    y_m0 = np.clip(f_run7(x, m0, c1), -10, 10)
    y_m1 = np.clip(f_run7(x, m1, c1), -10, 10)

    # Plot
    ax.fill_between(x, y_m0, y_m1, alpha=0.2, color='purple',
                    label='Difference region')
    ax.plot(x, y_m0, color=COLOR_M0, linewidth=2.5, label=r'$m=0$ (observed)')
    ax.plot(x, y_m1, color=COLOR_M1, linewidth=2.5, linestyle='--',
            label=r'$m=1$ (missing)')

    ax.set_xlabel(r'$x$ (input value)')
    ax.set_ylabel(r'$f(x, m, c)$')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-10, 10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, linewidth=0.4)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='none')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('run7_m_effect_single.pdf')
    plt.savefig('run7_m_effect_single.png', dpi=300)
    print("Saved: run7_m_effect_single.pdf/png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    plot_run7_single()
    print("Done!")