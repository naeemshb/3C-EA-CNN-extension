"""
Ablation Study Visualization for GECCO Paper
=============================================
Clean bar charts showing Test Accuracy and F1 Score for ablation variants.
Side-by-side version with mean and std displayed on bars.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set up publication-quality style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Ablation results (excluding Fixed ReLU and Value Only)
variants = [
    'Full System\nf(x,m,c)',
    'No Confidence\nf(x,m)',
    'No Missing Flag\nf(x,c)',
    'No Channel\nPropagation',
]

# Data from your results
test_acc_mean = [0.4349, 0.4070, 0.3860, 0.3814]
test_acc_std = [0.1163, 0.0591, 0.0853, 0.0542]

f1_mean = [0.2290, 0.2234, 0.2199, 0.1998]
f1_std = [0.0723, 0.0750, 0.0628, 0.0659]

x = np.arange(len(variants))
width = 0.35

# Side-by-side bars (Test Acc and F1)
fig, ax = plt.subplots(figsize=(8, 5))

bars1 = ax.bar(x - width/2, test_acc_mean, width,
               label='Test Accuracy', color='#2E86AB',
               edgecolor='white', linewidth=0.5)

bars2 = ax.bar(x + width/2, f1_mean, width,
               label='F1 Score', color='#E8630A',
               edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score')
ax.set_xlabel('')
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(variants)
ax.legend(loc='upper right')

ax.set_ylim(0.1, 0.60)
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Add value labels on bars (mean on top, std below)
for bar, val, std in zip(bars1, test_acc_mean, test_acc_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E86AB')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'±{std:.3f}', ha='center', va='bottom', fontsize=8, color='#2E86AB')

for bar, val, std in zip(bars2, f1_mean, f1_std):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E8630A')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'±{std:.3f}', ha='center', va='bottom', fontsize=8, color='#E8630A')

plt.tight_layout()
plt.savefig('ablation_sidebyside_clean.png', bbox_inches='tight', facecolor='white')
plt.savefig('ablation_sidebyside_clean.pdf', bbox_inches='tight', facecolor='white')
plt.show()