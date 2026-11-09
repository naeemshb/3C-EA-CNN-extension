"""
Create a professional publication-quality figure for the evolved
three-channel activation function - FIXED LAYOUT

f(x, m, c) = ((min(((x - m) + x), sqrt(x)) * x) / -0.1)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Professional academic styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'text.usetex': False,
})

fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
ax.set_xlim(-0.5, 7.5)
ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.axis('off')

# GECCO-style colors
color_operator = '#5B9BD5'      # Soft blue for operators
color_input = '#ED7D31'         # Soft orange for inputs
color_constant = '#C0C0C0'      # Light grey for constants
color_root = '#FFFFFF'          # White for root
edge_color = '#404040'          # Dark grey for edges
border_color = '#2E2E2E'        # Darker border

# FIXED layout - shifted right to keep everything in view
# Tree: ((min(((x - m) + x), sqrt(x)) * x) / -0.1)
positions = {
    # Level 0 (root) - centered
    'div': (3.5, 5.5),
    # Level 1
    'mult': (2.2, 4.3),
    'const': (5.2, 4.3),
    # Level 2
    'min': (1.2, 3.1),
    'x_mult': (3.2, 3.1),
    # Level 3
    'add': (0.5, 1.9),
    'sqrt': (2.0, 1.9),
    # Level 4
    'sub': (0.5, 0.7),
    'x_add': (1.5, 0.7),
    'x_sqrt': (2.8, 0.7),
    # Level 5 - SHIFTED RIGHT to stay in bounds
    'x_sub': (0.0, -0.2),
    'm': (1.0, -0.2),
}

def draw_node(ax, pos, label, node_type='operator', fontsize=10, radius=0.28):
    """Draw a professional node."""
    x, y = pos

    if node_type == 'root':
        circle = plt.Circle((x, y), radius, facecolor=color_root,
                           edgecolor=border_color, linewidth=1.8, zorder=3)
        text_color = '#1a1a1a'
    elif node_type == 'operator':
        circle = plt.Circle((x, y), radius, facecolor=color_operator,
                           edgecolor=border_color, linewidth=1.2, zorder=3)
        text_color = 'white'
    elif node_type == 'input':
        circle = plt.Circle((x, y), radius, facecolor=color_input,
                           edgecolor=border_color, linewidth=1.2, zorder=3)
        text_color = 'white'
    elif node_type == 'constant':
        circle = plt.Circle((x, y), radius, facecolor=color_constant,
                           edgecolor=border_color, linewidth=1.2, zorder=3)
        text_color = '#1a1a1a'

    ax.add_patch(circle)

    # Use slightly smaller font for longer labels
    if len(str(label)) > 2:
        fontsize = fontsize - 2

    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=text_color, zorder=4,
            fontfamily='serif')

def draw_edge(ax, start_pos, end_pos, radius=0.28):
    """Draw a smooth arrow."""
    x1, y1 = start_pos
    x2, y2 = end_pos

    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)

    if dist < 0.01:
        return

    # Adjust start and end for node radius
    x1_adj = x1 + radius * dx / dist
    y1_adj = y1 + radius * dy / dist
    x2_adj = x2 - radius * dx / dist
    y2_adj = y2 - radius * dy / dist

    ax.annotate('', xy=(x2_adj, y2_adj), xytext=(x1_adj, y1_adj),
                arrowprops=dict(
                    arrowstyle='-|>',
                    color=edge_color,
                    lw=1.2,
                    shrinkA=0,
                    shrinkB=0,
                    mutation_scale=10,
                ),
                zorder=1)

# Draw all edges first (behind nodes)
edges = [
    ('div', 'mult'), ('div', 'const'),
    ('mult', 'min'), ('mult', 'x_mult'),
    ('min', 'add'), ('min', 'sqrt'),
    ('add', 'sub'), ('add', 'x_add'),
    ('sqrt', 'x_sqrt'),
    ('sub', 'x_sub'), ('sub', 'm'),
]

for start, end in edges:
    draw_edge(ax, positions[start], positions[end])

# Draw all nodes
nodes = [
    ('div', '÷', 'root', 12),
    ('mult', '×', 'operator', 12),
    ('const', '−0.1', 'constant', 8),
    ('min', 'min', 'operator', 8),
    ('x_mult', 'x', 'input', 11),
    ('add', '+', 'operator', 12),
    ('sqrt', '√', 'operator', 11),
    ('sub', '−', 'operator', 12),
    ('x_add', 'x', 'input', 11),
    ('x_sqrt', 'x', 'input', 11),
    ('x_sub', 'x', 'input', 11),
    ('m', 'm', 'input', 11),
]

for name, label, ntype, fsize in nodes:
    draw_node(ax, positions[name], label, node_type=ntype, fontsize=fsize)

# Add equation above - clean mathematical typography
eq_text = r'$f = \frac{\min((x - m) + x,\, \sqrt{x}) \cdot x}{-0.1}$'
ax.text(3.5, 6.3, eq_text, ha='center', va='bottom', fontsize=12,
        fontfamily='serif')

# Professional legend - horizontal at bottom
legend_elements = [
    mpatches.Patch(facecolor=color_root, edgecolor=border_color,
                   linewidth=1.2, label='Root'),
    mpatches.Patch(facecolor=color_operator, edgecolor=border_color,
                   linewidth=1, label='Operators'),
    mpatches.Patch(facecolor=color_input, edgecolor=border_color,
                   linewidth=1, label='Inputs (x, m, c)'),
    mpatches.Patch(facecolor=color_constant, edgecolor=border_color,
                   linewidth=1, label='Constants'),
]

legend = ax.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=8,
    frameon=False,
    handlelength=1.2,
    handleheight=1.2,
    columnspacing=1.0,
)
plt.show()
plt.tight_layout(pad=0.5)
# plt.savefig('/mnt/user-data/outputs/evolved_tree_fixed.pdf',
#             dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# plt.savefig('/mnt/user-data/outputs/evolved_tree_fixed.png',
#             dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
# print("Fixed figure saved!")