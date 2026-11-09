# activation_viz.py - Professional Version
# Enhanced visualization for evolved activation functions

import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_activation_shape(tree, evaluate_tree_fn, title="", ax=None):
    """
    Plot f(x, m, c) for one activation tree over x in [-5, 5],
    with m = 0 (observed), c = 1 (full confidence).

    Parameters
    ----------
    tree : dict
        Activation tree.
    evaluate_tree_fn : callable
        Function like evaluate_tree(tree, x, m, c) -> Tensor.
        We pass it from the main file to avoid circular imports.
    title : str
        Title for the subplot.
    ax : matplotlib axes, optional
        If provided, draw on this axes; otherwise create a new figure.
    """
    x = torch.linspace(-5, 5, 400)
    m = torch.zeros_like(x)
    c = torch.ones_like(x)

    with torch.no_grad():
        y = evaluate_tree_fn(tree, x, m, c)
        y = torch.clamp(y, -10, 10)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    # Professional color scheme - publication-quality blue
    ax.plot(x.numpy(), y.numpy(), linewidth=2.5, color='#2E86AB', zorder=3)

    # Reference lines with subtle styling
    ax.axhline(0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.4, zorder=1)
    ax.axvline(0, color='#333333', linewidth=0.8, linestyle='-', alpha=0.4, zorder=1)

    # Professional grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, zorder=0)

    # Labels and title with improved typography
    ax.set_xlabel('Input Value (x)', fontsize=11, fontweight='500')
    ax.set_ylabel('Activation Output', fontsize=11, fontweight='500')
    ax.set_title(title, fontsize=12, fontweight='600', pad=10)

    # Improve tick labels
    ax.tick_params(labelsize=9, length=4, width=0.8)

    # Add subtle box around plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#666666')


def plot_generational_shapes(
        gen_best_log,
        evaluate_tree_fn,
        selected_generations=None,
        save_path=None,
        show=True,
):
    """
    Plot activation shapes of best individual per generation with professional styling.

    Parameters
    ----------
    gen_best_log : list of dict
        Output from run_evolution_with_nn; must contain
        row["generation"] and row["tree"] for each generation.
    evaluate_tree_fn : callable
        Function that evaluates a tree, passed from main file.
    selected_generations : list or set of int, optional
        If given, only plot these generations (e.g. [1,2,4,6,8,10]).
        If None, plot all generations.
    save_path : str, optional
        If not None, save figure to this path.
    show : bool
        If True, call plt.show().
    """
    # Filter generations if user provided a subset
    if selected_generations is not None:
        selected_generations = set(selected_generations)
        rows = [row for row in gen_best_log
                if row["generation"] in selected_generations]
    else:
        rows = gen_best_log

    if not rows:
        print("No generations to plot.")
        return

    num_plots = len(rows)
    cols = 3
    rows_count = (num_plots + cols - 1) // cols

    # Create figure with better sizing and spacing
    fig, axes = plt.subplots(rows_count, cols, figsize=(14, 4.5 * rows_count),
                             constrained_layout=True)
    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette for variety
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

    for i, row in enumerate(rows):
        tree = row["tree"]
        gen = row["generation"]
        fitness = row.get("fitness", None)

        # Evaluate tree
        x = torch.linspace(-5, 5, 400)
        m = torch.zeros_like(x)
        c = torch.ones_like(x)
        with torch.no_grad():
            y = evaluate_tree_fn(tree, x, m, c)
            y = torch.clamp(y, -10, 10)

        # Plot with cycling colors
        color = colors[i % len(colors)]
        axes[i].plot(x.numpy(), y.numpy(), linewidth=2.5, color=color, zorder=3)

        # Reference lines
        axes[i].axhline(0, color='#333333', linewidth=0.8, alpha=0.4, zorder=1)
        axes[i].axvline(0, color='#333333', linewidth=0.8, alpha=0.4, zorder=1)

        # Grid
        axes[i].grid(True, alpha=0.15, linestyle='--', linewidth=0.5, zorder=0)

        # Title with fitness if available
        if fitness is not None:
            title_text = f"Generation {gen}\n(Fitness: {fitness:.4f})"
        else:
            title_text = f"Generation {gen}"
        axes[i].set_title(title_text, fontsize=11, fontweight='600', pad=8)

        # Labels
        axes[i].set_xlabel('Input (x)', fontsize=10, fontweight='500')
        axes[i].set_ylabel('Output', fontsize=10, fontweight='500')

        # Tick params
        axes[i].tick_params(labelsize=8, length=3, width=0.8)

        # Spine styling
        for spine in axes[i].spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#666666')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add overall title
    fig.suptitle('Evolution of Activation Functions Across Generations',
                 fontsize=14, fontweight='bold', y=0.995)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white')
        print(f"Saved activation evolution figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_run_winners(winner_trees, evaluate_tree_fn, save_path=None, show=True):
    """
    Plot the winner activation from each run with professional styling.

    Parameters
    ----------
    winner_trees : list of trees
        One tree per run
    evaluate_tree_fn : callable
        Function to evaluate trees
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display the plot
    """
    num = len(winner_trees)
    cols = min(3, num)
    rows = (num + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows),
                             constrained_layout=True)
    if num == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#6A994E', '#BC4B51', '#3C6E71']

    # Plot evolved activations
    for i, tree in enumerate(winner_trees):
        x = torch.linspace(-5, 5, 400)
        m = torch.zeros_like(x)
        c = torch.ones_like(x)

        with torch.no_grad():
            y = evaluate_tree_fn(tree, x, m, c)
            y = torch.clamp(y, -10, 10)

        color = colors[i % len(colors)]
        axes[i].plot(x.numpy(), y.numpy(), linewidth=2.5, color=color, zorder=3)

        # Reference lines
        axes[i].axhline(0, color='#333333', linewidth=0.8, alpha=0.4, zorder=1)
        axes[i].axvline(0, color='#333333', linewidth=0.8, alpha=0.4, zorder=1)

        # Grid
        axes[i].grid(True, alpha=0.15, linestyle='--', linewidth=0.5, zorder=0)

        axes[i].set_title(f"Run {i + 1}", fontsize=11, fontweight='600', pad=8)
        axes[i].set_xlabel('Input (x)', fontsize=10, fontweight='500')
        axes[i].set_ylabel('Output', fontsize=10, fontweight='500')
        axes[i].tick_params(labelsize=8, length=3, width=0.8)

        for spine in axes[i].spines.values():
            spine.set_linewidth(0.8)
            spine.set_color('#666666')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add overall title
    fig.suptitle('',
                 fontsize=14, fontweight='bold', y=0.995)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor='white')
        print(f"Saved run-winner figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)