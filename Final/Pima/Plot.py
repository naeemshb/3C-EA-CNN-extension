import matplotlib.pyplot as plt
import numpy as np


def plot_bar_chart_missing_rate(missing_rates, test_acc_data, dataset_name="Sonar"):
    """
    Plot Test Accuracy as grouped bar chart for different methods across missing rates.

    Parameters:
    -----------
    missing_rates : list
        List of missing rate percentages (e.g., [10, 20, 30, 40, 50])
    test_acc_data : dict
        Dictionary with method names as keys and lists of (mean, std) tuples as values
    dataset_name : str
        Name of the dataset for the title
    """

    # Use professional color palette with better contrast
    colors = {
        'Evolved': '#1f77b4',  # Professional blue
        'ReLU': '#ff7f0e',  # Orange
        'Swish': '#2ca02c',  # Green
        'LeakyReLU': '#d62728',  # Red
        'ELU': '#9467bd'  # Purple
    }

    # Prepare data
    methods = list(test_acc_data.keys())
    n_methods = len(methods)
    n_rates = len(missing_rates)

    # Set up bar positions
    bar_width = 0.15
    index = np.arange(n_rates)

    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot bars for each method with edge color for better definition
    for i, method in enumerate(methods):
        means = [val[0] for val in test_acc_data[method]]

        offset = (i - n_methods / 2 + 0.5) * bar_width
        positions = index + offset

        bars = ax.bar(positions, means, bar_width,
                      label=method,
                      color=colors.get(method, None),
                      edgecolor='black',
                      linewidth=1.2,
                      alpha=0.85)

    # Professional styling
    ax.set_xlabel('Missing Data Rate (%)', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_ylabel('Test Accuracy', fontsize=16, fontweight='bold', labelpad=10)

    # X-axis
    ax.set_xticks(index)
    ax.set_xticklabels([f'{rate}%' for rate in missing_rates], fontsize=14, fontweight='500')

    # Y-axis
    ax.tick_params(axis='y', labelsize=13)

    # Legend with better positioning
    ax.legend(fontsize=13, loc='upper right', framealpha=0.95,
              edgecolor='black', fancybox=False, shadow=False,
              frameon=True, ncol=1, columnspacing=1)

    # Grid - horizontal only, more subtle
    ax.grid(True, alpha=0.2, linestyle='-', axis='y', linewidth=0.8)
    ax.set_axisbelow(True)  # Grid behind bars

    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    # Set y-axis limits for better visualization with some padding
    all_means = [val[0] for vals in test_acc_data.values() for val in vals]
    y_min = min(all_means) - 0.02
    y_max = max(all_means) + 0.02
    ax.set_ylim(y_min, y_max)

    # Add subtle spine styling
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    return fig


# ============================================================================
# DATA FROM YOUR EXPERIMENTS
# ============================================================================

# Missing rates tested
missing_rates = [10, 20, 30, 40, 50]

# Test Accuracy: (mean, std) for each missing rate
test_acc_data = {
    'Evolved': [
        (0.7029, 0.0277),  # 10%
        (0.7042, 0.0260),  # 20%
        (0.6974, 0.0325),  # 30%
        (0.6956, 0.0294),  # 40%
        (0.6828, 0.0386),  # 50%
    ],
    'ReLU': [
        (0.6883, 0.0359),  # 10%
        (0.7013, 0.0203),  # 20%
        (0.6792, 0.0336),  # 30%
        (0.6799, 0.0293),  # 40%
        (0.6649, 0.0195),  # 50%
    ],
    'Swish': [
        (0.6903, 0.0250),  # 10%
        (0.7005, 0.0175),  # 20%
        (0.6942, 0.0155),  # 30%
        (0.6929, 0.0241),  # 40%
        (0.6747, 0.0363),  # 50%
    ],
    'LeakyReLU': [
        (0.7015, 0.0254),  # 10%
        (0.7016, 0.0185),  # 20%
        (0.6964, 0.0257),  # 30%
        (0.6799, 0.0268),  # 40%
        (0.6714, 0.0245),  # 50%
    ],
    'ELU': [
        (0.7014, 0.0254),  # 10%
        (0.7022, 0.0148),  # 20%
        (0.6922, 0.0243),  # 30%
        (0.6773, 0.0207),  # 40%
        (0.6718, 0.0219),  # 50%
    ],
}

# ============================================================================
# CREATE AND SAVE PLOTS
# ============================================================================

if __name__ == "__main__":
    # Check if we have data for all missing rates
    expected_length = len(missing_rates)
    actual_length = len(test_acc_data['Evolved'])

    if actual_length < expected_length:
        print(f"⚠️  WARNING: Only have data for {actual_length} out of {expected_length} missing rates")
        print(f"   Using missing rates: {missing_rates[:actual_length]}")
        missing_rates = missing_rates[:actual_length]

    # Create the bar chart
    fig = plot_bar_chart_missing_rate(missing_rates, test_acc_data, dataset_name="Sonar")

    # Save the figure in the same directory as the script
    fig.savefig('performance_vs_missing_rate_bar.png',
                dpi=300, bbox_inches='tight')
    print("✅ Bar chart saved to: performance_vs_missing_rate_bar.png")

    # Also save as PDF for publication quality
    fig.savefig('performance_vs_missing_rate_bar.pdf',
                bbox_inches='tight')
    print("✅ Bar chart saved to: performance_vs_missing_rate_bar.pdf")

    plt.show()

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE - TEST ACCURACY")
    print("=" * 90)
    print(f"{'Method':<12}", end="")
    for rate in missing_rates:
        print(f"{rate}% Missing{'':<8}", end="")
    print()
    print("-" * 90)

    for method in test_acc_data.keys():
        print(f"{method:<12}", end="")
        for mean, std in test_acc_data[method]:
            print(f"{mean:.4f}±{std:.4f}  ", end="")
        print()

    print("=" * 90)