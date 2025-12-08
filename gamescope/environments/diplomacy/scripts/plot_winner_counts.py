#!/usr/bin/env python3
"""Plot winner counts comparison across probe conditions with confidence intervals.

Example usage:
    python -m gamescope.environments.diplomacy.scripts.plot_winner_counts \
        --input "No Probe:results/diplomacy/.../summary.json" \
        --input "Probe on Target:results/diplomacy/.../summary.json" \
        --output winner_counts.png \
        --target_power WEST
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def parse_input_spec(spec: str) -> tuple[str, Path]:
    """Parse 'Label:path' input specification."""
    if ':' not in spec:
        raise ValueError(f"Input spec must be 'Label:path', got: {spec}")
    label, path = spec.split(':', 1)
    return label.strip(), Path(path.strip())


def load_summary_data(files: dict[str, Path]) -> dict:
    """Load summary JSON data from file paths."""
    data = {}
    for condition, path in files.items():
        with open(path, 'r') as f:
            data[condition] = json.load(f)
    return data


def compute_wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Args:
        count: Number of successes
        n: Total number of trials
        z: Z-score for confidence level (default 1.96 for 95% CI)

    Returns:
        (lower_bound, upper_bound) as counts (not proportions)
    """
    if n == 0:
        return 0, 0

    p = count / n
    denominator = 1 + (z**2 / n)
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)

    return ci_lower * n, ci_upper * n


def plot_winner_counts(
    data: dict,
    powers: list[str],
    target_power: str,
    output_path: Path,
    title: str | None = None,
    show: bool = False,
):
    """Create bar plot comparing winner counts across conditions."""
    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    conditions = list(data.keys())

    # Prepare data for plotting
    plot_data = []
    for condition in conditions:
        winner_counts = data[condition]["winner_counts"]
        n_games = data[condition]["num_games"]
        for power in powers:
            count = winner_counts.get(power, 0)
            ci_lower, ci_upper = compute_wilson_ci(count, n_games)
            plot_data.append({
                "Condition": condition,
                "Power": power,
                "Count": count,
                "CI_Lower": ci_lower,
                "CI_Upper": ci_upper,
            })

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(powers))
    width = 0.25
    colors = sns.color_palette("husl", len(conditions))

    for i, condition in enumerate(conditions):
        x = x_pos + i * width
        heights = []
        yerr_lower = []
        yerr_upper = []

        for j, power in enumerate(powers):
            entry = next(d for d in plot_data if d["Condition"] == condition and d["Power"] == power)
            heights.append(entry["Count"])
            yerr_lower.append(entry["Count"] - entry["CI_Lower"])
            yerr_upper.append(entry["CI_Upper"] - entry["Count"])

        for j, power in enumerate(powers):
            label = condition if j == 0 else ""
            if power == target_power:
                ax.bar(x[j], heights[j], width, color=colors[i], alpha=1.0,
                       edgecolor='black', linewidth=2.5, zorder=3, label=label)
                ax.errorbar(x[j], heights[j], yerr=[[yerr_lower[j]], [yerr_upper[j]]], fmt='none',
                           color='black', capsize=6, capthick=2, linewidth=2, zorder=4)
            else:
                ax.bar(x[j], heights[j], width, color=colors[i], alpha=0.4,
                       edgecolor='gray', linewidth=1, zorder=2, label=label)
                ax.errorbar(x[j], heights[j], yerr=[[yerr_lower[j]], [yerr_upper[j]]], fmt='none',
                           color='gray', capsize=4, capthick=1.5, linewidth=1, alpha=0.6, zorder=2)

    ax.legend(loc='upper right')
    ax.set_xlabel("Power", fontsize=12)
    ax.set_ylabel("Winner Count", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f"Winner Counts by Condition (Target: {target_power}, 95% CI)", fontsize=14, fontweight='bold')

    ax.set_xticks(x_pos + width)
    power_labels = [f"{p} (Target)" if p == target_power else p for p in powers]
    ax.set_xticklabels(power_labels)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_path}")

    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot winner counts comparison across probe conditions"
    )
    parser.add_argument(
        "--input", "-i",
        action="append",
        required=True,
        help="Input specification as 'Label:path/to/summary.json'. Can be repeated."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="winner_counts_comparison.png",
        help="Output file path (default: winner_counts_comparison.png)"
    )
    parser.add_argument(
        "--powers",
        nargs="+",
        default=["EAST", "MIDDLE", "WEST"],
        help="List of powers to include (default: EAST MIDDLE WEST)"
    )
    parser.add_argument(
        "--target_power",
        type=str,
        default="WEST",
        help="Target power to highlight (default: WEST)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom plot title"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window"
    )

    args = parser.parse_args()

    # Parse input specifications
    files = {}
    for spec in args.input:
        label, path = parse_input_spec(spec)
        if not path.exists():
            raise FileNotFoundError(f"Summary file not found: {path}")
        files[label] = path

    # Load data
    data = load_summary_data(files)

    # Create plot
    plot_winner_counts(
        data=data,
        powers=args.powers,
        target_power=args.target_power,
        output_path=Path(args.output),
        title=args.title,
        show=args.show,
    )


if __name__ == "__main__":
    main()
