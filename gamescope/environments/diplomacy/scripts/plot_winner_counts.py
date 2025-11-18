#!/usr/bin/env python3
"""Plot winner counts comparison across probe conditions with confidence intervals."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# File paths
files = {
    "No Probe": Path("/mnt/ssd-1/david/adversarial_interpretability/results/diplomacy/qwen3_4b_eval_verdun_noprobe/diplobench_eval-20251103-230610-4accfc7/metrics/summary.json"),
    "Probe on Target": Path("/mnt/ssd-1/david/adversarial_interpretability/results/diplomacy/qwen3_4b_eval_verdun_probe/diplobench_eval-20251104-052309-4accfc7/metrics/summary.json"),
    "Probe on Target's Enemies": Path("/mnt/ssd-1/david/adversarial_interpretability/results/diplomacy/qwen3_4b_eval_verdun_probeenemy/diplobench_eval-20251104-112218-4accfc7/metrics/summary.json"),
}

# Load data
data = {}
for condition, path in files.items():
    with open(path, 'r') as f:
        data[condition] = json.load(f)

# Extract winner counts
powers = ["EAST", "MIDDLE", "WEST"]
conditions = list(files.keys())

# Prepare data for plotting
plot_data = []
for condition in conditions:
    winner_counts = data[condition]["winner_counts"]
    n_games = data[condition]["num_games"]
    for power in powers:
        count = winner_counts.get(power, 0)
        # Calculate win probability (treating as binomial)
        p = count / n_games
        # Calculate 95% CI using Wilson score interval (more accurate for small samples)
        # Wilson score: (p + z^2/(2n) ± z*sqrt(p(1-p)/n + z^2/(4n^2))) / (1 + z^2/n)
        z = 1.96  # 95% CI
        n = n_games
        denominator = 1 + (z**2 / n)
        center = (p + z**2 / (2 * n)) / denominator
        margin = (z / denominator) * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
        ci_lower = max(0, center - margin)
        ci_upper = min(1, center + margin)
        
        # Convert back to counts for error bars
        count_lower = ci_lower * n
        count_upper = ci_upper * n
        
        plot_data.append({
            "Condition": condition,
            "Power": power,
            "Count": count,
            "CI_Lower": count_lower,
            "CI_Upper": count_upper,
        })

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create bar plot with error bars
x_pos = np.arange(len(powers))
width = 0.25
colors = sns.color_palette("husl", len(conditions))

# Target power is WEST
target_power = "WEST"
target_idx = powers.index(target_power)

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
    
    # Plot bars for each power with different styling based on whether it's the target
    # Set label only once per condition (on first bar)
    for j, power in enumerate(powers):
        label = condition if j == 0 else ""
        if power == target_power:
            # Target bars: darker, thicker border, no alpha reduction
            bars = ax.bar(x[j], heights[j], width, color=colors[i], alpha=1.0, 
                         edgecolor='black', linewidth=2.5, zorder=3, label=label)
            ax.errorbar(x[j], heights[j], yerr=[[yerr_lower[j]], [yerr_upper[j]]], fmt='none', 
                       color='black', capsize=6, capthick=2, linewidth=2, zorder=4)
        else:
            # Non-target bars: lighter, thinner border
            bars = ax.bar(x[j], heights[j], width, color=colors[i], alpha=0.4, 
                         edgecolor='gray', linewidth=1, zorder=2, label=label)
            ax.errorbar(x[j], heights[j], yerr=[[yerr_lower[j]], [yerr_upper[j]]], fmt='none', 
                       color='gray', capsize=4, capthick=1.5, linewidth=1, alpha=0.6, zorder=2)

# Legend is created from bar labels
ax.legend(loc='upper right')

ax.set_xlabel("Power", fontsize=12)
ax.set_ylabel("Winner Count", fontsize=12)
ax.set_title("Winner Counts by Condition (Target: WEST, 95% CI)", fontsize=14, fontweight='bold')
ax.set_xticks(x_pos + width)
power_labels = [f"{p} (Target)" if p == target_power else p for p in powers]
ax.set_xticklabels(power_labels)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("winner_counts_comparison.png", dpi=300, bbox_inches='tight')
print("Plot saved as winner_counts_comparison.png")
plt.show()

