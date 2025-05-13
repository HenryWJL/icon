import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Use seaborn style for clean aesthetics
sns.set_theme(style='whitegrid', context='talk')

# Optional: LaTeX-style fonts (requires LaTeX installed)
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Data
models = ['ICon', 'W/o Multi-Level Contrast', 'W/o FPS']
success_rates = [0.300, 0.260, 0.220]
std = [0.043, 0.028, 0.085]

# Plot
fig, ax = plt.subplots()

# Remove x-axis labels
ax.set_xticks([])

# Bar plot with error bars
bars = ax.bar(
    models,
    success_rates,
    yerr=std,
    capsize=5,
    width=0.4,
    color=['#FAD7AC', '#D0CEE2', '#D5E8D4'],
    edgecolor='black',
    linewidth=0.8
)

# Optional: Remove x-axis ticks if using legend labels instead
# ax.set_xticks([])

# Add values on top of bars
for i, (bar, value) in enumerate(zip(bars, success_rates)):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        height + std[i] + 0.01,
        f'{value:.3f} ± {std[i]:.3f}',
        ha='center',
        va='bottom',
        fontsize=10
    )

# Legend: bottom, horizontal, with box and smaller font
ax.legend(
    handles=bars,
    labels=models,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=True,
    fontsize=12,
    borderpad=0.8,
    framealpha=0.9
)

# Y-axis label
ax.set_ylabel('Success Rate', fontsize=15)

# Dashed gridlines behind bars
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Title
plt.title('Open Box', fontsize=16)

# Layout adjustment
plt.tight_layout()
plt.savefig("ablation.svg", bbox_inches='tight', pad_inches=0)
plt.show()
