import numpy as np
import matplotlib.pyplot as plt

# Data
models = ['ICon', 'W/o Multi-Level Contrast', 'W/o FPS']
success_rates = [0.30, 0.26, 0.22]

# Plot
fig, ax = plt.subplots()
bars = ax.bar(models, success_rates, width=0.4, color=['#FAD7AC', '#D0CEE2', '#D5E8D4'])  # ['#FDC378', '#FFA533', '#FF8633']

# Remove x-axis labels
ax.set_xticks([])

# Add values on top of bars
for bar, value in zip(bars, success_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
            f'{value:.2f}', ha='center', va='bottom', fontsize=10)

# Assign labels for legend
for bar, label in zip(bars, models):
    bar.set_label(label)

# Legend: bottom, horizontal, with box and smaller font
ax.legend(
    loc='lower center',
    bbox_to_anchor=(0.5, -0.2),
    ncol=3,
    frameon=True,
    fontsize=12,
    borderpad=0.8,
    framealpha=0.9
)

# Y-axis label
ax.set_ylabel('Success Rate', fontsize=12)

# Dashed gridlines behind bars
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# Remove top and right borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# No title
plt.title('Open Box', fontsize=16)

# Layout adjustment
plt.tight_layout()
plt.savefig(f"ablation.svg")
plt.show()
