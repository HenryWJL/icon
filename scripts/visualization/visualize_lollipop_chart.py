import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Seaborn style
sns.set_theme(style='whitegrid', context='talk')

plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': False,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Data
models = ['Diff-C', 'Crossway-Diff-C', 'ICon-Diff-C']
avg_success = [0.866, 0.862, 0.882]
max_success = [0.92, 0.92, 0.92]

x = np.arange(len(models))
avg_colors = ['#FAD7AC', '#D0CEE2', '#D5E8D4']
max_colors = ['#FDC378', '#A69AC5', '#A8D5B5']

fig, ax = plt.subplots()

# Plot lollipop lines and dots
for i in range(len(models)):
    ax.vlines(x[i], avg_success[i], max_success[i], color='gray', linestyle='dashed', linewidth=1.5)
    ax.plot(x[i], avg_success[i], 'o', color=avg_colors[i], markersize=10)
    ax.plot(x[i], max_success[i], 'o', color=max_colors[i], markersize=10)

    delta = (max_success[i] - avg_success[i]) / max_success[i]
    ax.text(x[i] + 0.22, avg_success[i], f'-{delta:.3f}%', ha='center', va='bottom', fontsize=12, color='black')

# Remove x-axis ticks and labels
ax.set_xticks([])
ax.set_xticklabels([])

# Y-axis label
ax.set_ylabel('Success Rate', fontsize=15)

# Grid and aesthetics
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Title
plt.title('Open Door', fontsize=16)

# Build legend
legend_elements = [
    Patch(facecolor=max_colors[0], edgecolor='none', label='Diff-C (Max)'),
    Patch(facecolor=avg_colors[0], edgecolor='none', label='Diff-C (Avg)'),
    Patch(facecolor=max_colors[1], edgecolor='none', label='Crossway-Diff-C (Max)'),
    Patch(facecolor=avg_colors[1], edgecolor='none', label='Crossway-Diff-C (Avg)'),
    Patch(facecolor=max_colors[2], edgecolor='none', label='ICon-Diff-C (Max)'),
    Patch(facecolor=avg_colors[2], edgecolor='none', label='ICon-Diff-C (Avg)'),
]

ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.35),
    ncol=3,
    frameon=True,
    fontsize=11,
    borderpad=0.8,
    framealpha=0.95
)

plt.tight_layout()
plt.savefig("training_stability.svg", bbox_inches='tight', pad_inches=0.1)
plt.show()