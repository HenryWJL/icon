import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Simulate example data
np.random.seed(42)
models = ['Diff-C', 'Crossway-Diff-C', 'ICon-Diff-C']
data = {
    'Model': ['Diff-C'] * 10 + ['Crossway-Diff-C'] * 10 + ['ICon-Diff-C'] * 10,
    'Success Rate': (
        [0.84, 0.92, 0.84, 0.82, 0.84, 0.8, 0.9, 0.88, 0.86, 0.88] +
        [0.86, 0.86, 0.84, 0.88, 0.84, 0.82, 0.86, 0.88, 0.92, 0.86] +
        [0.88, 0.9, 0.88, 0.88, 0.82, 0.88, 0.9, 0.84, 0.9, 0.8]
    )
}
df = pd.DataFrame(data)

# Aesthetic setup
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Create the violin plot
plt.figure(figsize=(8, 6))
sns.violinplot(
    x='Model',
    y='Success Rate',
    data=df,
    inner='box',        # show box inside like a cylinder
    linewidth=1.2,
    palette='muted',    # or your custom color list
    cut=1               # keep pointed tips
)

# Labels and aesthetics
plt.ylabel("Success Rate")
plt.xlabel("")
plt.title("Success Rate Stability Across Models", pad=15)
plt.tight_layout()
plt.savefig("violin_cylinder_pointed.svg", bbox_inches='tight', pad_inches=0.2)
plt.show()
