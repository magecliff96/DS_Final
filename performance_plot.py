

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import re
import numpy as np

# === Step 1: Read from TXT ===
with open("val_outputs/macro_auc_summary.txt", "r") as f:
    lines = f.readlines()

# === Step 2: Extract relevant lines ===
auc_data_lines = [line.strip() for line in lines if re.match(r'^(total|rent|return)', line)]

# === Step 3: Build DataFrame ===
auc_df = pd.DataFrame([re.split(r'\s{2,}', line) for line in auc_data_lines],
                      columns=['label', 'RandomForest', 'CatBoost', 'XGBoost', 'LGBM'])

# Convert numeric columns to float
for col in ['RandomForest', 'CatBoost', 'XGBoost', 'LGBM']:
    auc_df[col] = auc_df[col].astype(float)

# === Step 4: Extract main class from label ===
auc_df['main_class'] = auc_df['label'].str.extract(r'^(total|rent|return)')

# === Step 5: Compute average AUC per model per main class ===
grouped_auc = auc_df.groupby('main_class')[['RandomForest', 'CatBoost', 'XGBoost', 'LGBM']].mean()

# === Step 6: Transpose and melt to long-form ===
melted_auc = grouped_auc.reset_index().melt(id_vars='main_class', var_name='Model', value_name='AUC')

# Define width and layout
bar_width = 0.2
models = ['RandomForest', 'CatBoost', 'XGBoost', 'LGBM']
main_classes = ['total', 'rent', 'return']
group_width = bar_width * len(models) + bar_width  # Add a spacing bar width for visual separation

# Define base colormaps for each main class
colormaps = {
    'total': cm.Greens,
    'rent': cm.Greens,
    'return': cm.Greens
}

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 6))

# Store legend items
legend_handles = []
legend_labels = []

# Plot each main class group
for i, main_class in enumerate(main_classes):
    subset = melted_auc[melted_auc['main_class'] == main_class]
    cmap = colormaps[main_class]
    colors = cmap(np.linspace(0.4, 0.8, len(models)))
    offset = i * group_width
    for j, (model, color) in enumerate(zip(models, colors)):
        auc = subset[subset['Model'] == model]['AUC'].values[0]
        xpos = offset + j * bar_width
        bar = ax.bar(xpos, auc, width=bar_width, color=color)
        if i == 0:
            legend_handles.append(bar)
            legend_labels.append(model)

# Center x-ticks beneath each group
x_ticks = [i * group_width + (bar_width * (len(models) - 1) / 2) for i in range(len(main_classes))]
ax.set_xticks(x_ticks)
ax.set_xticklabels(main_classes)

# Plot formatting
ax.set_ylim(0.5, 1.0)
ax.set_title("Average AUC per Model Grouped by Main Class")
ax.set_xlabel("Main Class")
ax.set_ylabel("Average AUC")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Legend
ax.legend(legend_handles, legend_labels, title="Model")

plt.tight_layout()
plt.show()
