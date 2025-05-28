

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
import numpy as np  
# === Step 1: Read from TXT ===
with open("macro_auc_summary.txt", "r") as f:
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

# === Step 7: Plot grouped by main class ===
fig, ax = plt.subplots(figsize=(10, 6))

# Set up positions
main_classes = melted_auc['main_class'].unique()
models = ['RandomForest', 'CatBoost', 'XGBoost', 'LGBM']
bar_width = 0.2
x = range(len(main_classes))

# Define shades of green
greens = cm.Greens(np.linspace(0.4, 0.8, len(models)))  # darker to lighter shades

# Plot
for i, (model, color) in enumerate(zip(models, greens)):
    aucs = melted_auc[melted_auc['Model'] == model]['AUC']
    positions = [pos + i * bar_width for pos in x]
    ax.bar(positions, aucs, width=bar_width, label=model, color=color)

# X-axis formatting
mid_positions = [pos + (bar_width * (len(models)-1)) / 2 for pos in x]
ax.set_xticks(mid_positions)
ax.set_xticklabels(main_classes)
ax.set_ylim(0.5, 1.0)

# Labels and legend
ax.set_title("Average AUC per Model Grouped by Main Class")
ax.set_xlabel("Main Class")
ax.set_ylabel("Average AUC")
ax.legend(title="Model")
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
