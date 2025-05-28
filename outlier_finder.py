import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import re
import os


def extract_weights_from_columns(columns):
    weights = []
    for i, col in enumerate(columns):
        match = re.search(r'(\d+)_([0-9]+|plus)', col)
        if match:
            low_str, high_str = match.groups()
            low = int(low_str)

            if high_str == "plus":
                if i > 0:
                    # Estimate last bin width using previous bin
                    prev_low, prev_high = map(int, re.search(r'(\d+)_([0-9]+)', columns[i - 1]).groups())
                    bin_width = prev_high - prev_low
                    high = low + bin_width  # or use low + bin_width * 1.5 for conservative
                else:
                    high = low + 10  # default fallback
            else:
                high = int(high_str)

            midpoint = (low + high) / 2
            weights.append(midpoint)
        else:
            raise ValueError(f"Could not extract bin range from column: {col}")
    return np.array(weights)



# === Load and Merge Data ===
input_df = pd.read_csv('input.csv')
ground_truth_df = pd.read_csv('ground_truth.csv')
merged_df = pd.merge(input_df, ground_truth_df, on='record_id')
OUTPUT_DIR = "outlier"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# === Reconstruct Ground Truth Values ===

# === Reconstruct Ground Truth Values ===
# Select columns dynamically and compute weights
total_cols = [col for col in merged_df.columns if col.startswith("total_") and "_" in col]
rent_cols  = [col for col in merged_df.columns if col.startswith("rent_") and "_" in col]
return_cols = [col for col in merged_df.columns if col.startswith("return_") and "_" in col]

# Sort the columns to keep bin order consistent
total_cols.sort()
rent_cols.sort()
return_cols.sort()

# Extract weights and compute dot product
merged_df['total_bikes']  = merged_df[total_cols].dot(extract_weights_from_columns(total_cols))
merged_df['rent_bikes']   = merged_df[rent_cols].dot(extract_weights_from_columns(rent_cols))
merged_df['return_bikes'] = merged_df[return_cols].dot(extract_weights_from_columns(return_cols))



target = 'total_bikes'
y = merged_df[target]
y_pred = y.mean()
rel_error = (y - y_pred) / y_pred * 100
z_scores = ((y - y_pred) ** 2).pow(0.5)
z_scores = (z_scores - z_scores.mean()) / z_scores.std()
merged_df['is_high_error'] = (z_scores > 2).astype(int)
merged_df['relative_error'] = rel_error




# Plot
plt.figure(figsize=(10, 6))
normal_points = merged_df['is_high_error'] == 0
outliers = merged_df['is_high_error'] == 1
color_values = merged_df.loc[outliers, 'relative_error']
vlim = max(abs(color_values.min()), abs(color_values.max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)

plt.scatter(merged_df.loc[normal_points, 'longitude'], merged_df.loc[normal_points, 'latitude'],
            facecolors='white', edgecolors='lightgray', alpha=0.5, label='Normal')

scatter = plt.scatter(merged_df.loc[outliers, 'longitude'], merged_df.loc[outliers, 'latitude'],
                      c=color_values, cmap='RdBu_r', norm=norm, s=40, alpha=0.9, label='Outlier (gradient)')

cbar = plt.colorbar(scatter)
cbar.set_label('Relative Error in Total Bikes (%)', rotation=270, labelpad=25)
ticks = np.linspace(-vlim, vlim, 7)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.0f}%' for t in ticks])

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Outlier Map - Total Bikes\nMean Total = {y_pred:.1f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_map_total_bikes.png'))
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv(os.path.join(OUTPUT_DIR, 'outliers_total_bikes.csv'), index=False)





target = 'return_bikes'
y = merged_df[target]
y_pred = y.mean()
rel_error = (y - y_pred) / y_pred * 100
z_scores = ((y - y_pred) ** 2).pow(0.5)
z_scores = (z_scores - z_scores.mean()) / z_scores.std()
merged_df['is_high_error'] = (z_scores > 2).astype(int)
merged_df['relative_error'] = rel_error

plt.figure(figsize=(10, 6))
normal_points = merged_df['is_high_error'] == 0
outliers = merged_df['is_high_error'] == 1
color_values = merged_df.loc[outliers, 'relative_error']
vlim = max(abs(color_values.min()), abs(color_values.max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)

plt.scatter(merged_df.loc[normal_points, 'longitude'], merged_df.loc[normal_points, 'latitude'],
            facecolors='white', edgecolors='lightgray', alpha=0.5, label='Normal')

scatter = plt.scatter(merged_df.loc[outliers, 'longitude'], merged_df.loc[outliers, 'latitude'],
                      c=color_values, cmap='RdBu_r', norm=norm, s=40, alpha=0.9, label='Outlier (gradient)')

cbar = plt.colorbar(scatter)
cbar.set_label('Relative Error in Return Bikes (%)', rotation=270, labelpad=25)
ticks = np.linspace(-vlim, vlim, 7)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.0f}%' for t in ticks])

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Outlier Map - Return Bikes\nMean Return = {y_pred:.1f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_map_return_bikes.png'))
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv(os.path.join(OUTPUT_DIR, 'outliers_return_bikes.csv'), index=False)



target = 'rent_bikes'
y = merged_df[target]
y_pred = y.mean()
rel_error = (y - y_pred) / y_pred * 100
z_scores = ((y - y_pred) ** 2).pow(0.5)
z_scores = (z_scores - z_scores.mean()) / z_scores.std()
merged_df['is_high_error'] = (z_scores > 2).astype(int)
merged_df['relative_error'] = rel_error

plt.figure(figsize=(10, 6))
normal_points = merged_df['is_high_error'] == 0
outliers = merged_df['is_high_error'] == 1
color_values = merged_df.loc[outliers, 'relative_error']
vlim = max(abs(color_values.min()), abs(color_values.max()))
norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)

plt.scatter(merged_df.loc[normal_points, 'longitude'], merged_df.loc[normal_points, 'latitude'],
            facecolors='white', edgecolors='lightgray', alpha=0.5, label='Normal')

scatter = plt.scatter(merged_df.loc[outliers, 'longitude'], merged_df.loc[outliers, 'latitude'],
                      c=color_values, cmap='RdBu_r', norm=norm, s=40, alpha=0.9, label='Outlier (gradient)')

cbar = plt.colorbar(scatter)
cbar.set_label('Relative Error in Rent Bikes (%)', rotation=270, labelpad=25)
ticks = np.linspace(-vlim, vlim, 7)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{t:.0f}%' for t in ticks])

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'Outlier Map - Rent Bikes\nMean Rent = {y_pred:.1f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'outlier_map_rent_bikes.png'))
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv(os.path.join(OUTPUT_DIR, 'outliers_rent_bikes.csv'), index=False)