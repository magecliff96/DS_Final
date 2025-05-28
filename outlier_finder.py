import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# === Load and Merge Data ===
input_df = pd.read_csv('input.csv')
ground_truth_df = pd.read_csv('ground_truth.csv')
merged_df = pd.merge(input_df, ground_truth_df, on='sno')

# === Reconstruct Ground Truth Values ===
weight_vector = np.array([15, 45, 90])
merged_df['total_bikes'] = merged_df.filter(like='total_').dot(weight_vector)
merged_df['rent_bikes'] = merged_df.filter(like='rent_').dot(weight_vector)
merged_df['return_bikes'] = merged_df.filter(like='return_').dot(weight_vector)



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
plt.title(f'Outlier Map - Total Bikes\nMean Predicted Total = {y_pred:.1f}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outlier_map_total_bikes.png')
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv('outliers_total_bikes.csv', index=False)




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
plt.savefig('outlier_map_return_bikes.png')
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv('outliers_return_bikes.csv', index=False)



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
plt.savefig('outlier_map_rent_bikes.png')
plt.close()

outlier_rows = merged_df.loc[merged_df['is_high_error'] == 1]
outlier_rows.to_csv('outliers_rent_bikes.csv', index=False)
