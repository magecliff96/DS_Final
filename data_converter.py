import pandas as pd
import numpy as np
from datetime import datetime

# === Config ===
INPUT_CSV = "data.csv"       # Replace with your actual CSV input filename
OUTPUT_INPUT_CSV = "input_i.csv"
OUTPUT_GT_CSV = "ground_truth_i.csv"
interval_size = 5
interval_scale = 16
FIRST_WEEKDAY_OF_YEAR = 2  # 0 = Monday, 6 = Sunday

# === Load data ===
df = pd.read_csv(INPUT_CSV)
df = df.iloc[::10].reset_index(drop=True)

# === Assign unique ID ===
df = df.reset_index(drop=True)
df["record_id"] = df.index  # unique identifier for each row

# === Parse datetime ===
df["date"] = pd.to_datetime(df["date"])
df["UpdateTime"] = pd.to_datetime(df["UpdateTime"])

df["day_of_year"] = df["date"].dt.dayofyear
df["seconds_of_day"] = df["UpdateTime"].dt.hour * 3600 + df["UpdateTime"].dt.minute * 60 + df["UpdateTime"].dt.second

# === Compute adjusted weekday ===
df["weekday"] = ((df["day_of_year"] - 1 + FIRST_WEEKDAY_OF_YEAR) % 7).astype(int)

# === Compute hour ===
df["hour"] = (df["seconds_of_day"] // 3600).astype(int)

# === Cyclical Encoding ===
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# === Extract location keywords ===
def extract_keywords(text):
    return {
        "is_near_mrt": int("MRT" in text.upper() or "捷運" in text),
        "is_near_park": int("PARK" in text.upper() or "公園" in text),
        "is_near_school": int("SCHOOL" in text.upper() or "大學" in text or "學" in text),
    }

keyword_features = df["Town"].apply(extract_keywords).apply(pd.Series)

# === Compute total & usage ratio ===
df["total"] = df["AvailableRentBikes"] + df["AvailableReturnBikes"]
df["usage_ratio"] = df.apply(lambda x: x["AvailableRentBikes"] / x["total"] if x["total"] > 0 else 0, axis=1)

# === Assemble input features ===
input_df = pd.DataFrame({
    "record_id": df["record_id"],
    "latitude": df["latitude"],
    "longitude": df["longitude"],
    "day_of_year": df["day_of_year"],
    "seconds_of_day": df["seconds_of_day"],
    "hour": df["hour"],
    "weekday": df["weekday"],
    "weekday_sin": df["weekday_sin"],
    "weekday_cos": df["weekday_cos"],
    "hour_sin": df["hour_sin"],
    "hour_cos": df["hour_cos"],
    "usage_ratio": df["usage_ratio"],
    "Temp": df["Temp"],
    "HUMD": df["HUMD"],
    "area": df["area"]
}).join(keyword_features)

# One-hot encode area
input_df = pd.get_dummies(input_df, columns=["area"], prefix="district")
district_cols = input_df.columns[input_df.columns.str.startswith('district_')]
input_df[district_cols] = input_df[district_cols].astype(int)

# Save input
input_df.to_csv(OUTPUT_INPUT_CSV, index=False)

# === Create Ground Truth with One-Hot Encoding ===
def one_hot_bucket(value, prefix, interval_size, interval_scale):
    max_range = interval_size * interval_scale
    bins = [f"{i}_{i + interval_size}" for i in range(0, max_range, interval_size)]
    bins.append(f"{max_range}_plus")
    if value >= max_range:
        bucket = len(bins) - 1
    else:
        bucket = value // interval_size
    encoded = [0] * len(bins)
    encoded[bucket] = 1
    return {f"{prefix}_{rng}": e for rng, e in zip(bins, encoded)}

ground_truth = []
for _, row in df.iterrows():
    entry = {"record_id": row["record_id"]}
    entry.update(one_hot_bucket(row["total"], "total", interval_size, interval_scale))
    entry.update(one_hot_bucket(row["AvailableRentBikes"], "rent", interval_size, interval_scale))
    entry.update(one_hot_bucket(row["AvailableReturnBikes"], "return", interval_size, interval_scale))
    ground_truth.append(entry)

ground_truth_df = pd.DataFrame(ground_truth)

# Remove all-zero label columns
id_col = "record_id"
label_cols = ground_truth_df.columns[1:]
non_empty_labels = [col for col in label_cols if ground_truth_df[col].sum() > 0]
filtered_ground_truth_df = ground_truth_df[[id_col] + non_empty_labels]

# Save ground truth
filtered_ground_truth_df.to_csv(OUTPUT_GT_CSV, index=False)

print(f"✅ input.csv and ground_truth.csv saved.")
print(f"Removed {len(label_cols) - len(non_empty_labels)} empty label columns.")

