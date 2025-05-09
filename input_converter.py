import json
import pandas as pd
from datetime import datetime

# === Config ===
input_name = 'input.json'
interval_size = 30

# === Load JSON ===
with open(input_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Convert infoDate and infoTime to numeric features ===
def convert_datetime(info_date_str, info_time_str):
    date_obj = datetime.strptime(info_date_str, "%Y-%m-%d")
    time_obj = datetime.strptime(info_time_str, "%Y-%m-%d %H:%M:%S")
    day_of_year = date_obj.timetuple().tm_yday
    seconds_of_day = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    return day_of_year, seconds_of_day

# === Create Input CSV ===
input_records = []
for d in data:
    day_of_year, seconds_of_day = convert_datetime(d["infoDate"], d["infoTime"])
    input_records.append({
        "sno": d["sno"],
        "latitude": d["latitude"],
        "longitude": d["longitude"],
        "day_of_year": day_of_year,
        "seconds_of_day": seconds_of_day
    })

input_df = pd.DataFrame(input_records)
input_df.to_csv("input.csv", index=False)

# === Create Ground Truth with One-Hot Encoding ===
def one_hot_bucket(value, prefix, interval_size):
    # Define cutoffs up to but not including the "plus" bucket
    bins = [f"{i}_{i + interval_size}" for i in range(0, interval_size * 2, interval_size)]
    bins.append(f"{interval_size * 2}_plus")  # Final bucket is "greater than or equal to" this

    # Determine which bucket the value falls into
    if value >= interval_size * 2:
        bucket = len(bins) - 1
    else:
        bucket = value // interval_size

    # One-hot encode
    encoded = [0] * len(bins)
    encoded[bucket] = 1
    return {f"{prefix}_{rng}": e for rng, e in zip(bins, encoded)}

ground_truth = []
for d in data:
    row = {"sno": d["sno"]}
    row.update(one_hot_bucket(d["total"], "total", interval_size))
    row.update(one_hot_bucket(d["available_rent_bikes"], "rent", interval_size))
    row.update(one_hot_bucket(d["available_return_bikes"], "return", interval_size))
    ground_truth.append(row)

ground_truth_df = pd.DataFrame(ground_truth)

# === Remove columns with only 0s ===
id_col = ground_truth_df.columns[0]
label_cols = ground_truth_df.columns[1:]
non_empty_labels = [col for col in label_cols if ground_truth_df[col].sum() > 0]
filtered_ground_truth_df = ground_truth_df[[id_col] + non_empty_labels]

# === Save Ground Truth CSV ===
filtered_ground_truth_df.to_csv("ground_truth.csv", index=False)

print(f"✅ input.csv and ground_truth.csv generated.")
print(f"Removed {len(label_cols) - len(non_empty_labels)} empty label columns.")
