import os
import json
import pandas as pd
from datetime import datetime

# === Config ===
folder_path = 'inputs'  # folder containing JSON files
interval_size = 10
interval_scale = 8
# === Load and Aggregate JSON Files ===
all_data = []
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_data.extend(data)  # accumulate all entries

# === Convert infoDate and infoTime to numeric features ===
def convert_datetime(info_date_str, info_time_str):
    date_obj = datetime.strptime(info_date_str, "%Y-%m-%d")
    time_obj = datetime.strptime(info_time_str, "%Y-%m-%d %H:%M:%S")
    day_of_year = date_obj.timetuple().tm_yday
    seconds_of_day = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
    hour = time_obj.hour
    weekday = date_obj.weekday()
    return day_of_year, seconds_of_day, hour, weekday

# === Extract Metadata Features ===
def extract_keywords(text):
    keywords = {
        "is_near_mrt": int("MRT" in text.upper() or "捷運" in text),
        "is_near_park": int("PARK" in text.upper() or "公園" in text),
        "is_near_school": int("SCHOOL" in text.upper() or "大學" in text or "學" in text),
    }
    return keywords

# === Create Input CSV ===
input_records = []
for d in all_data:
    day_of_year, seconds_of_day, hour, weekday = convert_datetime(d["infoDate"], d["infoTime"])
    usage_ratio = d["available_rent_bikes"] / d["total"] if d["total"] > 0 else 0
    keywords = extract_keywords(d["snaen"])
    input_records.append({
        "sno": d["sno"],
        "latitude": d["latitude"],
        "longitude": d["longitude"],
        "day_of_year": day_of_year,
        "seconds_of_day": seconds_of_day,
        "hour": hour,
        "weekday": weekday,
        "usage_ratio": usage_ratio,
        "sarea": d["sarea"],
        **keywords
    })

input_df = pd.DataFrame(input_records)

# One-hot encode 'sarea'
input_df = pd.get_dummies(input_df, columns=["sarea"], prefix="district")

# Ensure all one-hot columns are integers (not bools)
district_cols = input_df.columns[input_df.columns.str.startswith('district_')]
input_df[district_cols] = input_df[district_cols].astype(int)

input_df.to_csv("input.csv", index=False)

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
for d in all_data:
    row = {"sno": d["sno"]}
    row.update(one_hot_bucket(d["total"], "total", interval_size, interval_scale))
    row.update(one_hot_bucket(d["available_rent_bikes"], "rent", interval_size, interval_scale))
    row.update(one_hot_bucket(d["available_return_bikes"], "return", interval_size, interval_scale))
    ground_truth.append(row)

ground_truth_df = pd.DataFrame(ground_truth)

# === Remove columns with only 0s ===
id_col = ground_truth_df.columns[0]
label_cols = ground_truth_df.columns[1:]
non_empty_labels = [col for col in label_cols if ground_truth_df[col].sum() > 0]
filtered_ground_truth_df = ground_truth_df[[id_col] + non_empty_labels]

# === Save Ground Truth CSV ===
filtered_ground_truth_df.to_csv("ground_truth.csv", index=False)

print(f"✅ input.csv and ground_truth.csv generated from folder: {folder_path}")
print(f"Removed {len(label_cols) - len(non_empty_labels)} empty label columns.")

