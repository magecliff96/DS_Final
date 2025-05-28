import pandas as pd
import numpy as np

# ==== USER INPUT ====
INPUT_CSV = "input.csv"
OUTPUT_CSV = "input_v2.csv"
FIRST_WEEKDAY_OF_YEAR = 2  # 0 = Monday, 6 = Sunday

# ==== LOAD DATA ====
df = pd.read_csv(INPUT_CSV)

# ==== COMPUTE WEEKDAY ====
# Day of year (1–365/366) adjusted by first weekday of the year
df['weekday'] = ((df['day_of_year'] - 1 + FIRST_WEEKDAY_OF_YEAR) % 7).astype(int)

# ==== COMPUTE HOUR OF DAY ====
# Seconds since midnight to hour (0–23)
df['hour'] = (df['seconds_of_day'] // 3600).astype(int)

# ==== CYCLIC ENCODING ====
# Encode weekday cyclically
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

# Encode hour cyclically
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# ==== SAVE OUTPUT ====
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved with added time features to: {OUTPUT_CSV}")


##added time of date and extracted more locational data related to district and whether it is near school, park, or mrt