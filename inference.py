import pandas as pd
import numpy as np
import joblib
import os

# === Config ===
INPUT_FILE = 'input_i.csv'  # New input features (same format as used for training)
MODEL_DIR = 'model'           # Directory containing saved models
OUTPUT_DIR = 'inference_outputs'
batch_size = 1000  # You can adjust this depending on your available memory

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Features ===
features_df = pd.read_csv(INPUT_FILE)
id_col = features_df.columns[0]
X = features_df.iloc[:, 1:]  # Assumes first column is ID

# === Optional: Load Ground Truth Headers for Label Names ===
GROUND_TRUTH_PATH = 'ground_truth.csv'
ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
label_names = list(ground_truth_df.columns[1:])  # Skip the ID column


# === List of Model Files ===
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_model.pkl')]

# === Run Inference for Each Model ===
num_samples = X.shape[0]

for model_file in model_files:
    model_name = model_file.replace('_model.pkl', '')
    print(f"Loading model: {model_name}")

    model = joblib.load(os.path.join(MODEL_DIR, model_file), mmap_mode='r')

    print(f"Running inference in batches of {batch_size}...")
    y_pred_list = []

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_X = X.iloc[start:end]
        batch_pred = model.predict(batch_X)
        y_pred_list.append(batch_pred)

    y_pred = np.vstack(y_pred_list)

    # === Build Prediction DataFrame ===
    if len(label_names) == y_pred.shape[1]:
        pred_columns = [f"{model_name}_{name}" for name in label_names]
    else:
        pred_columns = [f"{model_name}_{i}" for i in range(y_pred.shape[1])]

    pred_df = pd.DataFrame(y_pred, columns=pred_columns)
    pred_df.insert(0, id_col, features_df[id_col].values)

    # === Save Predictions ===
    output_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"Saved predictions to: {output_path}")