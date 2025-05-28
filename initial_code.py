import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
import joblib
warnings.filterwarnings("ignore")
import os

# === Load Data ===
random_seed = 42
features_path = 'data/input.csv'
labels_path = 'data/ground_truth.csv'

features_df = pd.read_csv(features_path)
labels_df = pd.read_csv(labels_path)

id_col = features_df.columns[0]
merged_df = pd.merge(features_df, labels_df, on=id_col)

X = merged_df.iloc[:, 1:features_df.shape[1]]
y = merged_df.iloc[:, features_df.shape[1]:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

output_dir = "val_outputs"
os.makedirs(output_dir, exist_ok=True)

# === Define Models ===
models = {
    "RandomForest": RandomForestClassifier(
    n_estimators=10,        # ↓ from 100 → 10
    max_depth=10,           # limit tree growth
    max_features='sqrt',    # fewer features per split
    min_samples_leaf=5,     # regularize leaf size
    random_state=random_seed),
    "CatBoost": CatBoostClassifier(verbose=0, random_seed=random_seed),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_seed),
    "LGBM": LGBMClassifier(random_state=random_seed)
}

auc_results = pd.DataFrame(index=y.columns)

# === Train and Evaluate Each Model ===
for name, base_model in models.items():
    print(f"\nTraining {name}...")
    model = MultiOutputClassifier(base_model)
    model.fit(X_train, y_train)

    # Save the model
    model_path = os.path.join(output_dir, f"{name}_model.pkl")
    joblib.dump(model, model_path)

    # Predict probabilities and compute AUC scores
    y_prob = model.predict_proba(X_test)
    y_prob_matrix = np.column_stack([prob[:, 1] for prob in y_prob])

    auc_scores = []
    for i in range(y.shape[1]):
        try:
            auc = roc_auc_score(y_test.iloc[:, i], y_prob_matrix[:, i])
        except ValueError:
            auc = np.nan
        auc_scores.append(auc)
    auc_results[name] = auc_scores

    # Save predictions
    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame(y_pred, columns=y.columns)
    pred_df.insert(0, id_col, merged_df.loc[y_test.index, id_col].values)
    pred_path = os.path.join(output_dir, f"{name}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

# === Display and Save AUCs ===
auc_results_df = pd.DataFrame(auc_results, index=y.columns).round(4)
macro_auc = auc_results_df.mean().round(4)

print("\nAUC Scores Per Label:")
print(auc_results_df)

print("\nMacro AUC Summary:")
print(macro_auc)

# Save AUC result files
auc_results_df.to_csv(os.path.join(output_dir, "auc_scores_per_label.csv"))

with open(os.path.join(output_dir, "macro_auc_summary.txt"), "w") as f:
    f.write("Macro AUC Summary (average AUC per model):\n")
    f.write(macro_auc.to_string())
    f.write("\n\nAUC Scores Per Label:\n")
    f.write(auc_results_df.to_string())