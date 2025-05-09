import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np

# === Load Data ===
random_seed = 42
features_path = 'input.csv'  # your actual features file
labels_path = 'ground_truth.csv'  # your ground truth file

# Read CSVs
features_df = pd.read_csv(features_path)
labels_df = pd.read_csv(labels_path)

# Merge on ID column
id_col = features_df.columns[0]
merged_df = pd.merge(features_df, labels_df, on=id_col)

# Extract features and labels
X = merged_df.iloc[:, 1:features_df.shape[1]]  # Features (exclude ID)
y = merged_df.iloc[:, features_df.shape[1]:]   # Labels (after feature columns)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed
)

# === Train Model ===
rf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
multi_rf = MultiOutputClassifier(rf)
multi_rf.fit(X_train, y_train)

# === Predict Probabilities ===
y_prob = multi_rf.predict_proba(X_test)
# For each label, extract prob of class 1
y_prob_matrix = np.column_stack([prob[:, 1] for prob in y_prob])

# === Predict Labels and Save ===
y_pred = multi_rf.predict(X_test)
pred_df = pd.DataFrame(y_pred, columns=y.columns)
pred_df.insert(0, id_col, features_df.loc[y_test.index, id_col].values)
pred_df.to_csv("predictions.csv", index=False)

# === Compute AUC ===
print("AUC scores per label:")
for i, label in enumerate(y.columns):
    try:
        auc = roc_auc_score(y_test.iloc[:, i], y_prob_matrix[:, i])
        print(f"{label}: {auc:.4f}")
    except ValueError:
        print(f"{label}: Cannot compute AUC (only one class present in y_test)")
