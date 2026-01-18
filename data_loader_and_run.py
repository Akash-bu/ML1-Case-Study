import numpy as np
import pandas as pd

from model import Model

# Paths
TRAIN_PATH = "data_sets/train_df.csv"
TEST_PATH = "data_sets/test_df.csv"
SUB_PATH = "data_sets/submission.csv"
OUT_PATH = "data_sets/submission.csv"

def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

# Load data
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)

# Define target + feature columns
target_col = "num_errors"
ignore_cols = {target_col, "ID", "index"}

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

feature_cols = sorted(list((train_cols & test_cols) - ignore_cols))

# Prepare numpy arrays
X = train_df[feature_cols].values
y = train_df[target_col].values.astype(int)

# Train/val split
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# Train on train split
model = Model(lr=0.05, epochs=1000, l2=1e-3, use_class_weights=True, verbose=True)
model.fit(X_train, y_train)

# Validate
val_preds = model.predict(X_val)
score = macro_f1(y_val, val_preds, n_classes=len(np.unique(y)))
acc = np.mean(val_preds == y_val)
print(f"Accuracy: {acc:.4f}")
print(f"Validation macro-F1: {score:.4f}")

# Retrain on full data
model.fit(X, y)

# Predict test
X_test = test_df[feature_cols].values
preds = model.predict(X_test)

# # Map predictions to submission by index
# pred_df = pd.DataFrame({
#     "index": test_df["index"].values,
#     "Predicted": preds
# })

# sub_df = sub_df.drop(columns=["Predicted"]).merge(pred_df, on="index", how="left")

# # Save
# sub_df.to_csv(OUT_PATH, index=False)
# print(f"Saved predictions to {OUT_PATH}")
