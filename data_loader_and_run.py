import numpy as np
import pandas as pd
from helper_functions import macro_f1, plot_loss_curve

from model import Model

TRAIN_PATH = "data_sets/train_df.csv"
TEST_PATH = "data_sets/test_df.csv"
SUB_PATH = "data_sets/submission.csv"
OUT_PATH = "data_sets/submission.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sub_df = pd.read_csv(SUB_PATH)

# Define target + feature columns
target_col = "num_errors"
ignore_cols = {target_col, "ID", "index"}

train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

feature_cols = sorted(list((train_cols & test_cols) - ignore_cols))

X = train_df[feature_cols].values
y = train_df[target_col].values.astype(int)

"""
# Train/val split
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# Train on train split
model = Model(lr=0.01, epochs=1000, l2=0.001, use_class_weights=True, verbose=True)
model.fit(X_train, y_train, X_val=X_val, y_val=y_val, eval_every=50, eval_patience=3)

# plot_loss_curve(model.loss_history, title="Train Loss")

# print("Train label distribution:", np.bincount(y_train))
# print("Val predictions distribution:", np.bincount(y_val, minlength=4))

val_preds = model.predict(X_val)
score = macro_f1(y_val, val_preds, n_classes=len(np.unique(y)))
acc = np.mean(val_preds == y_val)
print(f"Accuracy: {acc:.4f}")
print(f"Validation macro-F1: {score:.4f}")
"""

def stratified_kfold_indices(y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, k)
        for i in range(k):
            folds[i].extend(splits[i])
    return [np.array(f, dtype=int) for f in folds]

k = 5
folds = stratified_kfold_indices(y, k=k, seed=42)

accs = []
f1s = []

for i in range(k):
    val_idx = folds[i]
    train_idx = np.hstack([folds[j] for j in range(k) if j != i])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = Model(lr=0.01, epochs=1000, l2=0.001, use_class_weights=True, verbose=False)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    acc = np.mean(val_preds == y_val)
    f1 = macro_f1(y_val, val_preds, n_classes=len(np.unique(y)))

    accs.append(acc)
    f1s.append(f1)

    print(f"Fold {i+1}/{k} | acc: {acc:.4f} | macro-F1: {f1:.4f}")

print(f"CV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"CV macro-F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# Retrain on full data
model.fit(X, y)

# Predict test
X_test = test_df[feature_cols].values
preds = model.predict(X_test)

# Map predictions to submission by index
# pred_df = pd.DataFrame({
#     "index": test_df["index"].values,
#     "Predicted": preds
# })

# sub_df = sub_df.drop(columns=["Predicted"]).merge(pred_df, on="index", how="left")

# sub_df.to_csv(OUT_PATH, index=False)
# print(f"Saved predictions to {OUT_PATH}")
