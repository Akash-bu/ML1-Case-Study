import numpy as np
import pandas as pd
from helper_functions import macro_f1, plot_loss_curve, zscore_fit, zscore_transform, pca_fit, pca_transform
from model import Model
import argparse 

parser = argparse.ArgumentParser() 

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--l2', type = float, default=1e-3)
parser.add_argument('--use_class_weights', type = bool, default=True)
parser.add_argument('--verbose', type = bool, default=False)
parser.add_argument('--hidden_size', type = int, default=64)

args = parser.parse_args()

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

# Train/val split
rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
split = int(0.8 * len(X))
train_idx, val_idx = idx[:split], idx[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

x_mean, x_std = zscore_fit(X_train)
X_train_n = zscore_transform(X_train, x_mean, x_std, clip=5.0)
X_val_n = zscore_transform(X_val, x_mean, x_std, clip=5.0)

n_components = min(100, X_train_n.shape[1])
pca_mean, pca_components = pca_fit(X_train_n, n_components=n_components)
X_train_pca = pca_transform(X_train_n, pca_mean, pca_components)
X_val_pca = pca_transform(X_val_n, pca_mean, pca_components)

# Train on train split
model = Model(
    lr=args.lr,
    epochs=args.epochs,
    hidden_size=args.hidden_size,
    l2=args.l2,
    use_class_weights=args.use_class_weights,
    verbose=args.verbose,
    use_internal_norm=False,
)
model.fit(X_train_pca, y_train, X_val_pca, y_val)

val_preds = model.predict(X_val_pca)
score = macro_f1(y_val, val_preds, n_classes=len(np.unique(y)))
acc = np.mean(val_preds == y_val)
print(f"Accuracy: {acc:.4f}")
print(f"Validation macro-F1: {score:.4f}")

# Retrain on full data
x_mean, x_std = zscore_fit(X)
X_n = zscore_transform(X, x_mean, x_std, clip=5.0)
n_components = min(100, X_n.shape[1])
pca_mean, pca_components = pca_fit(X_n, n_components=n_components)
X_pca = pca_transform(X_n, pca_mean, pca_components)

model = Model(
    lr=args.lr,
    epochs=args.epochs,
    hidden_size=args.hidden_size,
    l2=args.l2,
    use_class_weights=args.use_class_weights,
    verbose=args.verbose,
    use_internal_norm=False,
)
model.fit(X_pca, y)

# Predict test
X_test = test_df[feature_cols].values
X_test_n = zscore_transform(X_test, x_mean, x_std, clip=5.0)
X_test_pca = pca_transform(X_test_n, pca_mean, pca_components)
preds = model.predict(X_test_pca)

# Map predictions to submission by index
# pred_df = pd.DataFrame({
#     "index": test_df["index"].values,
#     "Predicted": preds
# })

# sub_df = sub_df.drop(columns=["Predicted"]).merge(pred_df, on="index", how="left")

# sub_df.to_csv(OUT_PATH, index=False)
# print(f"Saved predictions to {OUT_PATH}")
