import random

import numpy as np
import pandas as pd
from helper_functions import macro_f1, zscore_fit, zscore_transform, pca_fit, pca_transform
from model import Model
import argparse

def stratified_split(y, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train_idx, val_idx = [], []

    for c in np.unique(y):
        idxs = np.where(y == c)[0]
        rng.shuffle(idxs)
        k = min(360, len(idxs) - 1)
        val_idx.extend(idxs[:k].tolist())
        train_idx.extend(idxs[k:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.array(train_idx), np.array(val_idx)


def sample_params():
    return {
        "hidden_size": random.choice([128, 192, 256, 384]),
        "lr": 10 ** np.random.uniform(np.log10(3e-4), np.log10(3e-3)),
        "l2": 10 ** np.random.uniform(np.log10(1e-6), np.log10(3e-3)),
        "dropout": random.choice([0.0, 0.1, 0.2, 0.3]),
        "batch_size": random.choice([64, 128, 256]),
        "leaky_relu": random.choice([0.01, 0.02, 0.05]),
        "class_weight_alpha": random.choice([0.5, 1.0, 0.7, 0.8]),
    }


BEST_PARAMS = {
    "hidden_size": 256,
    "lr": 0.00285923304567468,
    "l2": 3.250491815920611e-05,
    "dropout": 0.0,
    "batch_size": 128,
    "leaky_relu": 0.01,
    "class_weight_alpha": 0.5,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--l2', type = float, default=1e-4)
    parser.add_argument('--use_class_weights', type = bool, default=True)
    parser.add_argument('--verbose', type = bool, default=False)
    parser.add_argument('--hidden_size', type = int, default=256)
    parser.add_argument('--clip', type = float, default=5.0)
    parser.add_argument('--use_internal_norm', type = bool, default=False)
    parser.add_argument('--seed', type = int, default=42)
    parser.add_argument('--batch_size', type = int, default=256)
    parser.add_argument('--beta1', type = float, default=0.9)
    parser.add_argument('--beta2', type = float, default=0.999)
    parser.add_argument('--eps', type = float, default=1e-8)
    parser.add_argument('--dropout', type = float, default=0.1)
    parser.add_argument('--leaky_relu', type = float, default=0.01)
    parser.add_argument('--class_weight_alpha', type = float, default=1.0)
    parser.add_argument('--run_final', action='store_true')
    return parser.parse_args()


def load_data():
    TRAIN_PATH = "data_sets/train_df.csv"
    TEST_PATH = "data_sets/test_df.csv"
    SUB_PATH = "data_sets/submission.csv"

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
    X_test = test_df[feature_cols].values
    return X, y, X_test, sub_df


def main(args):
    X, y, _, _ = load_data()

    # Train/val split
    train_idx, val_idx = stratified_split(y)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    x_mean, x_std = zscore_fit(X_train)
    X_train_n = zscore_transform(X_train, x_mean, x_std, clip=args.clip)
    X_val_n = zscore_transform(X_val, x_mean, x_std, clip=args.clip)

    n_components = min(100, X_train_n.shape[1])
    pca_mean, pca_components = pca_fit(X_train_n, n_components=n_components)
    X_train_pca = pca_transform(X_train_n, pca_mean, pca_components)
    X_val_pca = pca_transform(X_val_n, pca_mean, pca_components)

    # Train on train split
    params = sample_params()
    model = Model(
        lr=params["lr"],
        epochs=args.epochs,
        hidden_size=params["hidden_size"],
        l2=params["l2"],
        dropout=params["dropout"],
        batch_size=params["batch_size"],
        leaky_relu=params["leaky_relu"],
        class_weight_alpha=params["class_weight_alpha"],
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
    with open("random_search_results.txt", "a") as f:
        f.write(f" F1={score:.4f} | {params}\n")


def final_train_and_predict(args, params):
    X, y, X_test, sub_df = load_data()
    x_mean, x_std = zscore_fit(X)
    X_n = zscore_transform(X, x_mean, x_std, clip=args.clip)
    X_test_n = zscore_transform(X_test, x_mean, x_std, clip=args.clip)

    n_components = min(100, X_n.shape[1])
    pca_mean, pca_components = pca_fit(X_n, n_components=n_components)
    X_pca = pca_transform(X_n, pca_mean, pca_components)
    X_test_pca = pca_transform(X_test_n, pca_mean, pca_components)

    model = Model(
        lr=params["lr"],
        epochs=args.epochs,
        hidden_size=params["hidden_size"],
        l2=params["l2"],
        dropout=params["dropout"],
        batch_size=params["batch_size"],
        leaky_relu=params["leaky_relu"],
        class_weight_alpha=params["class_weight_alpha"],
        use_class_weights=args.use_class_weights,
        verbose=args.verbose,
        use_internal_norm=False,
    )
    model.fit(X_pca, y)

    preds = model.predict(X_test_pca)
    sub_df = sub_df.copy()
    sub_df["Predicted"] = preds.astype(int)
    out_path = "data_sets/submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.run_final:
        final_train_and_predict(args, BEST_PARAMS)
    else:
        for i in range(20):
            main(args)

# # Retrain on full data
# x_mean, x_std = zscore_fit(X)
# X_n = zscore_transform(X, x_mean, x_std, clip=5.0)
# n_components = min(100, X_n.shape[1])
# pca_mean, pca_components = pca_fit(X_n, n_components=n_components)
# X_pca = pca_transform(X_n, pca_mean, pca_components)
#
# model = Model(
#     lr=args.lr,
#     epochs=args.epochs,
#     hidden_size=args.hidden_size,
#     l2=args.l2,
#     use_class_weights=args.use_class_weights,
#     verbose=args.verbose,
#     use_internal_norm=False,
# )
# model.fit(X_pca, y)
#
# # Predict test
# X_test = test_df[feature_cols].values
# X_test_n = zscore_transform(X_test, x_mean, x_std, clip=5.0)
# X_test_pca = pca_transform(X_test_n, pca_mean, pca_components)
# preds = model.predict(X_test_pca)

