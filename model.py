import numpy as np
from base_class import BaseMLModel
from helper_functions import zscore_fit, zscore_transform, macro_f1

class Model(BaseMLModel):
    # Model: One-hidden-layer MLP (ReLU + Softmax)
    def __init__(
        self,
        lr=0.01,
        epochs=300,
        hidden_size=128,
        l2=1e-4,
        use_class_weights=True,
        verbose=False,
        clip=5.0,
        use_internal_norm=True,
        seed=42,
    ):
        super().__init__()
        self.lr = lr
        self.epochs = epochs
        self.hidden_size = hidden_size
        self.l2 = l2
        self.use_class_weights = use_class_weights
        self.verbose = verbose
        self.clip = clip
        self.use_internal_norm = use_internal_norm
        self.seed = seed

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.mean = None
        self.std = None
        self.n_features = None
        self.n_classes = None
        self.loss_history = []

    def _relu(self, Z):
        return np.maximum(0.0, Z)

    def _relu_grad(self, Z):
        return (Z > 0.0).astype(float)

    def _softmax(self, Z):
        Z = Z - np.max(Z, axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def _one_hot(self, y):
        return np.eye(self.n_classes)[y]

    def _forward(self, Xn):
        Z1 = Xn @ self.W1 + self.b1
        A1 = self._relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        probs = self._softmax(Z2)
        return Z1, A1, probs

    def fit(self, X, y, X_val=None, y_val=None, eval_every=50, eval_patience=6):
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))

        if self.use_internal_norm:
            self.mean, self.std = zscore_fit(X)
            Xn = zscore_transform(X, self.mean, self.std, clip=self.clip)
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])
            Xn = X

        Y = self._one_hot(y)
        n = Xn.shape[0]

        if self.use_class_weights:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            # print(f"Num of classes {counts}")
            # weights = len(y) / (self.n_classes * counts)
            weights = (counts.max() / counts) ** 0.85
            # weights = np.sqrt(len(y) / counts)
            weights /= weights.mean()
            sample_w = weights[y]
        else:
            sample_w = np.ones_like(y, dtype=float)

        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.n_features), size=(self.n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_size), size=(self.hidden_size, self.n_classes))
        self.b2 = np.zeros(self.n_classes)

        best_f1 = -1.0
        best_params = None
        no_improve = 0

        for epoch in range(self.epochs):
            Z1, A1, probs = self._forward(Xn)

            # Weighted cross-entropy loss
            loss = -np.sum(sample_w * np.log(probs[np.arange(n), y] + 1e-12)) / n
            loss += 0.5 * self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
            self.loss_history.append(loss)

            # Backprop
            dZ2 = (probs - Y) * (sample_w[:, None] / n)
            dW2 = A1.T @ dZ2 + self.l2 * self.W2
            db2 = np.sum(dZ2, axis=0)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * self._relu_grad(Z1)
            dW1 = Xn.T @ dZ1 + self.l2 * self.W1
            db1 = np.sum(dZ1, axis=0)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

            if self.verbose and (epoch % 100 == 0):
                print(f"epoch {epoch}, loss {loss:.4f}")

            if X_val is not None and y_val is not None and (epoch + 1) % eval_every == 0:
                val_preds = self.predict(X_val)
                f1 = macro_f1(y_val, val_preds, n_classes=self.n_classes)
                if self.verbose:
                    print(f"epoch {epoch+1}, val macro-F1 {f1:.4f}")

                if f1 > best_f1 + 1e-6:
                    best_f1 = f1
                    best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= eval_patience:
                        if self.verbose:
                            print(f"Early stop at epoch {epoch+1}")
                        break

        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2 = best_params
        return self

    def predict_proba(self, X):
        if self.use_internal_norm:
            Xn = zscore_transform(X, self.mean, self.std, clip=self.clip)
        else:
            Xn = X
        _, _, probs = self._forward(Xn)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
