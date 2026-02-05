import numpy as np
from base_class import BaseMLModel
from helper_functions import zscore_fit, zscore_transform, macro_f1

class Model(BaseMLModel):
    # Model: One-hidden-layer MLP (ReLU + Softmax)
    def __init__(
        self,
        lr=0.001,
        epochs=200,
        hidden_size=256,
        l2=1e-4,
        use_class_weights=True,
        verbose=False,
        clip=5.0,
        use_internal_norm=True,
        seed=42,
        batch_size=256,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        dropout=0.1,
        leaky_relu=0.01,
        class_weight_alpha=1.0,
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
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.dropout = dropout
        self.leaky_relu = leaky_relu
        self.class_weight_alpha = class_weight_alpha

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

    def _forward(self, Xn, training=False, rng=None):
        Z1 = Xn @ self.W1 + self.b1
        A1 = self._act(Z1)
        if training and self.dropout > 0.0:
            mask = (rng.random(A1.shape) >= self.dropout).astype(A1.dtype)
            A1 = A1 * mask / (1.0 - self.dropout)
        else:
            mask = None
        Z2 = A1 @ self.W2 + self.b2
        probs = self._softmax(Z2)
        return Z1, A1, probs, mask

    def _act(self, Z):
        a = self.leaky_relu
        return np.where(Z > 0, Z, a * Z)

    def _act_grad(self, Z):
        a = self.leaky_relu
        return np.where(Z > 0, 1.0, a)

    def fit(self, X, y, X_val=None, y_val=None, eval_every=1, eval_patience=20):
        self.n_features = X.shape[1]
        self.n_classes = int(np.max(y)) + 1

        if self.use_internal_norm:
            self.mean, self.std = zscore_fit(X)
            Xn = zscore_transform(X, self.mean, self.std, clip=self.clip)
        else:
            self.mean = np.zeros(X.shape[1])
            self.std = np.ones(X.shape[1])
            Xn = X

        n = Xn.shape[0]
        rng = np.random.default_rng(self.seed)

        if self.use_class_weights:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            weights = counts.sum() / (self.n_classes * np.maximum(counts, 1.0))
            weights = weights ** self.class_weight_alpha

            weights = weights / weights.mean()
            sample_w_all = weights[y]
        else:
            sample_w_all = np.ones_like(y, dtype=float)

        self.W1 = rng.normal(0.0, np.sqrt(2.0 / self.n_features), size=(self.n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_size), size=(self.hidden_size, self.n_classes))
        self.b2 = np.zeros(self.n_classes)

        # Adam state
        mW1 = np.zeros_like(self.W1)
        vW1 = np.zeros_like(self.W1)
        mb1 = np.zeros_like(self.b1)
        vb1 = np.zeros_like(self.b1)
        mW2 = np.zeros_like(self.W2)
        vW2 = np.zeros_like(self.W2)
        mb2 = np.zeros_like(self.b2)
        vb2 = np.zeros_like(self.b2)

        best_f1, best_params, no_improve = -1.0, None, 0
        t = 0

        for epoch in range(self.epochs):
            perm = rng.permutation(n)
            Xn_shuf = Xn[perm]
            y_shuf = y[perm]
            sw_shuf = sample_w_all[perm]

            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                end = min(n, start + self.batch_size)
                xb = Xn_shuf[start:end]
                yb = y_shuf[start:end]
                swb = sw_shuf[start:end]
                nb = xb.shape[0]

                # forward
                Z1, A1, probs, mask = self._forward(xb, training=True, rng=rng)

                # loss
                loss = -np.sum(swb * np.log(probs[np.arange(nb), yb] + 1e-12)) / nb
                loss += 0.5 * self.l2 * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
                epoch_loss += loss * nb

                # backprop
                Yb = np.eye(self.n_classes)[yb]
                dZ2 = (probs - Yb) * (swb[:, None] / nb)
                dW2 = A1.T @ dZ2 + self.l2 * self.W2
                db2 = np.sum(dZ2, axis=0)

                dA1 = dZ2 @ self.W2.T
                if mask is not None:
                    dA1 = dA1 * mask / (1.0 - self.dropout)

                dZ1 = dA1 * self._act_grad(Z1)
                dW1 = xb.T @ dZ1 + self.l2 * self.W1
                db1 = np.sum(dZ1, axis=0)

                # Adam update
                t += 1
                for P, dP, mP, vP in [
                    (self.W1, dW1, mW1, vW1),
                    (self.b1, db1, mb1, vb1),
                    (self.W2, dW2, mW2, vW2),
                    (self.b2, db2, mb2, vb2),
                ]:
                    mP[:] = self.beta1 * mP + (1 - self.beta1) * dP
                    vP[:] = self.beta2 * vP + (1 - self.beta2) * (dP * dP)
                    mhat = mP / (1 - self.beta1 ** t)
                    vhat = vP / (1 - self.beta2 ** t)
                    P[:] = P - self.lr * mhat / (np.sqrt(vhat) + self.eps)

            print(f"Epoch: {epoch}"
                  f"{'':6s} Loss: {epoch_loss / n:.4f}", end='\n')

            self.loss_history.append(epoch_loss / n)

            if X_val is not None and y_val is not None and (epoch + 1) % eval_every == 0:
                val_preds = self.predict(X_val)
                f1 = macro_f1(y_val, val_preds, n_classes=self.n_classes)

                if f1 > best_f1 + 1e-5:
                    best_f1 = f1
                    best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= eval_patience:
                        break

        if best_params is not None:
            self.W1, self.b1, self.W2, self.b2 = best_params
        return self

    def predict_proba(self, X):
        if self.use_internal_norm:
            Xn = zscore_transform(X, self.mean, self.std, clip=self.clip)
        else:
            Xn = X
        _, _, probs, _ = self._forward(Xn, training=False, rng=None)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
