import numpy as np
from base_class import BaseMLModel 
from helper_functions import zscore_fit, zscore_transform, zscore_fit_transform

class Model(BaseMLModel):

    def __init__(self, lr=0.05, epochs = 500, l2 = 0.0, use_class_weights = True, verbose = False):
        super().__init__() 
        self.lr = lr 
        self.epochs = epochs 
        self.l2 = l2 
        self.use_class_weights = use_class_weights 
        self.verbose = verbose 

        self.W = None 
        self.mean = None 
        self.std = None 
        self.n_features = None 
        self.n_classes = None 

    def _softmax(self, Z):
        Z = Z - np.max(Z, axis = 1, keepdims=True) #subtract every element of the row by max element in the row for numerical stability
        expZ = np.exp(Z) 
        return expZ / np.sum(expZ, axis=1, keepdims=True) 

    def _add_bias(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):

        self.n_features = X.shape[1] 
        self.n_classes = len(np.unique(y))

        #Normalization
        self.mean, self.std = zscore_fit(X)
        Xn = zscore_transform(X, self.mean, self.std)
        Xn = self._add_bias(Xn)

        Y = np.eye(self.n_classes)[y]
        
        #class imbalance handling
        if self.use_class_weights:
            counts = np.bincount(y, minlength=self.n_classes).astype(float)
            weights = (len(y) / (self.n_classes * counts)) 
            sample_w = weights[y] 
        else:
            sample_w = np.ones_like(y, dtype=float) 

        self.W = np.zeros((Xn.shape[1], self.n_classes))

        for epoch in range(self.epochs):
            scores = Xn @ self.W 
            probs = self._softmax(scores) 

            diff = (probs - Y) * sample_w[:, None]
            grad = (Xn.T @ diff) / Xn.shape[0] 

            if self.l2 > 0.0:
                reg = self.l2 * self.W 
                reg[-1, :] = 0.0 
                grad += reg 

            self.W -= self.lr * grad 

            if self.verbose and (epoch % 100 == 0):
                loss = -np.sum(sample_w * np.log(probs[np.arange(len(y)), y] + 1e-12)) / len(y) #weighted loss
                print(f"epoch {epoch}, loss {loss: .4f}")
        
        return self 

    def predict(self, X):

        Xn = zscore_transform(X, self.mean, self.std) 
        Xn = self._add_bias(Xn)
        probs = self._softmax(Xn @ self.W) 
        return np.argmax(probs, axis = 1)


        
        