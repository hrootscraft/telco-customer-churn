"""A small PyTorch MLP wrapped as a scikit-learn classifier.

Wrapping it keeps the neural net inside the same ``imblearn`` pipeline as every
other model, so it gets the same preprocessing, the same in-fold SMOTE and the
same cross-validation. The net is deliberately small (two hidden layers,
dropout, early stopping on a 10% validation slice): on 5,600 training rows a
bigger net only overfits.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 60,
        batch_size: int = 128,
        patience: int = 8,
        val_fraction: float = 0.1,
        random_state: int = 0,
    ):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state

    def _build(self, n_in: int) -> nn.Module:
        layers: list[nn.Module] = []
        d = n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.classes_ = np.unique(y).astype(int)
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        idx = rng.permutation(len(X))
        n_val = max(1, int(len(X) * self.val_fraction))
        va, tr = idx[:n_val], idx[n_val:]
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr])
        Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])

        self.model_ = self._build(X.shape[1])
        opt = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.BCEWithLogitsLoss()
        best, best_state, bad = np.inf, None, 0
        self.history_: list[tuple[float, float]] = []
        for _ in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(len(Xt))
            tot = 0.0
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.model_(Xt[b]).squeeze(-1), yt[b])
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            self.model_.eval()
            with torch.no_grad():
                vl = loss_fn(self.model_(Xv).squeeze(-1), yv).item()
            self.history_.append((tot / len(Xt), vl))
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            p = torch.sigmoid(self.model_(X).squeeze(-1)).numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        return tags
