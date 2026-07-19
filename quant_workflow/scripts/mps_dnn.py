"""MPS-only tabular neural network model for the robust CSI800 workflow.

This module is intentionally optional. The classical robust run must remain
usable without PyTorch; a run that selects this model must fail clearly when
Apple MPS is unavailable instead of silently using the CPU.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
import pandas as pd


class MPSDNNUnavailable(RuntimeError):
    """Raised when the requested MPS runtime cannot be used."""


def _torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise MPSDNNUnavailable("torch is not installed in the MPS environment") from exc
    return torch, nn


def require_mps() -> str:
    """Verify that the requested device is actually available."""
    torch, _ = _torch()
    if not torch.backends.mps.is_built():
        raise MPSDNNUnavailable("the installed PyTorch build has no MPS backend")
    if not torch.backends.mps.is_available():
        raise MPSDNNUnavailable("Apple MPS is not available on this machine")
    return "mps"


def _feature_and_label(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("dataset preparation must return a DataFrame")
    if isinstance(frame.columns, pd.MultiIndex):
        groups = set(frame.columns.get_level_values(0))
        if not {"feature", "label"}.issubset(groups):
            raise ValueError("prepared frame must contain feature and label groups")
        features = frame.xs("feature", axis=1, level=0, drop_level=True)
        labels = frame.xs("label", axis=1, level=0, drop_level=True)
    else:
        raise ValueError("prepared frame must use feature and label column groups")
    if isinstance(labels, pd.DataFrame):
        if labels.shape[1] != 1:
            raise ValueError("label group must contain exactly one column")
        label = labels.iloc[:, 0]
    else:
        label = labels
    features = features.apply(pd.to_numeric, errors="coerce")
    label = pd.to_numeric(label, errors="coerce")
    valid = np.isfinite(features.to_numpy(dtype=np.float32)).all(axis=1) & np.isfinite(label.to_numpy(dtype=np.float32))
    if not bool(valid.any()):
        raise ValueError("MPS DNN received no finite training rows")
    return features.loc[valid], label.loc[valid]


def _build_network(input_dim: int, hidden_layers: tuple[int, ...], dropout: float):
    """Build a module-level Sequential so cloudpickle can persist the model."""
    _, nn = _torch()
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_layers:
        layers.extend(
            [
                nn.Linear(current, width),
                nn.LayerNorm(width),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        )
        current = width
    layers.append(nn.Linear(current, 1))
    return nn.Sequential(*layers)


class MPSDNNModel:
    """A Qlib-compatible MLP that is deliberately MPS-only."""

    def __init__(
        self,
        lr: float = 0.001,
        max_steps: int = 300,
        batch_size: int = 2048,
        early_stop_rounds: int = 50,
        eval_steps: int = 20,
        hidden_layers: tuple[int, ...] | list[int] = (128, 64),
        dropout: float = 0.05,
        weight_decay: float = 0.0,
        seed: int = 42,
        device: str = "mps",
    ):
        if str(device).lower() != "mps":
            raise MPSDNNUnavailable("MPSDNNModel does not permit CPU or CUDA fallback")
        self.device_name = require_mps()
        self.lr = float(lr)
        self.max_steps = int(max_steps)
        self.batch_size = int(batch_size)
        self.early_stop_rounds = int(early_stop_rounds)
        self.eval_steps = int(eval_steps)
        self.hidden_layers = tuple(int(width) for width in hidden_layers)
        self.dropout = float(dropout)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        if not self.hidden_layers or any(width <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must contain positive widths")
        if self.max_steps <= 0 or self.batch_size <= 0 or self.eval_steps <= 0:
            raise ValueError("max_steps, batch_size, and eval_steps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self._network = None
        self._optimizer = None
        self._feature_columns: tuple[str, ...] | None = None
        self._input_dim: int | None = None
        self._fitted = False
        self.best_step: int | None = None

    def _build(self, input_dim: int):
        torch, _ = _torch()
        torch.manual_seed(self.seed)
        network = _build_network(input_dim, self.hidden_layers, self.dropout).to(torch.device(self.device_name))
        optimizer = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._network = network
        self._optimizer = optimizer

    def _frame(self, dataset: Any, segment: str, data_key: str) -> tuple[pd.DataFrame, pd.Series]:
        prepared = dataset.prepare(segment, col_set=["feature", "label"], data_key=data_key)
        return _feature_and_label(prepared)

    def _features(self, dataset: Any, segment: str) -> tuple[pd.DataFrame, pd.Index]:
        prepared = dataset.prepare(segment, col_set="feature", data_key="infer")
        if not isinstance(prepared, pd.DataFrame):
            raise ValueError("dataset feature preparation must return a DataFrame")
        if isinstance(prepared.columns, pd.MultiIndex):
            prepared = prepared.xs("feature", axis=1, level=0, drop_level=True)
        features = prepared.apply(pd.to_numeric, errors="coerce")
        if self._feature_columns is None:
            raise ValueError("model has not been fitted")
        if tuple(map(str, features.columns)) != self._feature_columns:
            raise ValueError("inference feature columns do not match training columns")
        values = features.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("inference features contain non-finite values")
        return features, prepared.index

    def _predict_values(self, values: np.ndarray) -> np.ndarray:
        torch, _ = _torch()
        if self._network is None:
            raise ValueError("model has not been fitted")
        self._network.eval()
        device = torch.device(self.device_name)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch_values = np.ascontiguousarray(values[start : start + self.batch_size], dtype=np.float32)
                batch = torch.from_numpy(batch_values).to(device)
                outputs.append(self._network(batch).reshape(-1).detach().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

    def fit(self, dataset: Any, **kwargs: Any) -> "MPSDNNModel":
        torch, _ = _torch()
        train_features, train_labels = self._frame(dataset, "train", "learn")
        valid_features, valid_labels = self._frame(dataset, "valid", "learn")
        self._feature_columns = tuple(map(str, train_features.columns))
        if tuple(map(str, valid_features.columns)) != self._feature_columns:
            raise ValueError("validation feature columns do not match training columns")
        self._input_dim = len(self._feature_columns)
        self._build(self._input_dim)
        device = torch.device(self.device_name)
        x_train = np.ascontiguousarray(train_features.to_numpy(dtype=np.float32))
        y_train = np.ascontiguousarray(train_labels.to_numpy(dtype=np.float32))
        x_valid = np.ascontiguousarray(valid_features.to_numpy(dtype=np.float32))
        y_valid = np.ascontiguousarray(valid_labels.to_numpy(dtype=np.float32))
        rng = np.random.default_rng(self.seed)
        best_loss = float("inf")
        stale_evals = 0
        best_state: Mapping[str, Any] | None = None
        self.best_step = None
        for step in range(1, self.max_steps + 1):
            self._network.train()
            self._optimizer.zero_grad(set_to_none=True)
            indices = rng.integers(0, len(x_train), size=min(self.batch_size, len(x_train)))
            values = torch.from_numpy(x_train[indices]).to(device)
            target = torch.from_numpy(y_train[indices]).to(device)
            loss = ((self._network(values).reshape(-1) - target) ** 2).mean()
            loss.backward()
            self._optimizer.step()
            if step % self.eval_steps != 0 and step != self.max_steps:
                continue
            validation = self._predict_values(x_valid)
            validation_loss = float(np.mean((validation - y_valid) ** 2))
            if validation_loss < best_loss:
                best_loss = validation_loss
                stale_evals = 0
                self.best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in self._network.state_dict().items()}
            else:
                stale_evals += 1
                if stale_evals >= self.early_stop_rounds:
                    break
        if best_state is None:
            raise RuntimeError("MPS DNN did not produce a valid validation checkpoint")
        self._network.load_state_dict(copy.deepcopy(best_state))
        self._fitted = True
        return self

    def predict(self, dataset: Any, segment: str = "test") -> pd.Series:
        if not self._fitted:
            raise ValueError("model is not fitted")
        features, index = self._features(dataset, segment)
        values = features.to_numpy(dtype=np.float32)
        return pd.Series(self._predict_values(values), index=index, name="score")
