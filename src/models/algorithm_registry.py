# -*- coding: utf-8 -*-
"""
Algorithm Registry for quant research.

Provides a data-driven registry of available algorithms with their capabilities
and configuration schemas. Never dynamically imports user module paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AlgorithmCapability:
    """Describes what an algorithm can do."""
    supports_classification: bool = False
    supports_regression: bool = False
    supports_feature_importance: bool = False
    supports_online_learning: bool = False
    requires_gpu: bool = False
    max_features: Optional[int] = None
    supported_loss_functions: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AlgorithmConfigSchema:
    """JSON-schema-like description of algorithm configuration."""
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class AlgorithmEntry:
    """A registered algorithm."""
    id: str
    name: str
    name_zh: str
    category: str  # "rule", "linear", "tree_ensemble", "neural"
    description: str
    description_zh: str
    capability: AlgorithmCapability
    config_schema: AlgorithmConfigSchema
    default_params: Dict[str, Any] = field(default_factory=dict)


# === Built-in algorithm definitions ===

_RULE_CAP = AlgorithmCapability(
    supports_classification=True,
    supports_regression=False,
    supports_feature_importance=False,
)

_LINEAR_CAP = AlgorithmCapability(
    supports_classification=False,
    supports_regression=True,
    supports_feature_importance=True,
    supported_loss_functions=["ols", "ridge", "lasso"],
)

_TREE_CAP = AlgorithmCapability(
    supports_classification=True,
    supports_regression=True,
    supports_feature_importance=True,
    supported_loss_functions=["mse", "mae", "huber", "binary_logloss", "cross_entropy"],
)

_LIGHTGBM_CAP = AlgorithmCapability(
    supports_classification=True,
    supports_regression=True,
    supports_feature_importance=True,
    supports_online_learning=False,
    max_features=None,
    supported_loss_functions=["mse", "mae", "huber", "binary_logloss", "cross_entropy", "poisson"],
)

_XGBOOST_CAP = AlgorithmCapability(
    supports_classification=True,
    supports_regression=True,
    supports_feature_importance=True,
    supports_online_learning=False,
    max_features=None,
    supported_loss_functions=["rmse", "mae", "logloss", "cross_entropy", "poisson"],
)

_CATBOOST_CAP = AlgorithmCapability(
    supports_classification=True,
    supports_regression=True,
    supports_feature_importance=True,
    supports_online_learning=False,
    max_features=None,
    supported_loss_functions=["RMSE", "MAE", "Logloss", "CrossEntropy", "Poisson"],
)


_REGISTRY: Dict[str, AlgorithmEntry] = {
    "rule": AlgorithmEntry(
        id="rule",
        name="Rule-based Selection",
        name_zh="规则选股",
        category="rule",
        description="Simple rule-based stock selection using predefined criteria.",
        description_zh="基于预定义规则的简单选股。",
        capability=_RULE_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "rule_type": {"type": "string", "enum": ["momentum", "value", "quality"], "description": "Type of rule to apply"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50, "description": "Number of stocks to select"},
                "lookback_days": {"type": "integer", "minimum": 1, "maximum": 365, "default": 20, "description": "Lookback period in days"},
            },
            required=["rule_type"],
        ),
        default_params={"rule_type": "momentum", "top_k": 50, "lookback_days": 20},
    ),
    "linear_ols": AlgorithmEntry(
        id="linear_ols",
        name="OLS Linear Regression",
        name_zh="OLS线性回归",
        category="linear",
        description="Ordinary Least Squares linear regression for return prediction.",
        description_zh="普通最小二乘线性回归，用于收益预测。",
        capability=_LINEAR_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "fit_intercept": {"type": "boolean", "default": True, "description": "Whether to fit intercept"},
                "normalize": {"type": "boolean", "default": False, "description": "Whether to normalize features"},
            },
        ),
        default_params={"fit_intercept": True, "normalize": False},
    ),
    "linear_ridge": AlgorithmEntry(
        id="linear_ridge",
        name="Ridge Regression",
        name_zh="岭回归",
        category="linear",
        description="Ridge regression with L2 regularization for return prediction.",
        description_zh="带L2正则化的岭回归，用于收益预测。",
        capability=_LINEAR_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "alpha": {"type": "number", "minimum": 0.001, "maximum": 100.0, "default": 1.0, "description": "Regularization strength"},
                "fit_intercept": {"type": "boolean", "default": True, "description": "Whether to fit intercept"},
            },
            required=["alpha"],
        ),
        default_params={"alpha": 1.0, "fit_intercept": True},
    ),
    "lightgbm": AlgorithmEntry(
        id="lightgbm",
        name="LightGBM",
        name_zh="LightGBM",
        category="tree_ensemble",
        description="LightGBM gradient boosting framework for stock selection.",
        description_zh="LightGBM梯度提升框架，用于选股。",
        capability=_LIGHTGBM_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 10000, "default": 100, "description": "Number of trees"},
                "max_depth": {"type": "integer", "minimum": -1, "maximum": 20, "default": 6, "description": "Maximum tree depth (-1 for unlimited)"},
                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0, "default": 0.1, "description": "Learning rate"},
                "num_leaves": {"type": "integer", "minimum": 2, "maximum": 1024, "default": 31, "description": "Number of leaves"},
                "subsample": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.8, "description": "Row subsample ratio"},
                "colsample_bytree": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.8, "description": "Column subsample per tree"},
                "reg_alpha": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 0.0, "description": "L1 regularization"},
                "reg_lambda": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 0.0, "description": "L2 regularization"},
                "objective": {"type": "string", "enum": ["regression", "binary", "cross_entropy"], "default": "regression", "description": "Objective function"},
                "random_state": {"type": "integer", "minimum": 0, "default": 42, "description": "Random seed"},
            },
        ),
        default_params={
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "regression", "random_state": 42,
        },
    ),
    "xgboost": AlgorithmEntry(
        id="xgboost",
        name="XGBoost",
        name_zh="XGBoost",
        category="tree_ensemble",
        description="XGBoost gradient boosting framework for stock selection.",
        description_zh="XGBoost梯度提升框架，用于选股。",
        capability=_XGBOOST_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "n_estimators": {"type": "integer", "minimum": 10, "maximum": 10000, "default": 100, "description": "Number of trees"},
                "max_depth": {"type": "integer", "minimum": 0, "maximum": 20, "default": 6, "description": "Maximum tree depth"},
                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0, "default": 0.1, "description": "Learning rate"},
                "subsample": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.8, "description": "Row subsample ratio"},
                "colsample_bytree": {"type": "number", "minimum": 0.1, "maximum": 1.0, "default": 0.8, "description": "Column subsample per tree"},
                "reg_alpha": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 0.0, "description": "L1 regularization"},
                "reg_lambda": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 1.0, "description": "L2 regularization"},
                "objective": {"type": "string", "enum": ["reg:squarederror", "reg:linear", "binary:logistic"], "default": "reg:squarederror", "description": "Objective function"},
                "random_state": {"type": "integer", "minimum": 0, "default": 42, "description": "Random seed"},
            },
        ),
        default_params={
            "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "objective": "reg:squarederror", "random_state": 42,
        },
    ),
    "catboost": AlgorithmEntry(
        id="catboost",
        name="CatBoost",
        name_zh="CatBoost",
        category="tree_ensemble",
        description="CatBoost gradient boosting framework with ordered boosting.",
        description_zh="CatBoost梯度提升框架，支持有序提升。",
        capability=_CATBOOST_CAP,
        config_schema=AlgorithmConfigSchema(
            parameters={
                "iterations": {"type": "integer", "minimum": 10, "maximum": 10000, "default": 100, "description": "Number of iterations"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 16, "default": 6, "description": "Tree depth"},
                "learning_rate": {"type": "number", "minimum": 0.001, "maximum": 1.0, "default": 0.1, "description": "Learning rate"},
                "l2_leaf_reg": {"type": "number", "minimum": 0.0, "maximum": 10.0, "default": 3.0, "description": "L2 regularization"},
                "random_seed": {"type": "integer", "minimum": 0, "default": 42, "description": "Random seed"},
                "loss_function": {"type": "string", "enum": ["RMSE", "MAE", "Logloss", "CrossEntropy"], "default": "RMSE", "description": "Loss function"},
                "verbose": {"type": "integer", "minimum": 0, "maximum": 100, "default": 0, "description": "Verbosity level"},
            },
        ),
        default_params={
            "iterations": 100, "depth": 6, "learning_rate": 0.1,
            "l2_leaf_reg": 3.0, "random_seed": 42, "loss_function": "RMSE", "verbose": 0,
        },
    ),
}


def get_algorithm(algorithm_id: str) -> Optional[AlgorithmEntry]:
    """Get a registered algorithm by ID."""
    return _REGISTRY.get(algorithm_id)


def list_algorithms(category: Optional[str] = None) -> List[AlgorithmEntry]:
    """List all registered algorithms, optionally filtered by category."""
    if category is None:
        return list(_REGISTRY.values())
    return [a for a in _REGISTRY.values() if a.category == category]


def algorithm_to_dict(entry: AlgorithmEntry) -> Dict[str, Any]:
    """Convert an AlgorithmEntry to a JSON-serializable dict."""
    return {
        "id": entry.id,
        "name": entry.name,
        "name_zh": entry.name_zh,
        "category": entry.category,
        "description": entry.description,
        "description_zh": entry.description_zh,
        "capability": {
            "supports_classification": entry.capability.supports_classification,
            "supports_regression": entry.capability.supports_regression,
            "supports_feature_importance": entry.capability.supports_feature_importance,
            "supports_online_learning": entry.capability.supports_online_learning,
            "requires_gpu": entry.capability.requires_gpu,
            "max_features": entry.capability.max_features,
            "supported_loss_functions": entry.capability.supported_loss_functions,
        },
        "config_schema": {
            "parameters": entry.config_schema.parameters,
            "required": entry.config_schema.required,
        },
        "default_params": entry.default_params,
    }
