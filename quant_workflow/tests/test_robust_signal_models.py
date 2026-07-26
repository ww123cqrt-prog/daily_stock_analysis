import numpy as np
import pandas as pd

from robust_signal_models import EqualWeightRankModel


class FakeDataset:
    def __init__(self, train: pd.DataFrame, test_features: pd.DataFrame) -> None:
        self.train = train
        self.test_features = test_features

    def prepare(self, segment, col_set="__all", data_key="infer"):
        if segment == "train":
            return self.train.copy()
        if segment == "test" and col_set == "feature":
            return self.test_features.copy()
        raise AssertionError((segment, col_set, data_key))


def test_equal_weight_rank_model_learns_signs_from_training_only() -> None:
    train_index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    f1 = [1, 2, 3, 1, 2, 3]
    f2 = [3, 2, 1, 3, 2, 1]
    labels = [0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    train = pd.DataFrame(
        {
            ("feature", "F1"): f1,
            ("feature", "F2"): f2,
            ("label", "LABEL0"): labels,
        },
        index=train_index,
    )
    test_index = pd.MultiIndex.from_product(
        [[pd.Timestamp("2025-01-02")], ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    test_features = pd.DataFrame({"F1": [3, 2, 1], "F2": [1, 2, 3]}, index=test_index)
    dataset = FakeDataset(train, test_features)
    model = EqualWeightRankModel()

    model.fit(dataset)
    prediction = model.predict(dataset)

    assert model.factor_signs == {"F1": 1.0, "F2": -1.0}
    assert prediction.loc[(pd.Timestamp("2025-01-02"), "A")] > prediction.loc[
        (pd.Timestamp("2025-01-02"), "C")
    ]
    assert np.isfinite(prediction).all()
