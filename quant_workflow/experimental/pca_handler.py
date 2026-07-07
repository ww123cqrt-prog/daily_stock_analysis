#!/usr/bin/env python3
"""
Experimental PCA-transformed Alpha158 features for Qlib.

Applies PCA to Alpha158 features before feeding to LightGBM.
This reduces multicollinearity and noise in the 158 features.

WARNING: this handler is kept for research only. It is not part of the
canonical training path because Qlib's grouped feature/label semantics need
additional validation after replacing the handler data frame.
"""
import sys
from pathlib import Path

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP


class Alpha158PCA(Alpha158):
    """Alpha158 with PCA dimensionality reduction."""

    def __init__(
        self,
        n_components=30,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=[],
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        self.n_components = n_components
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            freq=freq,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            fit_start_time=fit_start_time,
            fit_end_time=fit_end_time,
            process_type=process_type,
            filter_pipe=filter_pipe,
            inst_processors=inst_processors,
            **kwargs,
        )

    def setup_data(self, **kwargs):
        """Override to apply PCA after data loading."""
        super().setup_data(**kwargs)

        # Get the data
        df = self._data

        if df is None or len(df) == 0:
            return

        # Fit PCA on training data only
        fit_start = self.fit_start_time or df.index.get_level_values('datetime').min()
        fit_end = self.fit_end_time or df.index.get_level_values('datetime').max()

        train_mask = (
            (df.index.get_level_values('datetime') >= pd.Timestamp(fit_start))
            & (df.index.get_level_values('datetime') <= pd.Timestamp(fit_end))
        )
        train_df = df[train_mask]

        if len(train_df) == 0:
            return

        # Fit PCA
        n_comp = min(self.n_components, train_df.shape[1])
        pca = PCA(n_components=n_comp)

        # Handle NaN: fill with 0 for PCA fitting
        train_filled = train_df.fillna(0)
        pca.fit(train_filled.values)

        # Transform all data
        df_filled = df.fillna(0)
        transformed = pca.transform(df_filled.values)

        # Create new column names
        cols = ["PC{}".format(i+1) for i in range(n_comp)]
        new_df = pd.DataFrame(transformed, index=df.index, columns=cols)

        explained = pca.explained_variance_ratio_
        total_var = explained.sum()
        print("[PCA] {} components explain {:.1%} of variance".format(n_comp, total_var))
        print("[PCA] Top 5 components: {}".format(
            ", ".join(["{:.1%}".format(v) for v in explained[:5]])
        ))

        self._data = new_df
        self._pca = pca
