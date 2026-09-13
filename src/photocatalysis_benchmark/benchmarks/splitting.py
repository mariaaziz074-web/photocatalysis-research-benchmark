"""
src/photocatalysis_benchmark/benchmarks/splitting.py
====================================================
Five leakage-resistant train/test splitting strategies for photocatalysis benchmarking.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger("splitting")


class DataSplitter:
    """
    Implements 5 leakage-resistant splitting strategies.
    """

    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        self.df = df.copy()
        self.test_size = test_size
        self.random_state = random_state

    def random_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strategy 1: Random stratified split.
        Risk: None (baseline).
        """
        train_df, test_df = train_test_split(
            self.df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        logger.info(f"Random Split: Train={len(train_df)}, Test={len(test_df)}")
        return train_df, test_df

    def catalyst_holdout_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strategy 2: Catalyst-based holdout.
        Risk: Tests generalization to unseen catalyst formulations.
        """
        unique_catalysts = self.df["catalyst_name"].unique()
        np.random.seed(self.random_state)
        test_catalysts = np.random.choice(
            unique_catalysts,
            size=max(1, int(len(unique_catalysts) * self.test_size)),
            replace=False
        )

        test_df = self.df[self.df["catalyst_name"].isin(test_catalysts)]
        train_df = self.df[~self.df["catalyst_name"].isin(test_catalysts)]

        logger.info(f"Catalyst Holdout: Train={len(train_df)}, Test={len(test_df)}")
        logger.info(f"  Held-out catalysts: {list(test_catalysts)}")
        return train_df, test_df

    def dye_holdout_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strategy 3: Dye-based holdout.
        Risk: Tests generalization to unseen pollutants.
        """
        unique_dyes = self.df["dye_name"].unique()
        np.random.seed(self.random_state)
        test_dyes = np.random.choice(
            unique_dyes,
            size=max(1, int(len(unique_dyes) * self.test_size)),
            replace=False
        )

        test_df = self.df[self.df["dye_name"].isin(test_dyes)]
        train_df = self.df[~self.df["dye_name"].isin(test_dyes)]

        logger.info(f"Dye Holdout: Train={len(train_df)}, Test={len(test_df)}")
        logger.info(f"  Held-out dyes: {list(test_dyes)}")
        return train_df, test_df

    def paper_holdout_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strategy 4: Paper-based (DOI) holdout.
        Risk: Tests generalization across different laboratories/experimental protocols.
        """
        unique_dois = self.df["source_doi"].unique()
        np.random.seed(self.random_state)
        test_dois = np.random.choice(
            unique_dois,
            size=max(1, int(len(unique_dois) * self.test_size)),
            replace=False
        )

        test_df = self.df[self.df["source_doi"].isin(test_dois)]
        train_df = self.df[~self.df["source_doi"].isin(test_dois)]

        logger.info(f"Paper Holdout: Train={len(train_df)}, Test={len(test_df)}")
        logger.info(f"  Held-out papers: {len(test_dois)}")
        return train_df, test_df

    def time_based_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Strategy 5: Temporal split (chronological by publication year).
        Risk: Tests model performance on future experimental data.
        """
        # Extract year from DOI or use extraction_date
        if "extraction_date" in self.df.columns:
            self.df["year"] = pd.to_datetime(self.df["extraction_date"]).dt.year
        else:
            # Extract from data_id (PCD-YEAR-XXXX format)
            self.df["year"] = self.df["data_id"].str.extract(r'PCD-(\d{4})-')[0].astype(int)

        sorted_df = self.df.sort_values("year")
        split_idx = int(len(sorted_df) * (1 - self.test_size))

        train_df = sorted_df.iloc[:split_idx]
        test_df = sorted_df.iloc[split_idx:]

        logger.info(f"Time-Based Split: Train={len(train_df)}, Test={len(test_df)}")
        logger.info(f"  Train years: {train_df['year'].min()}-{train_df['year'].max()}")
        logger.info(f"  Test years: {test_df['year'].min()}-{test_df['year'].max()}")

        return train_df.drop(columns=["year"]), test_df.drop(columns=["year"])

    def get_all_splits(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate all 5 splits.
        """
        return {
            "random": self.random_split(),
            "catalyst_holdout": self.catalyst_holdout_split(),
            "dye_holdout": self.dye_holdout_split(),
            "paper_holdout": self.paper_holdout_split(),
            "time_based": self.time_based_split()
        }