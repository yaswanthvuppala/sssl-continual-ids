import os
import pickle

import numpy as np
from sklearn.ensemble import IsolationForest as SklearnIsolationForest


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector operating on SSL embedding space.
    Complementary to the autoencoder detector - uses density-based scoring.
    """
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
        n_jobs: int = 1,
    ):
        if not 0.0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = SklearnIsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.is_fitted = False

    def train(self, normal_embeddings: np.ndarray):
        """Fit the Isolation Forest on normal traffic embeddings."""
        print(f"Fitting Isolation Forest on {len(normal_embeddings)} normal embeddings...")
        self.model.fit(normal_embeddings)
        self.is_fitted = True
        print("Isolation Forest fitted successfully.")

    def score(self, embedding: np.ndarray) -> float:
        """
        Returns anomaly score in [0, 1]. Higher = more anomalous.
        Isolation Forest returns negative scores for anomalies, so we invert and normalize.
        """
        if not self.is_fitted:
            raise RuntimeError("IsolationForest must be fitted before scoring.")
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        raw_scores = self.model.score_samples(embedding)
        normalized = np.clip(-raw_scores, 0.0, 1.0)
        return float(normalized[0]) if len(normalized) == 1 else normalized

    def is_anomaly(self, embedding: np.ndarray) -> bool:
        """Returns True if the sample is predicted as an outlier."""
        if not self.is_fitted:
            raise RuntimeError("IsolationForest must be fitted before prediction.")
        if embedding.ndim == 1:
            embedding = embedding[np.newaxis, :]
        pred = self.model.predict(embedding)
        return bool(pred[0] == -1)

    def save(self, path: str = None):
        """Save the fitted model to disk."""
        path = path or "./checkpoints/isolation_forest.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Isolation Forest saved to {path}")

    def load(self, path: str = None):
        """Load a previously fitted model from disk."""
        path = path or "./checkpoints/isolation_forest.pkl"
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        print(f"Isolation Forest loaded from {path}")