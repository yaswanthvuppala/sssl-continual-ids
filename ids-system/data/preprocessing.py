import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

class FlowPreprocessor:
    """
    Handles preprocessing of network flow datasets:
    - Missing value imputation
    - Feature scaling (Z-score normalization)
    - Label encoding
    """
    def __init__(self, drop_cols=None):
        self.drop_cols = list(drop_cols or ["id"])
        self.transformer = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame, label_col: str = "Label") -> Tuple[np.ndarray, np.ndarray]:
        """Fits the scalers and encoders, then transforms the data."""
        df = self._clean_data(df.copy())
        
        # Separate features and labels
        y = df[label_col].values
        X = self._split_features(df, label_col)
        self.feature_cols = X.columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [c for c in X.columns if c not in self.numeric_cols]
        
        # Fit and transform labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Fit and transform features
        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        X_scaled = self.transformer.fit_transform(X)
        
        self.is_fitted = True
        return X_scaled.astype(np.float32), y_encoded.astype(np.int32)

    def transform(self, df: pd.DataFrame, label_col: str = "Label") -> Tuple[np.ndarray, np.ndarray]:
        """Transforms data using previously fitted scalers."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform().")
            
        df = self._clean_data(df.copy())
        
        if label_col in df.columns:
            y = df[label_col].values
            y_encoded = self.label_encoder.transform(y)
            X = self._split_features(df, label_col)
        else:
            y_encoded = None
            X = df
        
        # Ensure column order matches
        if self.feature_cols:
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = 0 if col in (self.numeric_cols or []) else "unknown"
            X = X[self.feature_cols]
            
        X_scaled = self.transformer.transform(X)
        
        return X_scaled.astype(np.float32), (y_encoded.astype(np.int32) if y_encoded is not None else None)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes infinite values and imputes NaNs."""
        df = df.replace([np.inf, -np.inf], np.nan)
        # Fill numeric NaNs with 0 (or could use median)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        categorical_cols = [c for c in df.columns if c not in numeric_cols]
        df[categorical_cols] = df[categorical_cols].fillna("unknown").astype(str).replace("", "unknown")
        return df

    def _split_features(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        """Drops labels and obvious non-feature identifiers/leakage columns."""
        columns_to_drop = set(self.drop_cols)
        columns_to_drop.add(label_col)
        for known_label_col in ["Label", "label", "attack_cat"]:
            if known_label_col != label_col:
                columns_to_drop.add(known_label_col)
        return df.drop(columns=[c for c in columns_to_drop if c in df.columns])
        
    def get_classes(self) -> np.ndarray:
        return self.label_encoder.classes_

    def save(self, path: str):
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "FlowPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)
