import os
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.dataset_loader import FlowDatasetLoader, KDD_FEATURE_COLUMNS
from data.preprocessing import FlowPreprocessor

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestKDDCup99Loader(unittest.TestCase):
    def test_uses_labeled_train_and_corrected_test_files(self):
        loader = FlowDatasetLoader(str(FIXTURES / "kddcup99"))
        train_df = loader.load_dataset("kddcup99", "train")
        test_df = loader.load_dataset("kddcup99", "test")

        self.assertEqual(len(KDD_FEATURE_COLUMNS), 41)
        self.assertEqual(train_df["Label"].tolist(), ["normal", "attack"])
        self.assertEqual(train_df["AttackCategory"].tolist(), ["normal", "dos"])
        self.assertEqual(test_df["AttackCategory"].tolist(), ["probe"])


class TestCICIDS2017Loader(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_disjoint(self):
        loader = FlowDatasetLoader(str(FIXTURES / "cicids2017"))
        train_df = loader.load_dataset("cicids2017", "train")
        test_df = loader.load_dataset("cicids2017", "test")
        repeated_test_df = loader.load_dataset("cicids2017", "test")

        self.assertEqual(len(train_df), 32)
        self.assertEqual(len(test_df), 8)
        self.assertEqual(
            test_df["Flow ID"].tolist(), repeated_test_df["Flow ID"].tolist()
        )
        self.assertTrue(
            set(train_df["Flow ID"]).isdisjoint(set(test_df["Flow ID"]))
        )
        self.assertEqual(set(test_df["Label"]), {"normal", "attack"})
        self.assertEqual(set(test_df["AttackCategory"]), {"normal", "dos"})


class TestStandardizedLabels(unittest.TestCase):
    def test_preprocessor_drops_all_label_metadata(self):
        df = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0, 3.0, 4.0],
                "feature_b": [4.0, 3.0, 2.0, 1.0],
                "Label": ["normal", "attack", "normal", "attack"],
                "AttackLabel": ["BENIGN", "DDoS", "BENIGN", "DDoS"],
                "AttackCategory": ["normal", "dos", "normal", "dos"],
            }
        )
        preprocessor = FlowPreprocessor()
        features, labels = preprocessor.fit_transform(df, label_col="Label")

        self.assertEqual(features.shape, (4, 2))
        self.assertEqual(labels.shape, (4,))


if __name__ == "__main__":
    unittest.main()
