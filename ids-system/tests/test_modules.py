"""
Unit tests for the SSSL-Based Continual IDS.

Run with:
    python -m pytest tests/test_modules.py -v
    or
    python tests/test_modules.py
"""
import os
import sys
import unittest
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoder.flow_encoder import build_flow_encoder
from encoder.projection_head import build_projection_head
from encoder.losses import nt_xent_loss
from classifiers.base_head import build_classifier_head
from gpm.svd_utils import compute_svd_basis
from gpm.memory_bank import MemoryBank
from gpm.gpm import GradientProjectionMemory
from anomaly.autoencoder_detector import AutoencoderDetector
from anomaly.isolation_forest import IsolationForestDetector
from anomaly.anomaly_utils import compute_severity, ensemble_anomaly_scores
from data.augmentations import augment_weak, augment_strong
from data.preprocessing import FlowPreprocessor
from data.dataset_loader import FlowDatasetLoader


class TestFlowEncoder(unittest.TestCase):
    def test_output_shape(self):
        enc = build_flow_encoder(input_dim=80, hidden_dim=512, embed_dim=256)
        x = tf.random.normal((4, 80))
        out = enc(x, training=False)
        self.assertEqual(out.shape, (4, 256))

    def test_freeze(self):
        enc = build_flow_encoder(input_dim=80)
        enc.trainable = False
        for v in enc.trainable_variables:
            self.fail("Should have no trainable variables after freeze")


class TestProjectionHead(unittest.TestCase):
    def test_output_shape(self):
        proj = build_projection_head(in_dim=256, out_dim=128)
        x = tf.random.normal((4, 256))
        out = proj(x, training=False)
        self.assertEqual(out.shape, (4, 128))


class TestNTXentLoss(unittest.TestCase):
    def test_loss_positive(self):
        z1 = tf.random.normal((8, 128))
        z2 = tf.random.normal((8, 128))
        loss = nt_xent_loss(z1, z2)
        self.assertGreater(float(loss), 0.0)

    def test_identical_views_low_loss(self):
        z = tf.random.normal((8, 128))
        loss = nt_xent_loss(z, z)
        # Identical views should yield very low loss
        self.assertLess(float(loss), 5.0)


class TestClassifierHead(unittest.TestCase):
    def test_output_shape(self):
        head = build_classifier_head(embed_dim=256, num_classes=2)
        x = tf.random.normal((4, 256))
        out = head(x, training=False)
        self.assertEqual(out.shape, (4, 2))


class TestSVDUtils(unittest.TestCase):
    def test_basis_shape(self):
        G = np.random.randn(10, 50).astype(np.float32)
        basis = compute_svd_basis(G, threshold=0.97)
        self.assertEqual(basis.shape[0], 50)  # D dimensions
        self.assertGreater(basis.shape[1], 0)
        self.assertLessEqual(basis.shape[1], 10)  # at most N components


class TestGPM(unittest.TestCase):
    def test_projection_preserves_shape(self):
        mb = MemoryBank(save_dir="./checkpoints/gpm_test")
        gpm = GradientProjectionMemory(threshold=0.97, memory_bank=mb)

        # Simulate a stored basis
        D = 50
        basis = np.random.randn(D, 3).astype(np.float32)
        basis, _ = np.linalg.qr(basis)  # orthonormalize
        mb.add_basis(basis)

        # Create a fake gradient
        g = tf.constant(np.random.randn(D).astype(np.float32))
        v = tf.Variable(np.random.randn(D).astype(np.float32))

        projected = gpm.project_gradients([g], [v])
        self.assertEqual(projected[0].shape, g.shape)

    def test_no_bases_returns_original(self):
        mb = MemoryBank(save_dir="./checkpoints/gpm_test2")
        gpm = GradientProjectionMemory(threshold=0.97, memory_bank=mb)

        g = tf.constant(np.random.randn(50).astype(np.float32))
        v = tf.Variable(np.random.randn(50).astype(np.float32))

        projected = gpm.project_gradients([g], [v])
        np.testing.assert_array_equal(projected[0].numpy(), g.numpy())


class TestAutoencoderDetector(unittest.TestCase):
    def test_score_range(self):
        det = AutoencoderDetector(embed_dim=64, latent_dim=16)
        # Train briefly on random data
        data = np.random.randn(100, 64).astype(np.float32)
        det.train(data, epochs=2, batch_size=32)
        score = det.score(data[0])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestIsolationForest(unittest.TestCase):
    def test_score_range(self):
        det = IsolationForestDetector()
        data = np.random.randn(200, 64).astype(np.float32)
        det.train(data)
        score = det.score(data[0])
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestAnomalyUtils(unittest.TestCase):
    def test_severity_levels(self):
        self.assertEqual(compute_severity(0.95), "CRITICAL")
        self.assertEqual(compute_severity(0.75), "HIGH")
        self.assertEqual(compute_severity(0.55), "MEDIUM")
        self.assertEqual(compute_severity(0.3), "LOW")

    def test_ensemble_scores(self):
        score = ensemble_anomaly_scores(0.8, 0.6)
        self.assertAlmostEqual(score, 0.8 * 0.6 + 0.6 * 0.4, places=5)


class TestAugmentations(unittest.TestCase):
    def test_weak_preserves_shape(self):
        x = tf.random.normal((80,))
        out = augment_weak(x)
        self.assertEqual(out.shape, x.shape)

    def test_strong_preserves_shape(self):
        x = tf.random.normal((80,))
        out = augment_strong(x)
        self.assertEqual(out.shape, x.shape)


class TestPreprocessor(unittest.TestCase):
    def test_fit_transform(self):
        loader = FlowDatasetLoader(data_path=".")
        df = loader.create_synthetic_data(num_samples=100, num_features=20)
        preprocessor = FlowPreprocessor()
        X, y = preprocessor.fit_transform(df)
        self.assertEqual(X.shape, (100, 20))
        self.assertEqual(len(y), 100)
        self.assertEqual(X.dtype, np.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
