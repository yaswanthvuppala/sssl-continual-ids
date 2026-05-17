import os
import sys
import numpy as np
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anomaly.anomaly_utils import compute_severity

@dataclass
class IDSAlert:
    """Structured alert output from the IDS inference engine."""
    flow_id: str
    attack_type: Optional[str]
    confidence: float
    anomaly_score: float
    severity: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self):
        label = self.attack_type if self.attack_type else "BENIGN"
        return (
            f"[{self.timestamp}] [{self.severity:8s}] "
            f"Flow={self.flow_id} | Type={label} | "
            f"Confidence={self.confidence:.3f} | Anomaly={self.anomaly_score:.3f}"
        )

class IDSInferenceEngine:
    """
    Real-time IDS scoring engine.
    Runs the frozen encoder, all classifier heads in parallel, and the anomaly detector
    to produce a unified alert per flow.
    """
    def __init__(
        self,
        encoder: tf.keras.Model,
        heads: Dict[str, tf.keras.Model],
        anomaly_detector,
        attack_threshold: float = 0.80,
        anomaly_threshold: float = 0.65,
    ):
        self.encoder = encoder
        self.encoder.trainable = False
        self.heads = heads
        self.anomaly = anomaly_detector
        self.attack_threshold = attack_threshold
        self.anomaly_threshold = anomaly_threshold

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encodes raw features through the frozen SSL encoder."""
        x_tf = tf.constant(x, dtype=tf.float32)
        if x_tf.ndim == 1:
            x_tf = tf.expand_dims(x_tf, 0)
        return self.encoder(x_tf, training=False).numpy()

    def score_single(self, flow_features: np.ndarray, flow_id: str = "unknown") -> IDSAlert:
        """
        Scores a single flow sample through all heads and the anomaly detector.
        """
        embedding = self.encode(flow_features)

        # --- Run all classifier heads ---
        best_conf = 0.0
        best_type: Optional[str] = None
        for attack_name, head in self.heads.items():
            logits = head(tf.constant(embedding, dtype=tf.float32), training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            # Class 1 is always the "attack" class in binary heads
            attack_prob = float(probs[0, 1]) if probs.shape[-1] == 2 else float(np.max(probs[0]))
            if attack_prob > best_conf:
                best_conf = attack_prob
                best_type = attack_name

        # --- Anomaly scoring ---
        anomaly_score = self.anomaly.score(embedding[0])

        # --- Decision logic ---
        if best_conf >= self.attack_threshold:
            label = best_type
        elif anomaly_score >= self.anomaly_threshold:
            label = "zero-day / unknown"
        else:
            label = None  # benign

        severity = compute_severity(anomaly_score)

        return IDSAlert(
            flow_id=flow_id,
            attack_type=label,
            confidence=best_conf,
            anomaly_score=anomaly_score,
            severity=severity,
        )

    def score_batch(self, flow_features: np.ndarray, flow_ids: Optional[List[str]] = None) -> List[IDSAlert]:
        """Scores a batch of flows."""
        if flow_ids is None:
            flow_ids = [f"flow_{i}" for i in range(len(flow_features))]
        alerts = []
        for i in range(len(flow_features)):
            alert = self.score_single(flow_features[i], flow_ids[i])
            alerts.append(alert)
        return alerts
