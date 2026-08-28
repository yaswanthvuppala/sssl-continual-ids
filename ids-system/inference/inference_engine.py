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
        attack_thresholds: Optional[Dict[str, float]] = None,
        default_threshold: float = 0.5,
        anomaly_threshold: float = 0.65,
        temperatures: Optional[Dict[str, float]] = None,
        attack_threshold: Optional[float] = None,
    ):
        self.encoder = encoder
        self.encoder.trainable = False
        self.heads = heads
        self.anomaly = anomaly_detector
        self.default_threshold = default_threshold
        
        # Support either per-head dict or legacy single parameter
        if attack_thresholds is not None:
            self.attack_thresholds = attack_thresholds
        elif attack_threshold is not None:
            self.attack_thresholds = {name: attack_threshold for name in heads.keys()}
        else:
            self.attack_thresholds = {name: 0.80 for name in heads.keys()}
            
        self.anomaly_threshold = anomaly_threshold
        self.temperatures = temperatures or {}

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
        best_margin = -float('inf')
        selected_type: Optional[str] = None
        selected_conf = 0.0
        
        max_raw_conf = 0.0
        max_raw_type: Optional[str] = None

        for attack_name, head in self.heads.items():
            logits = head(tf.constant(embedding, dtype=tf.float32), training=False)
            
            # Apply Temperature Scaling if temperature is defined
            if self.temperatures and attack_name in self.temperatures:
                T = self.temperatures[attack_name]
                if T > 0:
                    logits = logits / T
                    
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            # Class 1 is always the "attack" class in binary heads
            attack_prob = float(probs[0, 1]) if probs.shape[-1] == 2 else float(np.max(probs[0]))
            
            if attack_prob > max_raw_conf:
                max_raw_conf = attack_prob
                max_raw_type = attack_name
                
            threshold = self.attack_thresholds.get(attack_name, self.default_threshold)
            margin = attack_prob - threshold
            
            if attack_prob >= threshold:
                if margin > best_margin:
                    best_margin = margin
                    selected_type = attack_name
                    selected_conf = attack_prob

        # --- Anomaly scoring ---
        anomaly_score = self.anomaly.score(embedding[0])

        # --- Decision logic ---
        if selected_type is not None:
            label = selected_type
            confidence = selected_conf
        elif anomaly_score >= self.anomaly_threshold:
            label = "zero-day / unknown"
            confidence = max_raw_conf
        else:
            label = None  # benign
            confidence = max_raw_conf

        severity = compute_severity(anomaly_score)

        return IDSAlert(
            flow_id=flow_id,
            attack_type=label,
            confidence=confidence,
            anomaly_score=anomaly_score,
            severity=severity,
        )

    def score_batch(self, flow_features: np.ndarray, flow_ids: Optional[List[str]] = None, batch_size: int = 4096) -> List[IDSAlert]:
        """Scores a batch of flows using high-performance vectorized operations."""
        n_samples = len(flow_features)
        if flow_ids is None:
            flow_ids = [f"flow_{i}" for i in range(n_samples)]
            
        if n_samples == 0:
            return []

        alerts = []
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_x = tf.constant(flow_features[start_idx:end_idx], dtype=tf.float32)
            batch_emb = self.encoder(batch_x, training=False).numpy()

            # Classifier heads evaluation in vector form
            head_probs = {}
            head_margins = {}
            for attack_name, head in self.heads.items():
                logits = head(tf.constant(batch_emb, dtype=tf.float32), training=False)
                if self.temperatures and attack_name in self.temperatures:
                    T = self.temperatures[attack_name]
                    if T > 0:
                        logits = logits / T
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                atk_prob = probs[:, 1] if probs.shape[-1] == 2 else np.max(probs, axis=-1)
                thresh = self.attack_thresholds.get(attack_name, self.default_threshold)
                head_probs[attack_name] = atk_prob
                head_margins[attack_name] = atk_prob - thresh

            # Anomaly scoring in vector form
            anomaly_scores = self.anomaly.score(batch_emb)
            if np.isscalar(anomaly_scores):
                anomaly_scores = np.array([anomaly_scores])

            # Vectorized decision assignment
            batch_len = end_idx - start_idx
            for b in range(batch_len):
                best_margin = -float('inf')
                selected_type = None
                selected_conf = 0.0
                max_raw_conf = 0.0

                for attack_name in self.heads.keys():
                    prob = float(head_probs[attack_name][b])
                    margin = float(head_margins[attack_name][b])
                    thresh = self.attack_thresholds.get(attack_name, self.default_threshold)

                    if prob > max_raw_conf:
                        max_raw_conf = prob

                    if prob >= thresh and margin > best_margin:
                        best_margin = margin
                        selected_type = attack_name
                        selected_conf = prob

                anom_sc = float(anomaly_scores[b])
                if selected_type is not None:
                    label = selected_type
                    conf = selected_conf
                elif anom_sc >= self.anomaly_threshold:
                    label = "zero-day / unknown"
                    conf = max_raw_conf
                else:
                    label = None
                    conf = max_raw_conf

                alerts.append(IDSAlert(
                    flow_id=flow_ids[start_idx + b],
                    attack_type=label,
                    confidence=conf,
                    anomaly_score=anom_sc,
                    severity=compute_severity(anom_sc),
                ))

        return alerts
