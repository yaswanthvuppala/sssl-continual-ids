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

    def score_batch(self, flow_features: np.ndarray, flow_ids: Optional[List[str]] = None) -> List[IDSAlert]:
        """Scores a batch of flows in a single batched GPU forward pass."""
        if len(flow_features) == 0:
            return []
        if flow_ids is None:
            flow_ids = [f"flow_{i}" for i in range(len(flow_features))]

        # Single batched encoder pass
        embeddings = self.encode(flow_features)  # (N, embed_dim)
        emb_tf = tf.constant(embeddings, dtype=tf.float32)

        n_samples = len(flow_features)
        best_margins = np.full(n_samples, -np.inf)
        selected_types: List[Optional[str]] = [None] * n_samples
        selected_confs = np.zeros(n_samples, dtype=np.float32)
        max_raw_confs = np.zeros(n_samples, dtype=np.float32)

        # Batched evaluation per classifier head
        for attack_name, head in self.heads.items():
            logits = head(emb_tf, training=False)
            if self.temperatures and attack_name in self.temperatures:
                T = self.temperatures[attack_name]
                if T > 0:
                    logits = logits / T
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            attack_probs = probs[:, 1] if probs.shape[-1] == 2 else np.max(probs, axis=-1)

            threshold = self.attack_thresholds.get(attack_name, self.default_threshold)
            margins = attack_probs - threshold

            max_raw_confs = np.maximum(max_raw_confs, attack_probs)

            # Update best margin per sample
            for i in range(n_samples):
                if attack_probs[i] >= threshold and margins[i] > best_margins[i]:
                    best_margins[i] = margins[i]
                    selected_types[i] = attack_name
                    selected_confs[i] = float(attack_probs[i])

        # Batched anomaly scoring
        anomaly_scores = self.anomaly.score(embeddings)
        if isinstance(anomaly_scores, float):
            anomaly_scores = np.array([anomaly_scores])

        alerts = []
        for i in range(n_samples):
            stype = selected_types[i]
            anom_score = float(anomaly_scores[i])
            if stype is not None:
                label = stype
                conf = float(selected_confs[i])
            elif anom_score >= self.anomaly_threshold:
                label = "zero-day / unknown"
                conf = float(max_raw_confs[i])
            else:
                label = None
                conf = float(max_raw_confs[i])

            severity = compute_severity(anom_score)
            alerts.append(
                IDSAlert(
                    flow_id=flow_ids[i],
                    attack_type=label,
                    confidence=conf,
                    anomaly_score=anom_score,
                    severity=severity,
                )
            )
        return alerts

