import numpy as np
from typing import Tuple

def compute_severity(anomaly_score: float) -> str:
    """Maps a [0, 1] anomaly score to a severity label."""
    if anomaly_score > 0.9:
        return "CRITICAL"
    elif anomaly_score > 0.7:
        return "HIGH"
    elif anomaly_score > 0.5:
        return "MEDIUM"
    else:
        return "LOW"

def ensemble_anomaly_scores(ae_score: float, if_score: float, weight_ae: float = 0.6, weight_if: float = 0.4) -> float:
    """
    Combines scores from autoencoder and isolation forest into a single anomaly score.
    """
    return weight_ae * ae_score + weight_if * if_score

def find_optimal_threshold(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Finds the anomaly threshold that maximizes F1 on a validation set.
    scores: anomaly scores (higher = more anomalous)
    labels: 0 = normal, 1 = anomalous
    Returns: (optimal_threshold, best_f1)
    """
    best_f1 = 0.0
    best_threshold = 0.5
    
    for t in np.linspace(0.01, 0.99, 200):
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            
    return best_threshold, best_f1
