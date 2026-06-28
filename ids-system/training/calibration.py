import json
import os
import numpy as np
from scipy.optimize import minimize

class TemperatureScaler:
    """
    Learns a single temperature parameter T > 0 to calibrate logits.
    Minimizes Negative Log Likelihood (NLL) / cross-entropy on validation data.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def fit(self, logits: np.ndarray, labels: np.ndarray):
        """
        Fits temperature T on validation logits and labels.
        logits: shape (N, C) or (N,)
        labels: shape (N,) (integer class labels)
        """
        # Ensure logits are numpy array
        logits = np.asarray(logits, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        
        # If logits is 1D (binary probabilities/scores or single output logit),
        # convert to 2D binary logits [logit_benign, logit_attack] if possible.
        if logits.ndim == 1:
            logits = np.column_stack([np.zeros_like(logits), logits])
            
        # Define NLL loss function
        def nll_loss(T_val):
            # Clip T_val to avoid negative or near-zero temperatures
            t = max(T_val[0], 1e-4)
            scaled_logits = logits / t
            
            # Compute log-softmax manually for stability
            max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
            exp_logits = np.exp(scaled_logits - max_logits)
            softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Avoid log(0)
            eps = 1e-15
            softmax_probs = np.clip(softmax_probs, eps, 1 - eps)
            
            # Select probability of correct class
            n = len(labels)
            correct_probs = softmax_probs[np.arange(n), labels]
            loss = -np.mean(np.log(correct_probs))
            return loss

        # Run optimization
        initial_T = [self.temperature]
        res = minimize(nll_loss, initial_T, method='L-BFGS-B', bounds=[(1e-3, 10.0)])
        if res.success:
            self.temperature = float(res.x[0])
        else:
            print(f"[WARN] Temperature scaling optimization failed. Using initial T={self.temperature}")
            
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature to logits and computes softmax."""
        logits = np.asarray(logits, dtype=np.float32)
        is_1d = (logits.ndim == 1)
        if is_1d:
            logits = np.column_stack([np.zeros_like(logits), logits])
            
        scaled_logits = logits / self.temperature
        max_logits = np.max(scaled_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        if is_1d:
            return probs[:, 1]
        return probs

    def save(self, path: str):
        """Saves temperature parameter to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    def load(self, path: str):
        """Loads temperature parameter from a JSON file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.temperature = data.get("temperature", 1.0)
        else:
            print(f"[WARN] Temperature file {path} not found. Using default T=1.0")
            self.temperature = 1.0
