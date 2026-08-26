import numpy as np
import tensorflow as tf
from typing import List, Callable, Tuple

from gpm.svd_utils import compute_svd_basis
from gpm.memory_bank import MemoryBank

class GradientProjectionMemory:
    """
    Gradient Projection Memory (GPM) for preventing catastrophic forgetting.
    It captures the gradient subspaces of previous tasks and projects new task
    gradients into the null-space of these past subspaces.
    """
    def __init__(self, threshold: float = 0.97, memory_bank: MemoryBank = None):
        self.threshold = threshold
        self.memory_bank = memory_bank if memory_bank is not None else MemoryBank()
        
    def capture_gradient_basis(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        loss_fn: Callable,
        max_batches: int = 512,
        min_gradient_norm: float = 1e-12,
    ):
        """
        After task T, compute SVD basis of gradient vectors PER LAYER and store it.
        """
        print("Capturing per-layer gradient basis for current task...")
        n_vars = len(model.trainable_variables)
        per_layer_grads = [[] for _ in range(n_vars)]
        total_batches = 0
        skipped_batches = 0
        
        for x, y in dataset:
            total_batches += 1
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                loss = loss_fn(y, preds)
                
            grads = tape.gradient(loss, model.trainable_variables)
            
            has_nan = False
            for g in grads:
                if g is not None:
                    if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                        has_nan = True
                        break
            if has_nan:
                skipped_batches += 1
                continue

            for idx, g in enumerate(grads):
                if g is not None:
                    g_flat = g.numpy().ravel()
                    per_layer_grads[idx].append(g_flat)

            if max_batches and len(per_layer_grads[0]) >= max_batches:
                print(f"[GPM] Reached gradient capture cap of {max_batches} valid batches.")
                break

        task_layer_bases = []
        for idx, layer_g_list in enumerate(per_layer_grads):
            if not layer_g_list:
                task_layer_bases.append(np.array([]))
                continue
            G_layer = np.stack(layer_g_list)  # (N_batches, param_dim)
            basis_l = compute_svd_basis(G_layer, self.threshold)
            task_layer_bases.append(basis_l)

        self.memory_bank.add_basis(task_layer_bases)
        comp_counts = [b.shape[1] if b.ndim > 1 else 0 for b in task_layer_bases]
        print(f"[GPM] Captured per-layer basis. Components per layer: {comp_counts}")

    def project_gradients(self, grads: List[tf.Tensor], variables: List[tf.Variable]) -> List[tf.Tensor]:
        """
        Project current gradients onto the null-space of stored bases LAYER-BY-LAYER.
        """
        bases_list = self.memory_bank.get_all_bases()
        if not bases_list:
            return grads

        projected_grads = []
        for idx, (g, v) in enumerate(zip(grads, variables)):
            if g is None:
                projected_grads.append(g)
                continue

            g_flat = g.numpy().ravel().astype(np.float32)

            for task_bases in bases_list:
                if isinstance(task_bases, list) and idx < len(task_bases):
                    basis_l = task_bases[idx]
                    if basis_l is not None and basis_l.size > 0 and basis_l.ndim == 2:
                        if basis_l.shape[0] == g_flat.shape[0]:
                            proj = basis_l @ (basis_l.T @ g_flat)
                            g_flat = g_flat - proj
                elif isinstance(task_bases, np.ndarray) and task_bases.size > 0:
                    # Legacy fallback for global flat bases
                    if task_bases.shape[0] == g_flat.shape[0]:
                        proj = task_bases @ (task_bases.T @ g_flat)
                        g_flat = g_flat - proj

            projected_grads.append(tf.reshape(tf.constant(g_flat, dtype=v.dtype), v.shape))

        return projected_grads
