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
        After task T, compute SVD basis of gradient vectors and store it.
        """
        print("Capturing gradient basis for current task...")
        grad_matrix = []
        total_batches = 0
        skipped_batches = 0
        
        # We iterate over the dataset and capture gradients per batch
        for x, y in dataset:
            total_batches += 1
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                loss = loss_fn(y, preds)
                
            grads = tape.gradient(loss, model.trainable_variables)
            
            # Flatten and concatenate all gradients into a single vector
            # Skip if any component gradient has NaN or Inf
            has_nan = False
            for g in grads:
                if g is not None:
                    if tf.reduce_any(tf.math.is_nan(g)) or tf.reduce_any(tf.math.is_inf(g)):
                        has_nan = True
                        break
            if has_nan:
                skipped_batches += 1
                print("WARNING: NaN/Inf detected in batch gradient during GPM capture. Skipping batch.")
                continue

            valid_grads = [g.numpy().ravel() for g in grads if g is not None]
            if not valid_grads:
                skipped_batches += 1
                continue
            flat_grad = np.concatenate(valid_grads).astype(np.float32, copy=False)
            if not np.isfinite(flat_grad).all() or np.linalg.norm(flat_grad) <= min_gradient_norm:
                skipped_batches += 1
                continue
            grad_matrix.append(flat_grad)

            if max_batches and len(grad_matrix) >= max_batches:
                print(f"[GPM] Reached gradient capture cap of {max_batches} valid batches.")
                break
            
        if not grad_matrix:
            print("WARNING: No valid gradient batches collected for GPM capture. Skipping add_basis.")
            return

        # Shape: (N_batches, D_parameters)
        G = np.stack(grad_matrix)
        print(
            f"[GPM] Captured {len(grad_matrix)} valid gradient batches "
            f"({skipped_batches} skipped, {total_batches} scanned) for SVD."
        )
        
        # Compute basis capturing threshold energy
        basis = compute_svd_basis(G, self.threshold)
        
        if basis.size == 0 or basis.shape[1] == 0:
            print("WARNING: SVD basis is empty. Skipping memory bank insertion.")
            return

        self.memory_bank.add_basis(basis)
        print(f"[GPM] Captured basis with {basis.shape[1]} components.")

    def project_gradients(self, grads: List[tf.Tensor], variables: List[tf.Variable]) -> List[tf.Tensor]:
        """
        Project current gradients onto the null-space of stored bases.
        Each basis was captured as a single flattened vector over ALL head parameters.
        We therefore flatten, project, and reshape the entire gradient vector at once.
        """
        bases = self.memory_bank.get_all_bases()
        if not bases:
            return grads  # No past tasks, return original gradients

        # Flatten all per-layer gradients into one big vector (same layout as capture)
        flat_parts = []
        shapes = []
        for g, v in zip(grads, variables):
            if g is None:
                flat_parts.append(None)
                shapes.append(None)
            else:
                g_np = g.numpy().ravel()
                flat_parts.append(g_np)
                shapes.append((v.shape, len(g_np)))

        valid = [p for p in flat_parts if p is not None]
        if not valid:
            return grads

        g_flat = np.concatenate(valid)  # shape: (total_params,)

        # Project onto null-space of all stored bases
        for basis in bases:
            if basis.size == 0 or basis.shape[1] == 0:
                continue
            if basis.shape[0] != g_flat.shape[0]:
                # Dimension mismatch (different head architecture) — skip this basis
                print(f"[GPM] Skipping basis (shape {basis.shape}) — "
                      f"incompatible with current head params ({g_flat.shape[0]})")
                continue
            proj = basis @ (basis.T @ g_flat)
            g_flat = g_flat - proj

        # Slice back into per-layer tensors
        projected_grads = []
        offset = 0
        flat_idx = 0
        for g, v in zip(grads, variables):
            if g is None:
                projected_grads.append(g)
            else:
                size = shapes[flat_idx][1]
                chunk = g_flat[offset:offset + size]
                projected_grads.append(tf.reshape(tf.constant(chunk, dtype=v.dtype), v.shape))
                offset += size
                flat_idx += 1

        return projected_grads
