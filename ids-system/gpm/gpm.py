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
        
    def capture_gradient_basis(self, model: tf.keras.Model, dataset: tf.data.Dataset, loss_fn: Callable):
        """
        After task T, compute SVD basis of gradient vectors and store it.
        """
        print("Capturing gradient basis for current task...")
        grad_matrix = []
        
        # We iterate over the dataset and capture gradients per batch
        for x, y in dataset:
            with tf.GradientTape() as tape:
                preds = model(x, training=False)
                loss = loss_fn(y, preds)
                
            grads = tape.gradient(loss, model.trainable_variables)
            
            # Flatten and concatenate all gradients into a single vector
            flat_grad = np.concatenate([
                g.numpy().ravel() for g in grads if g is not None
            ])
            grad_matrix.append(flat_grad)
            
        # Shape: (N_batches, D_parameters)
        G = np.stack(grad_matrix)
        
        # Compute basis capturing threshold energy
        basis = compute_svd_basis(G, self.threshold)
        
        self.memory_bank.add_basis(basis)
        print(f"[GPM] Captured basis with {basis.shape[1]} components.")

    def project_gradients(self, grads: List[tf.Tensor], variables: List[tf.Variable]) -> List[tf.Tensor]:
        """
        Project current gradients onto the null-space of stored bases.
        """
        bases = self.memory_bank.get_all_bases()
        if not bases:
            return grads  # No past tasks, return original gradients
            
        projected_grads = []
        for g, v in zip(grads, variables):
            if g is None:
                projected_grads.append(g)
                continue
                
            g_np = g.numpy().ravel()
            
            # Orthogonal projection onto the null-space of all past task bases
            for basis in bases:
                # Calculate projection of gradient onto the basis: basis * (basis^T * g)
                proj = basis @ (basis.T @ g_np)
                # Subtract projection to get null-space component
                g_np = g_np - proj
                
            # Reshape back to original variable shape
            projected_g = tf.reshape(tf.constant(g_np, dtype=v.dtype), v.shape)
            projected_grads.append(projected_g)
            
        return projected_grads
