import numpy as np

def compute_svd_basis(gradient_matrix: np.ndarray, threshold: float = 0.97) -> np.ndarray:
    """
    Computes the SVD of a gradient matrix and returns the principal basis vectors
    that capture `threshold` fraction of the total energy.
    
    gradient_matrix: shape (N_samples, D_parameters)
    threshold: Energy threshold (e.g., 0.97 for 97%)
    """
    # SVD on transpose: G.T = U * S * V.T
    # We want the column space of G.T (which are the principal gradient directions)
    U, S, _ = np.linalg.svd(gradient_matrix.T, full_matrices=False)
    
    # Calculate energy to determine how many components to keep
    energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    
    # Find number of components to reach threshold
    k = int(np.searchsorted(energy, threshold)) + 1
    
    # Extract the top k basis vectors
    basis = U[:, :k]
    
    return basis
