import numpy as np

def compute_svd_basis(gradient_matrix: np.ndarray, threshold: float = 0.97) -> np.ndarray:
    """
    Computes the SVD of a gradient matrix and returns the principal basis vectors
    that capture `threshold` fraction of the total energy.
    
    gradient_matrix: shape (N_samples, D_parameters)
    threshold: Energy threshold (e.g., 0.97 for 97%)
    """
    # Replace NaN/Inf with 0.0 and drop all-zero gradient rows.
    gradient_matrix = np.nan_to_num(
        gradient_matrix, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
    
    # If matrix is all zeros, return an empty array (no basis directions)
    if not np.any(gradient_matrix):
        print("WARNING: Gradient matrix is all zeros (or contains only NaNs/Infs). Returning empty basis.")
        return np.empty((gradient_matrix.shape[1], 0))

    row_norms = np.linalg.norm(gradient_matrix, axis=1)
    gradient_matrix = gradient_matrix[row_norms > 1e-12]
    if gradient_matrix.shape[0] == 0:
        print("WARNING: Gradient matrix contains no non-zero rows. Returning empty basis.")
        return np.empty((gradient_matrix.shape[1], 0))

    n_samples, n_params = gradient_matrix.shape

    try:
        if n_samples <= n_params:
            # The GPM basis lives in parameter space, but for IDS heads the number
            # of captured batches is usually far smaller than parameter count.
            # Decomposing G @ G.T avoids a large tall-matrix SVD.
            gram = gradient_matrix @ gradient_matrix.T
            eigvals, eigvecs = np.linalg.eigh(gram.astype(np.float64, copy=False))
            order = np.argsort(eigvals)[::-1]
            eigvals = np.maximum(eigvals[order], 0.0)
            eigvecs = eigvecs[:, order]
            keep = eigvals > 1e-12
            S = np.sqrt(eigvals[keep])
            eigvecs = eigvecs[:, keep]
            if S.size == 0:
                print("WARNING: Sum of squared singular values is zero. Returning empty basis.")
                return np.empty((n_params, 0))
            U = (gradient_matrix.T @ eigvecs) / S
        else:
            # SVD on transpose: G.T = U * S * V.T
            # We want the column space of G.T (principal gradient directions).
            U, S, _ = np.linalg.svd(gradient_matrix.T, full_matrices=False)
    except np.linalg.LinAlgError as e:
        print(f"WARNING: SVD did not converge: {e}. Returning empty basis.")
        return np.empty((n_params, 0))
    
    sum_S_squared = np.sum(S ** 2)
    if sum_S_squared == 0:
        print("WARNING: Sum of squared singular values is zero. Returning empty basis.")
        return np.empty((n_params, 0))

    # Calculate energy to determine how many components to keep
    energy = np.cumsum(S ** 2) / sum_S_squared
    
    # Find number of components to reach threshold
    k = int(np.searchsorted(energy, threshold)) + 1
    
    # Extract the top k basis vectors
    basis = U[:, :k]
    basis, _ = np.linalg.qr(basis)
    
    return basis.astype(np.float32, copy=False)
