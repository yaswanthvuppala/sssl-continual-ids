import os
import pickle
import numpy as np
from typing import List, Optional

class MemoryBank:
    """
    Stores and manages the basis vectors of past tasks.
    """
    def __init__(self, save_dir: str = "./checkpoints/gpm"):
        self.save_dir = save_dir
        self.bases: List[np.ndarray] = []
        os.makedirs(self.save_dir, exist_ok=True)
        
    def add_basis(self, basis: np.ndarray):
        """Adds a new basis to the memory bank."""
        self.bases.append(basis)
        
    def get_all_bases(self) -> List[np.ndarray]:
        """Returns all stored bases."""
        return self.bases
        
    def save(self, filename: str = "memory_bank.pkl"):
        """Saves the memory bank to disk."""
        path = os.path.join(self.save_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(self.bases, f)
        print(f"Memory bank saved to {path} with {len(self.bases)} task bases.")
            
    def load(self, filename: str = "memory_bank.pkl"):
        """Loads the memory bank from disk."""
        path = os.path.join(self.save_dir, filename)
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.bases = pickle.load(f)
            print(f"Loaded memory bank with {len(self.bases)} task bases.")
        else:
            print(f"No existing memory bank found at {path}. Starting fresh.")
