from typing import Dict
import numpy as np

# Shortcut for type descriptors
NpzFile = np.lib.npyio.NpzFile

class SavedModelInfo:
    def __init__(
        self,
        embeddings: np.ndarray,
        importanceMatrices: NpzFile,
        globalInteractionMtx: np.ndarray
    ) -> None:
        self.embeddings: np.ndarray = embeddings
        self.importanceMatrices: NpzFile = importanceMatrices
        self.globalInteractionMtx: np.ndarray = globalInteractionMtx

