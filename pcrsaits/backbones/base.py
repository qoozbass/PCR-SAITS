from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import numpy as np

class BackboneAdapter(ABC):
    checkpoint_ext = ".pypots"

    @abstractmethod
    def fit(self, train_windows: np.ndarray,
            val_windows: Optional[np.ndarray] = None,
            val_windows_ori: Optional[np.ndarray] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def impute(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        raise NotImplementedError
