from pathlib import Path
import numpy as np
from .base import BackboneAdapter
from ._device import choose_device

try:
    from pypots.imputation import SAITS
except Exception:
    SAITS = None

class SAITSBackbone(BackboneAdapter):
    checkpoint_ext = ".pypots"

    def __init__(self, n_steps, n_features, epochs, batch_size, patience,
                 d_model, d_ffn, n_heads, n_layers, dropout, verbose=True):
        if SAITS is None:
            raise RuntimeError("PyPOTS is not installed. Please install pypots.")
        self.model = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=n_layers,
            d_model=d_model,
            d_ffn=d_ffn,
            n_heads=n_heads,
            d_k=d_model // max(1, n_heads),
            d_v=d_model // max(1, n_heads),
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=choose_device(),
        )
        self.verbose = verbose

    def fit(self, train_windows, val_windows, val_windows_ori=None):
        train_set = {"X": train_windows.astype(np.float32)}
        if val_windows_ori is None:
            val_windows_ori = val_windows
        val_set = {
            "X": val_windows.astype(np.float32),
            "X_ori": val_windows_ori.astype(np.float32),
        }
        self.model.fit(train_set, val_set)

    def impute(self, windows):
        out = self.model.impute({"X": windows.astype(np.float32)})
        return np.asarray(out, dtype=float)

    def save(self, path: Path):
        self.model.save(str(path))

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        obj.model.load(str(path))
        return obj
