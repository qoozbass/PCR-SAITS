from pathlib import Path
import numpy as np
from .base import BackboneAdapter
from ._device import choose_device

try:
    from pypots.imputation import BRITS
except Exception:
    BRITS = None

class BRITSBackbone(BackboneAdapter):
    checkpoint_ext = ".pypots"

    def __init__(self, n_steps, n_features, epochs, batch_size, patience,
                 rnn_hidden_size, verbose=True):
        if BRITS is None:
            raise RuntimeError("PyPOTS BRITS is not installed. Please install pypots.")
        self.model = BRITS(
            n_steps=n_steps,
            n_features=n_features,
            rnn_hidden_size=rnn_hidden_size,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=choose_device(),
            verbose=verbose,
            saving_path=None,
        )
        self.verbose = verbose

    def fit(self, train_windows, val_windows=None, val_windows_ori=None):
        train_set = {"X": train_windows.astype(np.float32)}
        if val_windows is not None and val_windows_ori is not None:
            val_set = {
                "X": val_windows.astype(np.float32),
                "X_ori": val_windows_ori.astype(np.float32),
            }
            try:
                self.model.fit(train_set, val_set)
                return
            except Exception as exc:
                print(
                    f"[WARN] BRITS fit with val_set failed: {exc}. "
                    "Falling back to train only.",
                    flush=True,
                )
        self.model.fit(train_set)

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
