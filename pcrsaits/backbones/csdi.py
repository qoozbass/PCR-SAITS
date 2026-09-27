from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import BackboneAdapter
from ._device import choose_device

try:
    from pypots.imputation import CSDI
except ImportError:
    CSDI = None


_VALID_TARGET_STRATEGIES = {"mix", "random"}
_VALID_SCHEDULES = {"quad", "linear"}
_VALID_AGGREGATIONS = {"median", "mean"}


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class CSDIBackbone(BackboneAdapter):
    """PyPOTS CSDI adapter for the PCR backbone contract.

    ``sample`` keeps the probabilistic diffusion ensemble while ``impute``
    reduces that ensemble to one deterministic point imputation for the
    unchanged PCR residual core.

    When ``sampling_seed`` is an integer, inference is deterministic for a
    fixed trained model and fixed input. The adapter forks *all* visible CUDA
    generators as well as the CPU generator, so inference does not leak RNG
    changes to the caller even on multi-GPU hosts.
    """

    checkpoint_ext = ".pypots"

    def __init__(
        self,
        n_steps: int,
        n_features: int,
        epochs: int,
        batch_size: int,
        patience: Optional[int],
        n_layers: int,
        n_heads: int,
        n_channels: int,
        d_time_embedding: int,
        d_feature_embedding: int,
        d_diffusion_embedding: int,
        n_diffusion_steps: int = 50,
        target_strategy: str = "random",
        is_unconditional: bool = False,
        schedule: str = "quad",
        beta_start: float = 0.0001,
        beta_end: float = 0.5,
        n_sampling_times: int = 20,
        aggregation: str = "median",
        sampling_seed: Optional[int] = 7,
        verbose: bool = True,
    ):
        if CSDI is None:
            raise RuntimeError(
                "PyPOTS CSDI is not available. Install a PyPOTS version "
                "that provides pypots.imputation.CSDI."
            )

        self.n_steps = _positive_int(n_steps, name="n_steps")
        self.n_features = _positive_int(n_features, name="n_features")
        epochs = _positive_int(epochs, name="epochs")
        batch_size = _positive_int(batch_size, name="batch_size")
        n_layers = _positive_int(n_layers, name="n_layers")
        n_heads = _positive_int(n_heads, name="n_heads")
        n_channels = _positive_int(n_channels, name="n_channels")
        d_time_embedding = _positive_int(
            d_time_embedding, name="d_time_embedding"
        )
        d_feature_embedding = _positive_int(
            d_feature_embedding, name="d_feature_embedding"
        )
        d_diffusion_embedding = _positive_int(
            d_diffusion_embedding, name="d_diffusion_embedding"
        )
        n_diffusion_steps = _positive_int(
            n_diffusion_steps, name="n_diffusion_steps"
        )
        self.n_sampling_times = _positive_int(
            n_sampling_times, name="n_sampling_times"
        )

        if patience is not None:
            patience = _positive_int(patience, name="patience")

        if target_strategy not in _VALID_TARGET_STRATEGIES:
            raise ValueError(
                "target_strategy must be one of "
                f"{sorted(_VALID_TARGET_STRATEGIES)}, got {target_strategy!r}."
            )
        if schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"schedule must be one of {sorted(_VALID_SCHEDULES)}, "
                f"got {schedule!r}."
            )
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {sorted(_VALID_AGGREGATIONS)}, "
                f"got {aggregation!r}."
            )
        if not isinstance(is_unconditional, (bool, np.bool_)):
            raise ValueError("is_unconditional must be a boolean.")
        if not isinstance(verbose, (bool, np.bool_)):
            raise ValueError("verbose must be a boolean.")
        if n_channels % n_heads != 0:
            raise ValueError(
                "n_channels must be divisible by n_heads because the CSDI "
                "Transformer uses n_channels as d_model."
            )

        beta_start = float(beta_start)
        beta_end = float(beta_end)
        if not np.isfinite(beta_start) or not np.isfinite(beta_end):
            raise ValueError("beta_start and beta_end must be finite.")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError(
                "Require 0 < beta_start < beta_end < 1 for the diffusion "
                "noise schedule."
            )

        if sampling_seed is not None:
            if isinstance(sampling_seed, bool) or not isinstance(
                sampling_seed, (int, np.integer)
            ):
                raise ValueError("sampling_seed must be an integer or None.")
            sampling_seed = int(sampling_seed)

        self.aggregation = aggregation
        self.sampling_seed = sampling_seed
        self.verbose = bool(verbose)
        self.device = choose_device()

        self.model = CSDI(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=n_layers,
            n_heads=n_heads,
            n_channels=n_channels,
            d_time_embedding=d_time_embedding,
            d_feature_embedding=d_feature_embedding,
            d_diffusion_embedding=d_diffusion_embedding,
            n_diffusion_steps=n_diffusion_steps,
            target_strategy=target_strategy,
            is_unconditional=bool(is_unconditional),
            schedule=schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            num_workers=0,
            device=self.device,
            saving_path=None,
            verbose=self.verbose,
        )

    def _as_windows(self, windows, *, name: str) -> np.ndarray:
        arr = np.asarray(windows, dtype=float)
        if arr.ndim != 3:
            raise ValueError(
                f"{name} must have shape [n_windows, n_steps, n_features], "
                f"got {arr.shape}."
            )
        if arr.shape[1] != self.n_steps:
            raise ValueError(
                f"{name} has {arr.shape[1]} steps but the adapter expects "
                f"{self.n_steps}."
            )
        if arr.shape[2] != self.n_features:
            raise ValueError(
                f"{name} has {arr.shape[2]} features but the adapter expects "
                f"{self.n_features}."
            )
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one window.")
        return arr

    def fit(
        self,
        train_windows: np.ndarray,
        val_windows: Optional[np.ndarray] = None,
        val_windows_ori: Optional[np.ndarray] = None,
    ) -> None:
        train = self._as_windows(train_windows, name="train_windows")
        train_set = {"X": train.astype(np.float32)}

        if (val_windows is None) != (val_windows_ori is None):
            raise ValueError(
                "CSDI validation requires both val_windows and "
                "val_windows_ori, or neither."
            )

        val_set = None
        if val_windows is not None:
            val = self._as_windows(val_windows, name="val_windows")
            val_ori = self._as_windows(
                val_windows_ori,
                name="val_windows_ori",
            )
            if val.shape != val_ori.shape:
                raise ValueError(
                    "val_windows and val_windows_ori must have identical "
                    f"shapes, got {val.shape} and {val_ori.shape}."
                )
            val_set = {
                "X": val.astype(np.float32),
                "X_ori": val_ori.astype(np.float32),
            }

        self.model.fit(train_set, val_set)

    def _rng_context(self):
        if self.sampling_seed is None:
            return nullcontext()

        devices = (
            list(range(torch.cuda.device_count()))
            if torch.cuda.is_available()
            else []
        )
        return torch.random.fork_rng(devices=devices)

    def sample(
        self,
        windows: np.ndarray,
        n_sampling_times: Optional[int] = None,
    ) -> np.ndarray:
        """Return CSDI samples with shape ``[N, S, L, F]``."""
        original = self._as_windows(windows, name="windows")
        x = original.astype(np.float32)

        n = (
            self.n_sampling_times
            if n_sampling_times is None
            else _positive_int(n_sampling_times, name="n_sampling_times")
        )

        with self._rng_context():
            if self.sampling_seed is not None:
                # torch.manual_seed seeds CPU and CUDA generators. All visible
                # CUDA generators are protected by fork_rng above.
                torch.manual_seed(self.sampling_seed)

            result = self.model.predict(
                {"X": x},
                n_sampling_times=n,
            )

        if "imputation" not in result:
            raise RuntimeError(
                "PyPOTS CSDI predict() did not return an 'imputation' field."
            )

        samples = np.asarray(result["imputation"], dtype=float)

        # Defensive compatibility if a dependency version squeezes S when S=1.
        if samples.ndim == 3:
            samples = samples[:, None, :, :]

        expected = (
            original.shape[0],
            n,
            self.n_steps,
            self.n_features,
        )
        if samples.shape != expected:
            raise RuntimeError(
                "Unexpected CSDI sample shape: "
                f"expected {expected}, got {samples.shape}."
            )

        observed = np.isfinite(original)
        for i in range(samples.shape[1]):
            sample_i = samples[:, i]
            sample_i[observed] = original[observed]

        return samples

    def impute(self, windows: np.ndarray) -> np.ndarray:
        """Return deterministic CSDI point imputation for the PCR core."""
        original = self._as_windows(windows, name="windows")
        samples = self.sample(original)

        if self.aggregation == "median":
            out = np.median(samples, axis=1)
        else:
            out = np.mean(samples, axis=1)

        observed = np.isfinite(original)
        out[observed] = original[observed]
        return np.asarray(out, dtype=float)

    def save(self, path: Path) -> None:
        self.model.save(str(path))

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        obj.model.load(str(path))
        return obj
