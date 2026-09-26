from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .corrector import PCRCorrector
from .masks import build_correction_scope, make_holdout_train_mask
from .metadata import resolve_core_feature_names


_PUBLIC_VARIANT = "pcrsaitsv14_no_seasonal_branch"
_CHECKPOINT_SCHEMA_VERSION = 1


def _validate_2d_values(values, *, n_features: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}.")
    if arr.shape[1] != n_features:
        raise ValueError(
            f"{name} has {arr.shape[1]} features but the model expects "
            f"{n_features}."
        )
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one time step.")
    return arr


class _PublicPCRBase:
    """Thin public layer over the Phase-4-audited PCRCorrector.

    The supplied backbone must already be trained. This layer intentionally
    does not change PCR feature construction, loss, masks, windows, or
    correction mathematics.
    """

    public_name = "PCR"

    def __init__(
        self,
        *,
        backbone,
        feature_names: Sequence[str],
        feature_groups: Optional[Sequence[str]] = None,
        metadata_mode: str = "explicit",
        n_steps: int = 48,
        base_impute_stride: int = 24,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        batch_size: int = 256,
        patience: int = 10,
        verbose: bool = True,
    ):
        if not hasattr(backbone, "impute") or not callable(backbone.impute):
            raise TypeError(
                "backbone must provide an impute(windows) method. "
                "Use a trained SAITSBackbone/BRITSBackbone or a compatible "
                "BackboneAdapter."
            )
        if int(n_steps) <= 0:
            raise ValueError("n_steps must be positive.")
        if int(base_impute_stride) <= 0:
            raise ValueError("base_impute_stride must be positive.")
        if int(epochs) <= 0:
            raise ValueError("epochs must be positive.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive.")
        if int(patience) <= 0:
            raise ValueError("patience must be positive.")

        original_names = list(feature_names)
        core_names, normalized_groups = resolve_core_feature_names(
            original_names,
            metadata_mode=metadata_mode,
            feature_groups=feature_groups,
        )

        self.backbone = backbone
        self.feature_names = original_names
        self.feature_groups = normalized_groups
        self.metadata_mode = metadata_mode

        self._constructor_config = {
            "feature_names": list(original_names),
            "feature_groups": (
                list(normalized_groups)
                if normalized_groups is not None
                else None
            ),
            "metadata_mode": metadata_mode,
            "n_steps": int(n_steps),
            "base_impute_stride": int(base_impute_stride),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "patience": int(patience),
            "verbose": bool(verbose),
        }

        # The proposed public configurations for both PCR-SAITS and PCR-BRITS
        # share the audited no-seasonal/domain-tagged/direct-residual PCR core.
        self.corrector = PCRCorrector(
            variant=_PUBLIC_VARIANT,
            n_steps=int(n_steps),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            epochs=int(epochs),
            batch_size=int(batch_size),
            patience=int(patience),
            preserve_loss_weight=0.0,
            rel_loss_weight=0.0,
            sparse_loss_weight=0.0,
            base_model=backbone,
            feature_names=core_names,
            base_impute_stride=int(base_impute_stride),
            verbose=bool(verbose),
        )

    @property
    def core(self) -> PCRCorrector:
        """Return the audited low-level PCRCorrector instance."""
        return self.corrector

    def fit(
        self,
        train_values,
        val_values,
        *,
        seed: int = 7,
        holdout_ratio: float = 0.15,
        pointwise_fraction: float = 0.5,
        block_patterns=(6, 12, 24, 48),
        block_buffer: int = 1,
    ):
        """Fit the PCR residual corrector using deterministic holdout masks.

        Seed semantics preserve the paper runner convention:
        training holdout seed = seed + 100
        validation holdout seed = seed + 200

        The base backbone is expected to be trained before calling this method.
        """
        train = _validate_2d_values(
            train_values,
            n_features=len(self.feature_names),
            name="train_values",
        )
        val = _validate_2d_values(
            val_values,
            n_features=len(self.feature_names),
            name="val_values",
        )

        train_mask, train_gap = make_holdout_train_mask(
            train,
            seed=int(seed) + 100,
            holdout_ratio=holdout_ratio,
            pointwise_fraction=pointwise_fraction,
            block_patterns=tuple(block_patterns),
            block_buffer=block_buffer,
        )
        val_mask, val_gap = make_holdout_train_mask(
            val,
            seed=int(seed) + 200,
            holdout_ratio=holdout_ratio,
            pointwise_fraction=pointwise_fraction,
            block_patterns=tuple(block_patterns),
            block_buffer=block_buffer,
        )

        self.corrector.fit(
            train,
            train_mask,
            train_gap,
            val,
            val_mask,
            val_gap,
        )
        return self

    def impute(self, masked_values) -> np.ndarray:
        """Impute then correct every input-missing cell.

        Originally observed cells are restored exactly by the audited
        PCRCorrector.correct() implementation.
        """
        masked = _validate_2d_values(
            masked_values,
            n_features=len(self.feature_names),
            name="masked_values",
        )
        base = self.corrector._impute_full_series_with_base(masked)
        correction_mask, gap_len = build_correction_scope(masked)
        return self.corrector.correct(
            masked,
            base,
            correction_mask,
            gap_len,
        )

    def save(self, path) -> None:
        """Save PCR weights + public metadata.

        Backbone weights are intentionally not bundled. Save/load the backbone
        with its adapter, then pass the restored backbone to `load`.
        """
        payload = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "public_class": self.public_name,
            "constructor_config": self._constructor_config,
            "state_dict": self.corrector.model.state_dict(),
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path, *, backbone):
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported public checkpoint schema version: "
                f"{payload.get('schema_version')!r}."
            )
        if payload.get("public_class") != cls.public_name:
            raise ValueError(
                f"Checkpoint is for {payload.get('public_class')!r}, "
                f"not {cls.public_name!r}."
            )
        obj = cls(backbone=backbone, **payload["constructor_config"])
        obj.corrector.model.load_state_dict(payload["state_dict"])
        obj.corrector.model.eval()
        return obj


class PCRSAITS(_PublicPCRBase):
    """Public convenience API for PCR correction over a trained SAITS backbone."""

    public_name = "PCRSAITS"


class PCRBRITS(_PublicPCRBase):
    """Public convenience API for PCR correction over a trained BRITS backbone."""

    public_name = "PCRBRITS"
