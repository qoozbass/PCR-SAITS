from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .backbones._device import choose_device
from .features import build_correction_rows, build_example_arrays
from .masks import apply_mask
from .model import PCRExampleDataset, PCRResidualNet
from .windows import build_windows, reconstruct_from_windows


class PCRCorrector:
    """Extraction of legacy `PCRSAITSV1CleanWrapper`.

    Phase 3 intentionally preserves the legacy explicit-mask fit API.
    High-level public convenience APIs are deferred to Phase 5.
    """

    checkpoint_ext = ".pt"

    def __init__(
        self,
        variant,
        n_steps,
        learning_rate,
        weight_decay,
        epochs,
        batch_size,
        patience,
        preserve_loss_weight,
        rel_loss_weight,
        sparse_loss_weight,
        base_model,
        feature_names,
        base_impute_stride=None,
        verbose=True,
    ):
        self.variant = variant
        self.n_steps = int(n_steps)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.preserve_loss_weight = preserve_loss_weight
        self.rel_loss_weight = rel_loss_weight
        self.sparse_loss_weight = sparse_loss_weight
        self.base_model = base_model
        self.feature_names = list(feature_names)
        self.base_impute_stride = (
            int(base_impute_stride)
            if base_impute_stride is not None
            else int(n_steps)
        )
        self.verbose = verbose
        self.device = choose_device()

        self.use_rel_loss = variant == "pcrsaitsv14_with_rel_loss"
        self.use_preserve_loss = variant == "pcrsaitsv14_with_preserve_loss"
        self.use_sparse_loss = variant == "pcrsaitsv14_with_sparse_loss"
        self.use_seasonal_branch = (
            variant != "pcrsaitsv14_no_seasonal_branch"
        )
        self.use_local_branch = variant != "pcrsaitsv14_no_local_branch"
        self.use_domain_tags = variant != "pcr_mlp_no_domain_tags"
        self.direct_residual = variant != "pcrsaitsv14_masked_residual"
        self.input_dim = 10 if self.use_domain_tags else 8

        self.model = PCRResidualNet(self.input_dim).to(self.device)

    def _build_examples(
        self,
        original_values,
        masked_values,
        base_imputed_values,
        target_mask,
        gap_len,
    ):
        return build_example_arrays(
            original_values=original_values,
            masked_values=masked_values,
            base_imputed_values=base_imputed_values,
            target_mask=target_mask,
            gap_len=gap_len,
            feature_names=self.feature_names,
            use_local_branch=self.use_local_branch,
            use_seasonal_branch=self.use_seasonal_branch,
            use_domain_tags=self.use_domain_tags,
        )

    def _loss(self, delta, corr_mask, target, base_err, easy_mask):
        pred_residual = corr_mask * delta
        corrected_err = torch.abs(pred_residual - target)
        rec = F.smooth_l1_loss(pred_residual, target)
        preserve = (easy_mask * torch.abs(pred_residual)).mean()
        rel = torch.relu(corrected_err - base_err).mean()
        sparse = torch.abs(corr_mask).mean()

        total = rec
        if self.use_preserve_loss:
            total = total + self.preserve_loss_weight * preserve
        if self.use_rel_loss:
            total = total + self.rel_loss_weight * rel
        if self.use_sparse_loss:
            total = total + self.sparse_loss_weight * sparse
        return total

    def _impute_full_series_with_base(self, masked_values: np.ndarray):
        stride = max(1, int(self.base_impute_stride))
        windows, starts = build_windows(
            masked_values,
            self.n_steps,
            stride=stride,
        )
        imputed_windows = self.base_model.impute(
            windows.astype(np.float32)
        )
        return reconstruct_from_windows(
            imputed_windows,
            starts,
            len(masked_values),
        )

    def fit(
        self,
        train_values,
        train_holdout_mask,
        train_gap_len,
        val_values,
        val_holdout_mask,
        val_gap_len,
    ):
        train_masked = apply_mask(train_values, train_holdout_mask)
        val_masked = apply_mask(val_values, val_holdout_mask)

        train_base = self._impute_full_series_with_base(train_masked)
        val_base = self._impute_full_series_with_base(val_masked)

        train_X, train_y, train_base_err, train_easy = self._build_examples(
            train_values,
            train_masked,
            train_base,
            train_holdout_mask,
            train_gap_len,
        )
        val_X, val_y, val_base_err, val_easy = self._build_examples(
            val_values,
            val_masked,
            val_base,
            val_holdout_mask,
            val_gap_len,
        )

        if len(train_X) == 0 or len(val_X) == 0:
            raise RuntimeError(
                "PCR training data is empty. Check mask generation."
            )

        train_loader = DataLoader(
            PCRExampleDataset(
                train_X,
                train_y,
                train_base_err,
                train_easy,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            PCRExampleDataset(
                val_X,
                val_y,
                val_base_err,
                val_easy,
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state, best_loss, bad_epochs = None, float("inf"), 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_sum, train_count = 0.0, 0

            for X, y, base_err, easy_mask in train_loader:
                X = X.to(self.device)
                y = y.to(self.device).unsqueeze(-1)
                base_err = base_err.to(self.device).unsqueeze(-1)
                easy_mask = easy_mask.to(self.device).unsqueeze(-1)

                opt.zero_grad(set_to_none=True)
                delta, corr_mask = self.model(
                    X,
                    direct_residual=self.direct_residual,
                )
                loss = self._loss(
                    delta,
                    corr_mask,
                    y,
                    base_err,
                    easy_mask,
                )

                if not torch.isfinite(loss):
                    if self.verbose:
                        print(
                            f"[WARN] Non-finite PCR loss in {self.variant}; "
                            "skipping batch."
                        )
                    opt.zero_grad(set_to_none=True)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )
                opt.step()

                train_sum += float(loss.detach().cpu()) * len(X)
                train_count += len(X)

            self.model.eval()
            val_sum, val_count = 0.0, 0

            with torch.no_grad():
                for X, y, base_err, easy_mask in val_loader:
                    X = X.to(self.device)
                    y = y.to(self.device).unsqueeze(-1)
                    base_err = base_err.to(self.device).unsqueeze(-1)
                    easy_mask = easy_mask.to(self.device).unsqueeze(-1)

                    delta, corr_mask = self.model(
                        X,
                        direct_residual=self.direct_residual,
                    )
                    loss = self._loss(
                        delta,
                        corr_mask,
                        y,
                        base_err,
                        easy_mask,
                    )

                    if not torch.isfinite(loss):
                        if self.verbose:
                            print(
                                "[WARN] Non-finite PCR validation loss in "
                                f"{self.variant}; skipping batch."
                            )
                        continue

                    val_sum += float(loss.detach().cpu()) * len(X)
                    val_count += len(X)

            train_epoch_loss = (
                train_sum / train_count
                if train_count > 0
                else float("inf")
            )
            val_epoch_loss = (
                val_sum / val_count
                if val_count > 0
                else float("inf")
            )

            self.model.eval()
            with torch.no_grad():
                sample_X = torch.tensor(
                    val_X[: min(4096, len(val_X))],
                    dtype=torch.float32,
                    device=self.device,
                )
                d_dbg, m_dbg = self.model(
                    sample_X,
                    direct_residual=self.direct_residual,
                )
                delta_abs_mean = float(torch.abs(d_dbg).mean().cpu())
                mask_mean = float(m_dbg.mean().cpu())
                applied_delta_mean = float(
                    torch.abs(d_dbg * m_dbg).mean().cpu()
                )

            if self.verbose:
                print(
                    f"[INFO] {self.variant} epoch={epoch}/{self.epochs} "
                    f"train_loss={train_epoch_loss:.6f} "
                    f"val_loss={val_epoch_loss:.6f} "
                    f"delta_abs_mean={delta_abs_mean:.6f} "
                    f"mask_mean={mask_mean:.6f} "
                    f"applied_delta_mean={applied_delta_mean:.6f}"
                )

            if val_epoch_loss + 1e-8 < best_loss:
                best_loss = val_epoch_loss
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(
                            f"[INFO] {self.variant} early stopping at "
                            f"epoch {epoch}"
                        )
                    break

        if best_state is None:
            raise RuntimeError(
                f"No valid state found for {self.variant}."
            )

        self.model.load_state_dict(best_state)

    def correct(
        self,
        original_masked_values,
        base_imputed_values,
        correction_mask,
        gap_len,
    ):
        """Apply PCR to the requested correction scope."""
        corrected = base_imputed_values.copy()

        rows, coords = build_correction_rows(
            original_masked_values=original_masked_values,
            base_imputed_values=base_imputed_values,
            correction_mask=correction_mask,
            gap_len=gap_len,
            feature_names=self.feature_names,
            use_local_branch=self.use_local_branch,
            use_seasonal_branch=self.use_seasonal_branch,
            use_domain_tags=self.use_domain_tags,
        )

        if rows:
            X = torch.tensor(
                np.stack(rows),
                dtype=torch.float32,
                device=self.device,
            )
            self.model.eval()
            with torch.no_grad():
                delta, corr_mask = self.model(
                    X,
                    direct_residual=self.direct_residual,
                )
                residual = (
                    delta.squeeze(-1) * corr_mask.squeeze(-1)
                ).cpu().numpy()

            for (t, f), r in zip(coords, residual):
                corrected[t, f] = corrected[t, f] + float(r)

        observed = np.isfinite(original_masked_values)
        corrected[observed] = original_masked_values[observed]
        return corrected

    def save(self, path: Path):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "variant": self.variant,
            },
            path,
        )

    @classmethod
    def load_from_checkpoint(cls, path: Path, **kwargs):
        obj = cls(**kwargs)
        state = torch.load(path, map_location=obj.device)
        obj.model.load_state_dict(state["state_dict"])
        obj.model.eval()
        return obj


# Temporary compatibility alias for Phase 4 legacy-equivalence harnesses.
PCRSAITSV1CleanWrapper = PCRCorrector
