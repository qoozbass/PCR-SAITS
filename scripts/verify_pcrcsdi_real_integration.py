from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# This runner lives inside <project>/scripts. Put the patched project root at
# the front of sys.path so the proof exercises the candidate source tree, not
# an already-installed pcrsaits==1.0.0 distribution from the environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pcrsaits
from pcrsaits import CSDIBackbone, PCRCSDI, build_windows


def package_version(name: str):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def candidate_project_version():
    pyproject = PROJECT_ROOT / "pyproject.toml"
    import re
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"\s*$', pyproject.read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError("Cannot read project version from pyproject.toml")
    return m.group(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the real PyPOTS PCR-CSDI integration proof."
    )
    p.add_argument(
        "--output",
        default="PCRCSDI_REAL_INTEGRATION_RESULT.json",
        help=(
            "Path for the JSON evidence file. For strict closure, place "
            "this outside the patched source tree."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    source_file = Path(pcrsaits.__file__).resolve()
    try:
        source_is_project = source_file.is_relative_to(PROJECT_ROOT)
    except AttributeError:  # Python < 3.9 fallback
        source_is_project = str(source_file).startswith(str(PROJECT_ROOT))
    if not source_is_project:
        raise RuntimeError(
            "Integration proof imported pcrsaits from outside the patched "
            f"project tree: {source_file}"
        )

    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train = rng.normal(size=(48, 2))
    val = rng.normal(size=(24, 2))
    test = rng.normal(size=(16, 2))

    kwargs = dict(
        n_steps=4,
        n_features=2,
        epochs=1,
        batch_size=2,
        patience=1,
        n_layers=1,
        n_heads=1,
        n_channels=8,
        d_time_embedding=8,
        d_feature_embedding=4,
        d_diffusion_embedding=8,
        n_diffusion_steps=2,
        n_sampling_times=2,
        aggregation="median",
        sampling_seed=seed,
        verbose=False,
    )

    train_w, _ = build_windows(train, 4, stride=4)
    val_ori_w, _ = build_windows(val, 4, stride=4)
    val_masked = val.copy()
    val_masked[3:5, 0] = np.nan
    val_w, _ = build_windows(val_masked, 4, stride=4)

    backbone = CSDIBackbone(**kwargs)
    backbone.fit(train_w, val_w, val_ori_w)

    masked = test.copy()
    masked[4:7, 0] = np.nan
    masked_w, _ = build_windows(masked, 4, stride=4)

    samples_a = backbone.sample(masked_w)
    samples_b = backbone.sample(masked_w)
    deterministic = bool(np.array_equal(samples_a, samples_b))
    expected_shape = [4, 2, 4, 2]
    sample_shape_ok = list(samples_a.shape) == expected_shape

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csdi_path = td / "csdi.pypots"
        backbone.save(csdi_path)
        restored_backbone = CSDIBackbone.load_from_checkpoint(
            csdi_path,
            **kwargs,
        )
        checkpoint_equal = bool(
            np.array_equal(samples_a, restored_backbone.sample(masked_w))
        )

        pcr = PCRCSDI(
            backbone=restored_backbone,
            feature_names=["sensor_1", "sensor_2"],
            feature_groups=["sensor", "sensor"],
            n_steps=4,
            base_impute_stride=4,
            epochs=2,
            batch_size=16,
            patience=1,
            verbose=False,
        )
        pcr.fit(
            train,
            val,
            seed=seed,
            holdout_ratio=0.15,
            pointwise_fraction=0.5,
            block_patterns=(1, 2, 3),
            block_buffer=1,
        )
        out1 = pcr.impute(masked)
        observed = np.isfinite(masked)
        observed_preserved = bool(
            np.array_equal(out1[observed], masked[observed])
        )
        finite_missing = bool(np.isfinite(out1[~observed]).all())

        pcr_path = td / "pcrcsdi.pt"
        pcr.save(pcr_path)
        restored_pcr = PCRCSDI.load(
            pcr_path,
            backbone=restored_backbone,
        )
        out2 = restored_pcr.impute(masked)
        public_checkpoint_equal = bool(np.array_equal(out1, out2))

    checks = {
        "patched_source_imported": source_is_project,
        "real_pypots_csdi_fit": True,
        "sample_shape_ok": sample_shape_ok,
        "seeded_sampling_deterministic": deterministic,
        "csdi_checkpoint_roundtrip_equal": checkpoint_equal,
        "pcrcsdi_fit_impute": True,
        "observed_preserved": observed_preserved,
        "finite_missing_outputs": finite_missing,
        "pcrcsdi_checkpoint_roundtrip_equal": public_checkpoint_equal,
    }

    report = {
        "schema_version": 2,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "torch": torch.__version__,
            "pypots": package_version("pypots"),
            "pcrsaits_distribution": package_version("pcrsaits"),
            "candidate_pyproject_version": candidate_project_version(),
            "pcrsaits_source_file": str(source_file),
            "patched_project_root": str(PROJECT_ROOT),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "device_used_by_backbone": str(backbone.device),
        },
        "seed": seed,
        "patched_source_imported": checks["patched_source_imported"],
        "real_pypots_csdi_fit": checks["real_pypots_csdi_fit"],
        "sample_shape": list(samples_a.shape),
        "expected_sample_shape": expected_shape,
        "sample_shape_ok": checks["sample_shape_ok"],
        "seeded_sampling_deterministic": checks["seeded_sampling_deterministic"],
        "csdi_checkpoint_roundtrip_equal": checks["csdi_checkpoint_roundtrip_equal"],
        "pcrcsdi_fit_impute": checks["pcrcsdi_fit_impute"],
        "observed_preserved": checks["observed_preserved"],
        "finite_missing_outputs": checks["finite_missing_outputs"],
        "pcrcsdi_checkpoint_roundtrip_equal": checks["pcrcsdi_checkpoint_roundtrip_equal"],
        "status": "PASS" if all(checks.values()) else "FAIL",
    }

    result_path = Path(args.output).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"WROTE {result_path.resolve()}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
