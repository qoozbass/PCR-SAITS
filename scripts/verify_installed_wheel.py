from __future__ import annotations
import argparse
import importlib.metadata as md
import json
import site
import sys
import sysconfig
import tempfile
from pathlib import Path

import numpy as np
import torch


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _provenance_ok(source_file: Path, distribution_root: Path, site_roots: list[Path]) -> bool:
    return any(_under(source_file, root) for root in site_roots) and _under(source_file, distribution_root)


def main():
    p = argparse.ArgumentParser(description="Verify a built pcrsaits wheel from site-packages using real CSDI/PCRCSDI.")
    p.add_argument("--expected-version", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--forbid-source-root")
    a = p.parse_args()

    import pcrsaits
    from pcrsaits import CSDIBackbone, PCRCSDI, build_windows

    source_file = Path(pcrsaits.__file__).resolve()
    if a.forbid_source_root:
        forbidden = Path(a.forbid_source_root).resolve()
        try:
            inside = source_file.is_relative_to(forbidden)
        except AttributeError:
            inside = str(source_file).startswith(str(forbidden))
        if inside:
            raise RuntimeError(f"Imported local source instead of installed wheel: {source_file}")

    dist = md.distribution("pcrsaits")
    dist_version = dist.version
    dist_root = Path(dist.locate_file("")).resolve()
    site_roots = []
    try:
        site_roots.extend(Path(x).resolve() for x in site.getsitepackages())
    except Exception:
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        site_roots.append(Path(purelib).resolve())
    # Preserve order while removing duplicates.
    site_roots = list(dict.fromkeys(site_roots))

    source_under_site_packages = any(_under(source_file, root) for root in site_roots)
    source_under_distribution_root = _under(source_file, dist_root)
    if not source_under_site_packages or not source_under_distribution_root:
        raise RuntimeError(
            "pcrsaits import is not proven to come from this clean environment's "
            f"site-packages: source={source_file}, sys.prefix={Path(sys.prefix).resolve()}, "
            f"site_roots={site_roots}, distribution_root={dist_root}"
        )

    seed = 7
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    train = rng.normal(size=(48, 2))
    val = rng.normal(size=(24, 2))
    test = rng.normal(size=(16, 2))
    kwargs = dict(
        n_steps=4, n_features=2, epochs=1, batch_size=2, patience=1,
        n_layers=1, n_heads=1, n_channels=8,
        d_time_embedding=8, d_feature_embedding=4, d_diffusion_embedding=8,
        n_diffusion_steps=2, n_sampling_times=2, aggregation="median",
        sampling_seed=seed, verbose=False,
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
    s1 = backbone.sample(masked_w)
    s2 = backbone.sample(masked_w)

    checks = {
        "installed_version_ok": dist_version == a.expected_version,
        "sample_shape_ok": list(s1.shape) == [4, 2, 4, 2],
        "seeded_sampling_deterministic": bool(np.array_equal(s1, s2)),
    }

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bp = td / "csdi.pypots"
        backbone.save(bp)
        restored_backbone = CSDIBackbone.load_from_checkpoint(bp, **kwargs)
        checks["csdi_checkpoint_roundtrip_equal"] = bool(
            np.array_equal(s1, restored_backbone.sample(masked_w))
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
            train, val, seed=seed, holdout_ratio=0.15,
            pointwise_fraction=0.5, block_patterns=(1, 2, 3), block_buffer=1,
        )
        out1 = pcr.impute(masked)
        observed = np.isfinite(masked)
        checks["observed_preserved"] = bool(np.array_equal(out1[observed], masked[observed]))
        checks["finite_missing_outputs"] = bool(np.isfinite(out1[~observed]).all())

        pp = td / "pcrcsdi.pt"
        pcr.save(pp)
        restored_pcr = PCRCSDI.load(pp, backbone=restored_backbone)
        out2 = restored_pcr.impute(masked)
        checks["pcrcsdi_checkpoint_roundtrip_equal"] = bool(np.array_equal(out1, out2))

    report = {
        "schema_version": 1,
        "pcrsaits_distribution": dist_version,
        "pcrsaits_source_file": str(source_file),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "site_packages_roots": [str(x) for x in site_roots],
        "distribution_root": str(dist_root),
        "source_under_site_packages": source_under_site_packages,
        "source_under_distribution_root": source_under_distribution_root,
        "pypots": md.version("pypots"),
        "torch": torch.__version__,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
