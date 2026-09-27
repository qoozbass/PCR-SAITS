from __future__ import annotations
import argparse
import importlib.metadata as md
import inspect
import json
import platform
import sys
from pathlib import Path


def version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    a = p.parse_args()

    import torch
    from pypots.imputation import CSDI

    sig = inspect.signature(CSDI)
    required_surface = {
        "n_steps", "n_features", "n_layers", "n_heads", "n_channels",
        "d_time_embedding", "d_feature_embedding", "d_diffusion_embedding",
        "n_diffusion_steps", "target_strategy", "is_unconditional", "schedule",
        "beta_start", "beta_end", "batch_size", "epochs", "patience", "device",
        "saving_path", "verbose",
    }
    missing = sorted(required_surface - set(sig.parameters))
    tested = version("pypots")
    report = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": version("numpy"),
        "torch": version("torch"),
        "pypots": tested,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "csdi_public_import": True,
        "csdi_constructor_missing_parameters": missing,
        "api_surface_ok": not missing,
        "tested_pypots_version": tested,
        "conservative_minimum_candidate": tested,
        "minimum_candidate_note": (
            "This value is the PyPOTS version actually proven by this run. "
            "It is a conservative release-floor candidate, not evidence that older versions fail."
        ),
        "status": "PASS" if tested and not missing else "FAIL",
    }
    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
