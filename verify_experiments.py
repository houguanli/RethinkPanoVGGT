"""Verify independent copies and optionally run every suite and CUDA smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ARMS = {"A1_no_geora": "no_geora", "A2_no_camera_geora": "no_camera_geora",
        "A3_no_patch_geora": "no_patch_geora", "A4_random_patch_bank": "random_patch_bank"}


def source_files(root):
    return {str(p.relative_to(root)): p for p in root.rglob("*")
            if p.is_file() and not any(part in ("logs", "__pycache__") for part in p.relative_to(root).parts)}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    reference = source_files(ROOT / "Rethink_pano_new_exp_omega")
    canonical = yaml.safe_load(reference["configs/train.yaml"].read_text())
    report = {}
    for folder, variant in ARMS.items():
        directory = ROOT / folder
        files = source_files(directory)
        assert files.keys() == reference.keys(), f"File inventory mismatch: {folder}"
        for name, path in files.items():
            assert not path.is_symlink() and path.stat().st_ino != reference[name].stat().st_ino
            if name not in ("README.md", "configs/train.yaml"):
                assert path.read_bytes() == reference[name].read_bytes(), f"Unexpected difference: {folder}/{name}"
        config = yaml.safe_load(files["configs/train.yaml"].read_text())
        assert config["ablation"]["geora_ablation"] == variant
        config["ablation"]["geora_ablation"] = "full"
        assert config == canonical, f"Non-ablation hyperparameters differ: {folder}"
        report[folder] = {"independent_files": len(files), "variant": variant,
                          "config_sha256": hashlib.sha256(files["configs/train.yaml"].read_bytes()).hexdigest()}
        log_dir = directory / "logs/verification_20260929"
        if args.test or args.smoke:
            log_dir.mkdir(parents=True, exist_ok=True)
        if args.test:
            with (log_dir / "tests.log").open("w") as log:
                subprocess.run([sys.executable, "tests/run_tests.py"], cwd=directory,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report[folder]["tests"] = "passed"
        if args.smoke:
            with (log_dir / "smoke.log").open("w") as log:
                subprocess.run([sys.executable, "tests/smoke_ablation.py", "--device", args.device,
                                "--output-dir", str(log_dir / "smoke")], cwd=directory,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            report[folder]["smoke"] = json.loads((log_dir / "smoke/smoke_summary.json").read_text())
        print(f"{folder}: independent config/code verified; tests={args.test}, smoke={args.smoke}", flush=True)
    # A later layout-only check must not overwrite the expensive test evidence.
    report_name = "verification_20260929.json" if args.test or args.smoke else "layout_verification_20260929.json"
    output = ROOT / "logs" / report_name
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
