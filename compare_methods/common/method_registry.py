"""Method registry used by compare-method orchestration scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = REPO_ROOT / "compare_methods"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: Path
    env_name: str
    requirements: List[Path]
    editable: bool = False
    config: Path = Path()
    finetune_command: Optional[List[str]] = None
    evaluate_command: Optional[List[str]] = None
    supports_native_finetune: bool = True


METHODS: Dict[str, MethodSpec] = {
    "panovggt_camera": MethodSpec(
        name="panovggt_camera",
        path=COMPARE_ROOT / "camera_pose" / "PanoVGGT",
        env_name="cmp_panovggt",
        requirements=[COMPARE_ROOT / "camera_pose" / "PanoVGGT" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panovggt_camera_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "training/launch.py", "--config", "panocity_4rtx5000"],
        evaluate_command=["python", "inference.py", "--help"],
    ),
    "panovggt_depth": MethodSpec(
        name="panovggt_depth",
        path=COMPARE_ROOT / "depth_geometry" / "PanoVGGT",
        env_name="cmp_panovggt",
        requirements=[COMPARE_ROOT / "depth_geometry" / "PanoVGGT" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panovggt_depth_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "training/launch.py", "--config", "panocity_4rtx5000"],
        evaluate_command=["python", "inference.py", "--help"],
    ),
    "reloc3r": MethodSpec(
        name="reloc3r",
        path=COMPARE_ROOT / "camera_pose" / "Reloc3r",
        env_name="cmp_reloc3r",
        requirements=[
            COMPARE_ROOT / "camera_pose" / "Reloc3r" / "requirements.txt",
            COMPARE_ROOT / "camera_pose" / "Reloc3r" / "requirements_optional.txt",
        ],
        config=COMPARE_ROOT / "configs" / "reloc3r_panocity_4rtx5000.yaml",
        finetune_command=["python", "finetune_panocity.py"],
        evaluate_command=["python", "evaluate_panocity.py"],
    ),
    "vggt_omega_camera": MethodSpec(
        name="vggt_omega_camera",
        path=COMPARE_ROOT / "camera_pose" / "VGGT-Omega",
        env_name="cmp_vggt_omega",
        requirements=[COMPARE_ROOT / "camera_pose" / "VGGT-Omega" / "requirements.txt"],
        editable=True,
        config=COMPARE_ROOT / "configs" / "vggt_omega_camera_panocity_4rtx5000.yaml",
        finetune_command=["python", "finetune_panocity.py"],
        evaluate_command=["python", "evaluate_panocity.py"],
        supports_native_finetune=False,
    ),
    "vggt_omega_depth": MethodSpec(
        name="vggt_omega_depth",
        path=COMPARE_ROOT / "depth_geometry" / "VGGT-Omega",
        env_name="cmp_vggt_omega",
        requirements=[COMPARE_ROOT / "depth_geometry" / "VGGT-Omega" / "requirements.txt"],
        editable=True,
        config=COMPARE_ROOT / "configs" / "vggt_omega_depth_panocity_4rtx5000.yaml",
        finetune_command=["python", "finetune_panocity.py"],
        evaluate_command=["python", "evaluate_panocity.py"],
        supports_native_finetune=False,
    ),
    "dap": MethodSpec(
        name="dap",
        path=COMPARE_ROOT / "depth_geometry" / "DAP",
        env_name="cmp_dap",
        requirements=[COMPARE_ROOT / "depth_geometry" / "DAP" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "dap_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "train_panocity.py", "--config", "config/train_panocity_4rtx5000.yaml", "--output-dir", "outputs/panocity_4rtx5000"],
        evaluate_command=["python", "evaluate_panocity.py"],
    ),
    "panda": MethodSpec(
        name="panda",
        path=COMPARE_ROOT / "depth_geometry" / "PanDA",
        env_name="cmp_panda",
        requirements=[COMPARE_ROOT / "depth_geometry" / "PanDA" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panda_panocity_4rtx5000.yaml",
        finetune_command=["python", "train_metric_depth/train.py", "--config", "config/metric_depth/train_panocity_4rtx5000.yaml", "--name", "panocity_4rtx5000", "--gpu", "0,1,2,3"],
        evaluate_command=["python", "evaluate_panocity.py"],
    ),
}


def method_names() -> List[str]:
    return sorted(METHODS)


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown method '{name}'. Available: {', '.join(method_names())}") from exc
