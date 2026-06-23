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
    smoke_finetune_command: Optional[List[str]] = None
    smoke_evaluate_command: Optional[List[str]] = None
    supports_native_finetune: bool = True


@dataclass(frozen=True)
class MethodGroup:
    name: str
    env_name: str
    finetune_method: str
    evaluate_methods: List[str]


METHODS: Dict[str, MethodSpec] = {
    "panovggt_camera": MethodSpec(
        name="panovggt_camera",
        path=COMPARE_ROOT / "camera_pose" / "PanoVGGT",
        env_name="cmp_panovggt",
        requirements=[COMPARE_ROOT / "camera_pose" / "PanoVGGT" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panovggt_camera_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "training/launch.py", "--config", "panocity_4rtx5000"],
        evaluate_command=[
            "python", "point_optimize_panocity.py",
            "--config", "panocity_4rtx5000",
            "--checkpoint", "logs/panocity_4rtx5000/ckpts/checkpoint.pt",
            "--split", "test",
            "--max-samples", "32",
            "--output-dir", "outputs/panocity_4rtx5000/point_opt",
        ],
        smoke_finetune_command=["torchrun", "--standalone", "--nproc_per_node=1", "training/launch.py", "--config", "panocity_smoke"],
        smoke_evaluate_command=[
            "python", "point_optimize_panocity.py",
            "--config", "panocity_smoke",
            "--checkpoint", "outputs/panocity_deep_smoke/ckpts/checkpoint.pt",
            "--split", "smoke",
            "--max-samples", "2",
            "--output-dir", "outputs/panocity_deep_smoke/point_opt",
        ],
    ),
    "panovggt_depth": MethodSpec(
        name="panovggt_depth",
        path=COMPARE_ROOT / "depth_geometry" / "PanoVGGT",
        env_name="cmp_panovggt",
        requirements=[COMPARE_ROOT / "depth_geometry" / "PanoVGGT" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panovggt_depth_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "training/launch.py", "--config", "panocity_4rtx5000"],
        evaluate_command=[
            "python", "point_optimize_panocity.py",
            "--config", "panocity_4rtx5000",
            "--checkpoint", "../../camera_pose/PanoVGGT/logs/panocity_4rtx5000/ckpts/checkpoint.pt",
            "--split", "test",
            "--max-samples", "32",
            "--output-dir", "outputs/panocity_4rtx5000/point_opt",
        ],
        smoke_finetune_command=["torchrun", "--standalone", "--nproc_per_node=1", "training/launch.py", "--config", "panocity_smoke"],
        smoke_evaluate_command=[
            "python", "point_optimize_panocity.py",
            "--config", "panocity_smoke",
            "--checkpoint", "../../camera_pose/PanoVGGT/outputs/panocity_deep_smoke/ckpts/checkpoint.pt",
            "--split", "smoke",
            "--max-samples", "2",
            "--output-dir", "outputs/panocity_deep_smoke/point_opt",
        ],
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
        finetune_command=[
            "torchrun", "--standalone", "--nproc_per_node=4", "train.py",
            "--train_dataset", "PanoCityReloc3r(split='train', resolution=512)",
            "--test_dataset", "PanoCityReloc3r(split='test', resolution=512)",
            "--epochs", "20", "--batch_size", "8", "--num_workers", "8",
            "--pretrained", "../../../ckpt/Reloc3r-512/Reloc3r-512.pth",
            "--output_dir", "outputs/panocity_4rtx5000",
        ],
        evaluate_command=["python", "evaluate_panocity.py"],
        smoke_finetune_command=[
            "python", "train.py",
            "--train_dataset", "PanoCityReloc3r(split='smoke', resolution=512, max_samples=8)",
            "--test_dataset", "PanoCityReloc3r(split='smoke', resolution=512, max_samples=4)",
            "--epochs", "1", "--batch_size", "1", "--num_workers", "0",
            "--eval_freq", "0", "--save_freq", "1", "--keep_freq", "0", "--print_freq", "1",
            "--max_steps", "2",
            "--pretrained", "../../../ckpt/Reloc3r-512/Reloc3r-512.pth",
            "--output_dir", "outputs/panocity_deep_smoke",
        ],
        smoke_evaluate_command=["python", "evaluate_panocity.py", "--stage", "smoke"],
    ),
    "vggt_omega_camera": MethodSpec(
        name="vggt_omega_camera",
        path=COMPARE_ROOT / "camera_pose" / "VGGT-Omega",
        env_name="cmp_vggt_omega",
        requirements=[COMPARE_ROOT / "camera_pose" / "VGGT-Omega" / "requirements.txt"],
        editable=True,
        config=COMPARE_ROOT / "configs" / "vggt_omega_camera_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "train_panocity.py", "--config", "../../configs/vggt_omega_camera_panocity_4rtx5000.yaml", "--checkpoint", "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt", "--output-dir", "outputs/panocity_4rtx5000"],
        evaluate_command=["python", "evaluate_panocity.py"],
        smoke_finetune_command=["python", "train_panocity.py", "--config", "../../configs/vggt_omega_camera_panocity_4rtx5000.yaml", "--checkpoint", "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt", "--output-dir", "outputs/panocity_deep_smoke", "--smoke", "--max-steps", "2"],
        smoke_evaluate_command=["python", "evaluate_panocity.py", "--stage", "smoke"],
    ),
    "vggt_omega_depth": MethodSpec(
        name="vggt_omega_depth",
        path=COMPARE_ROOT / "depth_geometry" / "VGGT-Omega",
        env_name="cmp_vggt_omega",
        requirements=[COMPARE_ROOT / "depth_geometry" / "VGGT-Omega" / "requirements.txt"],
        editable=True,
        config=COMPARE_ROOT / "configs" / "vggt_omega_depth_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "train_panocity.py", "--config", "../../configs/vggt_omega_depth_panocity_4rtx5000.yaml", "--checkpoint", "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt", "--output-dir", "outputs/panocity_4rtx5000"],
        evaluate_command=["python", "evaluate_panocity.py"],
        smoke_finetune_command=["python", "train_panocity.py", "--config", "../../configs/vggt_omega_depth_panocity_4rtx5000.yaml", "--checkpoint", "../../../ckpt/VGGT-Omega/vggt_omega_1b_512.pt", "--output-dir", "outputs/panocity_deep_smoke", "--smoke", "--max-steps", "2"],
        smoke_evaluate_command=["python", "evaluate_panocity.py", "--stage", "smoke"],
    ),
    "dap": MethodSpec(
        name="dap",
        path=COMPARE_ROOT / "depth_geometry" / "DAP",
        env_name="cmp_dap",
        requirements=[COMPARE_ROOT / "depth_geometry" / "DAP" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "dap_panocity_4rtx5000.yaml",
        finetune_command=["torchrun", "--standalone", "--nproc_per_node=4", "train_panocity.py", "--config", "config/train_panocity_4rtx5000.yaml", "--output-dir", "outputs/panocity_4rtx5000"],
        evaluate_command=["python", "evaluate_panocity.py"],
        smoke_finetune_command=["python", "train_panocity.py", "--config", "config/train_panocity_4rtx5000.yaml", "--output-dir", "outputs/panocity_deep_smoke", "--smoke", "--max-steps", "2"],
        smoke_evaluate_command=["python", "evaluate_panocity.py", "--stage", "smoke"],
    ),
    "panda": MethodSpec(
        name="panda",
        path=COMPARE_ROOT / "depth_geometry" / "PanDA",
        env_name="cmp_panda",
        requirements=[COMPARE_ROOT / "depth_geometry" / "PanDA" / "requirements.txt"],
        config=COMPARE_ROOT / "configs" / "panda_panocity_4rtx5000.yaml",
        finetune_command=["python", "train_metric_depth/train.py", "--config", "config/metric_depth/train_panocity_4rtx5000.yaml", "--name", "panocity_4rtx5000", "--gpu", "0,1,2,3"],
        evaluate_command=["python", "evaluate_panocity.py"],
        smoke_finetune_command=["python", "train_metric_depth/train.py", "--config", "config/metric_depth/train_panocity_4rtx5000.yaml", "--name", "panocity_deep_smoke", "--gpu", "0", "--smoke", "--max-steps", "2", "--output-dir", "outputs/panocity_deep_smoke"],
        smoke_evaluate_command=["python", "evaluate_panocity.py", "--stage", "smoke"],
    ),
}


GROUPS: Dict[str, MethodGroup] = {
    "panovggt": MethodGroup(
        name="panovggt",
        env_name="cmp_panovggt",
        finetune_method="panovggt_camera",
        evaluate_methods=["panovggt_camera", "panovggt_depth"],
    ),
    "reloc3r": MethodGroup(
        name="reloc3r",
        env_name="cmp_reloc3r",
        finetune_method="reloc3r",
        evaluate_methods=["reloc3r"],
    ),
    "vggt_omega": MethodGroup(
        name="vggt_omega",
        env_name="cmp_vggt_omega",
        finetune_method="vggt_omega_camera",
        evaluate_methods=["vggt_omega_camera", "vggt_omega_depth"],
    ),
    "dap": MethodGroup(
        name="dap",
        env_name="cmp_dap",
        finetune_method="dap",
        evaluate_methods=["dap"],
    ),
    "panda": MethodGroup(
        name="panda",
        env_name="cmp_panda",
        finetune_method="panda",
        evaluate_methods=["panda"],
    ),
}

ALIASES: Dict[str, str] = {
    "panovggtcamera": "panovggt_camera",
    "panovggtdepth": "panovggt_depth",
    "relo3r": "reloc3r",
    "vggtomega": "vggt_omega",
    "vggtomegacamera": "vggt_omega_camera",
    "vggtomegadepth": "vggt_omega_depth",
}


def normalize_name(name: str) -> str:
    return ALIASES.get(name, name)


def method_names() -> List[str]:
    return sorted(METHODS)


def group_names() -> List[str]:
    return sorted(GROUPS)


def runnable_names() -> List[str]:
    return sorted(set(method_names()) | set(group_names()) | set(ALIASES))


def get_method(name: str) -> MethodSpec:
    name = normalize_name(name)
    try:
        return METHODS[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown method '{name}'. Available: {', '.join(method_names())}") from exc


def get_group(name: str) -> MethodGroup:
    name = normalize_name(name)
    try:
        return GROUPS[name]
    except KeyError as exc:
        raise SystemExit(f"Unknown method group '{name}'. Available: {', '.join(group_names())}") from exc


def is_group(name: str) -> bool:
    name = normalize_name(name)
    return name in GROUPS
