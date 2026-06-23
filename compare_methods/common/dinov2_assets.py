"""DINOv2 asset handling for PanoVGGT compare runs."""

from __future__ import annotations

import os
import shutil
import ssl
import urllib.request
from pathlib import Path


DINOV2_VITL14_REG4_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/"
    "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
)
SUPPORTED_DINOV2_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitl14_pretrain.pth",
)
SUPPORTED_DINOV2_SUFFIXES = {".safetensors", ".pth", ".pt", ".bin"}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_supported_dinov2_path(raw_path: str) -> Path | None:
    if not raw_path:
        return None

    path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if path.is_dir():
        for filename in SUPPORTED_DINOV2_FILENAMES:
            candidate = path / filename
            if candidate.is_file():
                return candidate.resolve()
        for suffix in SUPPORTED_DINOV2_SUFFIXES:
            matches = sorted(path.glob(f"*{suffix}"))
            if matches:
                return matches[0].resolve()
        return None

    if path.is_file() and path.suffix in SUPPORTED_DINOV2_SUFFIXES:
        return path.resolve()
    return None


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        tmp.unlink()

    context = ssl._create_unverified_context() if _truthy(os.environ.get("DINOV2_INSECURE_DOWNLOAD")) else None
    request_kwargs = {"context": context} if context is not None else {}
    with urllib.request.urlopen(url, **request_kwargs) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(target)


def prepare_panovggt_dinov2_asset(repo_root: Path, *, dry_run: bool = False) -> None:
    """Validate or download DINOv2 weights when PanoVGGT explicitly asks for them."""

    if not _truthy(os.environ.get("PANOVGGT_LOAD_DINOV2_PRETRAINED", "false")):
        return

    weights_path = os.environ.get("DINOV2_WEIGHTS_PATH", "")
    resolved = _resolve_supported_dinov2_path(weights_path)
    if resolved is not None:
        os.environ["DINOV2_WEIGHTS_PATH"] = str(resolved)
        print(f"[asset] PanoVGGT DINOv2 checkpoint: {weights_path} -> {resolved}", flush=True)
        return

    allow_download = _truthy(os.environ.get("DINOV2_ALLOW_DOWNLOAD", "false"))
    if weights_path and not allow_download:
        raise SystemExit(
            "DINOV2_WEIGHTS_PATH was set but no supported DINOv2 weight file was found: "
            f"{weights_path}. Point it at a .pth/.pt/.bin/.safetensors file or a directory "
            "containing model.safetensors, pytorch_model.bin, or dinov2_vitl14_reg4_pretrain.pth. "
            "Set DINOV2_ALLOW_DOWNLOAD=1 to let the pipeline fetch the official DINOv2 ViT-L/14 reg4 weights."
        )

    if not allow_download:
        raise SystemExit(
            "PANOVGGT_LOAD_DINOV2_PRETRAINED=1 requires a valid DINOV2_WEIGHTS_PATH, "
            "or set DINOV2_ALLOW_DOWNLOAD=1 to download the official DINOv2 ViT-L/14 reg4 weights."
        )

    raw_cache_dir = os.environ.get("DINOV2_CACHE_DIR", "")
    cache_dir = (
        Path(os.path.expandvars(os.path.expanduser(raw_cache_dir)))
        if raw_cache_dir
        else repo_root / "ckpt" / "PanoVGGT"
    )
    target = cache_dir / "dinov2_vitl14_reg4_pretrain.pth"
    url = os.environ.get("DINOV2_URL", DINOV2_VITL14_REG4_URL)

    if target.exists():
        os.environ["DINOV2_WEIGHTS_PATH"] = str(target.resolve())
        print(f"[asset] PanoVGGT DINOv2 checkpoint: {target.resolve()}", flush=True)
        return

    if dry_run:
        print(f"[asset] PanoVGGT DINOv2 checkpoint missing; would download {url} -> {target}", flush=True)
        return

    print(f"[asset] PanoVGGT DINOv2 checkpoint missing; downloading {url} -> {target}", flush=True)
    try:
        _download(url, target)
    except Exception as exc:
        raise SystemExit(
            f"Failed to download DINOv2 weights from {url}: {exc}. "
            "If your server uses a self-signed TLS proxy, either pre-download the file and set "
            "DINOV2_WEIGHTS_PATH, or retry with DINOV2_INSECURE_DOWNLOAD=1."
        ) from exc

    os.environ["DINOV2_WEIGHTS_PATH"] = str(target.resolve())
    print(f"[asset] PanoVGGT DINOv2 checkpoint: {target.resolve()}", flush=True)
