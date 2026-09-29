"""Real small-model train/save/reload/completion/eval, never a benchmark result."""

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.train_pano_omega import parse_args, build_model, configure_trainable
from training.train_erp_completion import completion_losses
from vggt_omega.models.erp_completion import ERPRemainingBandHead, splat_omega_window_depth_to_erp
from vggt_omega.models.layers.vision_transformer import init_weights_vit
from scripts.evaluate_depth_checkpoint import compute_erp_completion_depth_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    opts = parser.parse_args()
    output = opts.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Smoke output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(57)
    args = parse_args(["--config", str(ROOT / "configs/train.yaml"), "--smoke", "--window-size", "32"])
    # Keep the production 6 yaw x 2 pitch geometry; only model width/image size shrink.
    model = build_model(args).to(opts.device)
    model.apply(init_weights_vit)  # Includes checkpoint-owned attention bias masks.
    configure_trainable(model, "all")
    # Production expects foundation weights; initialize every small-model
    # tensor explicitly so empty pretrained-only allocations are never used.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim >= 2:
                torch.nn.init.normal_(parameter, std=0.02)
            elif name.endswith("weight"):
                parameter.fill_(1)
            else:
                parameter.zero_()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    images = torch.rand(1, 2, 3, 32, 64, device=opts.device)
    train_losses = []
    before_training = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    model.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(pano_images=images, return_sampler_output=True)
        loss = (prediction["depth"].float() - 2).square().mean()
        loss = loss + prediction["camera_and_register_tokens"].float().square().mean()
        if not torch.isfinite(loss):
            bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
            raise RuntimeError(f"Non-finite loss at step {len(train_losses)}; prior={train_losses}; bad={bad[:5]}")
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    updated_parameters = sum(not torch.equal(before_training[name], p)
                             for name, p in model.named_parameters() if name in before_training)
    assert updated_parameters > 0, "Optimizer did not update any active parameters"
    del before_training
    for module in model.modules():
        if getattr(module, "zero_frozen", False):
            assert all(not p.requires_grad and p.grad is None and p.count_nonzero() == 0 for p in module.parameters())
    teacher_path = output / "teacher.pt"
    torch.save({"model": model.state_dict(), "geora_ablation": args.geora_ablation}, teacher_path)
    reloaded = build_model(args).to(opts.device)
    reloaded.load_state_dict(torch.load(teacher_path, map_location=opts.device, weights_only=False)["model"])
    reloaded.eval()
    with torch.no_grad():
        prediction = reloaded(pano_images=images, return_sampler_output=True)
        splat = splat_omega_window_depth_to_erp(
            prediction["depth"], prediction["pano_camera_meta"], num_panos=2, erp_height=32, erp_width=64,
        )
    head = ERPRemainingBandHead(width=8).to(opts.device)
    head_optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
    rgb = images.reshape(2, 3, 32, 64)
    target = torch.ones(2, 1, 32, 64, device=opts.device) * 2
    head_losses = []
    for stage in ("main", "refine"):
        head_optimizer.zero_grad(set_to_none=True)
        completed = head(rgb, splat.depth, splat.valid_mask, blend_width_pixels=2)
        loss = completion_losses(completed, target, torch.ones_like(target, dtype=torch.bool),
                                 rgb, stage, 2, num_panos=2)["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        head_optimizer.step()
        head_losses.append(loss.item())
    completion_path = output / "completion.pt"
    torch.save({"completion_head": head.state_dict(), "omega_checkpoint": str(teacher_path),
                "geora_ablation": args.geora_ablation}, completion_path)
    head.load_state_dict(torch.load(completion_path, map_location=opts.device, weights_only=False)["completion_head"])
    head.eval()
    with torch.no_grad():
        metrics = compute_erp_completion_depth_metrics(
            pred_window_z=prediction["depth"], pano_rgb=images,
            gt_erp_depth=target.reshape(1, 2, 1, 32, 64), source_depth_semantics="range",
            max_range_depth=80, camera_meta=prediction["pano_camera_meta"], completion_head=head,
            completion_head_args={"height":32, "width_erp":64, "blend_width_pixels":2,
                                  "core_latitude_degrees":90},
            gt_common_mask=torch.ones(1, 2, 1, 32, 64, device=opts.device, dtype=torch.bool),
        )
    report = {"test_only": True, "device": opts.device, "variant": args.geora_ablation,
              "updated_parameter_tensors": updated_parameters,
              "warmup_losses": train_losses, "completion_main_refine_losses": head_losses,
              "teacher_checkpoint": str(teacher_path), "completion_checkpoint": str(completion_path),
              "metrics": {key: float(value) for key, value in metrics.items()}}
    (output / "smoke_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
