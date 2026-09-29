import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from training import train_pano_omega as train
from training.matched_warmup import validate_start, validate_teacher, input_metadata
from vggt_omega.data.pano_sampler import PanoWindowSampler


class MatchedWarmupTest(unittest.TestCase):
    def args(self, arm):
        config = ("multipano_4090_mixed4_omega_canonical_warmup_2h.yaml" if arm == "A"
                  else "multipano_4090_mixed4_belt60_completion.yaml")
        args = train.parse_args(["--config", f"tests/fixtures/{config}", "--matched-warmup-arm", arm])
        args.checkpoint = args.base_checkpoint = Path("foundation.pt")
        return args

    def test_both_configs_and_stages_match_their_arm(self):
        for arm in ("A", "B"):
            validate_start(self.args(arm), {})

    def test_rejects_runtime_stage_reverting_b_to_canonical(self):
        args = self.args("B")
        args.training_stages[0]["pitch_degrees"] = "-15"
        with self.assertRaises(ValueError):
            validate_start(args, {})

    def test_rejects_old_warmup_as_foundation_or_teacher(self):
        args = self.args("B")
        with self.assertRaises(ValueError):
            validate_start(args, {"model_delta": {}})
        with self.assertRaises(ValueError):
            validate_teacher(args, {"args": vars(self.args("A")), "step": 1000}, "B")

    def test_teacher_must_finish_real_two_hour_budget(self):
        args = self.args("B")
        payload = {"args": vars(args), "training_status": {
            "state": "completed", "stop_reason": "max_steps", "elapsed_seconds": 7201}}
        with self.assertRaises(ValueError):
            validate_teacher(args, payload, "B")
        payload["training_status"]["stop_reason"] = "max_duration"
        validate_teacher(args, payload, "B")

    def test_b_actual_window_metadata_is_24_views(self):
        torch.set_num_threads(2)
        sampled = PanoWindowSampler(window_size=384, num_yaw=6, pitch_degrees=[-25, 25], fov_degrees=75)(torch.zeros(2, 3, 8, 16))
        prediction = {"pano_windows": sampled.windows.reshape(1, 24, 3, 384, 384),
                      "pano_camera_meta": sampled.camera_meta}
        info = input_metadata(prediction, torch.zeros(1, 2, 3, 8, 16), self.args("B"))
        self.assertEqual(info["window_shape"][1], 24)
        self.assertEqual(info["degrees"]["pitch"], [-25, 25])

    def test_failed_update_never_writes_success_last(self):
        class Dataset(torch.utils.data.Dataset):
            root = Path("unused")
            def __len__(self): return 1
            def __getitem__(self, index): return {"pano_image": torch.zeros(2, 3, 8, 16)}
        with tempfile.TemporaryDirectory() as tmp:
            args = train.parse_args([])
            args.device, args.distributed = "cpu", "off"
            args.checkpoint = args.base_checkpoint = None
            args.output_dir = Path(tmp)
            args.epochs = args.max_steps = 1
            args.num_workers = 0
            args.trainable = "all"
            args.progress_bar = args.tensorboard = False
            args.save_last = True
            with patch.object(train, "build_dataset", return_value=Dataset()), patch.object(train, "build_model", return_value=torch.nn.Linear(2, 2)), patch.object(train, "train_step", side_effect=RuntimeError("injected backward failure")):
                with self.assertRaisesRegex(RuntimeError, "injected backward"):
                    train.train(args)
            self.assertFalse((Path(tmp) / "last.pt").exists())
            self.assertTrue((Path(tmp) / "interrupted.pt").exists())
            status = json.loads((Path(tmp) / "status.json").read_text())
            self.assertEqual((status["state"], status["step"]), ("failed", 0))

    def test_no_camera_pairs_are_missing(self):
        from scripts.evaluate_depth_checkpoint import empty_camera_pose_sample_metrics
        metrics = empty_camera_pose_sample_metrics()
        self.assertEqual(metrics["camera_pose_pair_count"], 0)
        self.assertTrue(torch.isnan(torch.tensor(metrics["camera_pose_auc5"])))

    def test_no_gt_depth_is_missing(self):
        from scripts.evaluate_depth_checkpoint import compute_depth_metrics
        depth = torch.ones(1, 4, 8)
        metrics = compute_depth_metrics(depth, depth, torch.zeros_like(depth, dtype=torch.bool))
        self.assertEqual(metrics["depth_valid_pixels"], 0)
        self.assertTrue(torch.isnan(torch.tensor(metrics["depth_irls_abs_rel"])))

    def test_eval_rejects_cross_arm_teacher_but_keeps_legacy(self):
        from vggt_omega.models.erp_completion import validate_completion_sampler_args
        args = self.args("B")
        validate_completion_sampler_args(args, {}, Path("legacy.pt"))
        payload = {"matched_warmup_arm": "B", "omega_checkpoint": "B.pt"}
        with self.assertRaises(ValueError):
            validate_completion_sampler_args(args, payload, Path("A.pt"))

    def test_update_audit_detects_trainable_changes_and_fixed_initializer(self):
        from training.matched_warmup import capture_update_paths, finish_update_paths
        model = torch.nn.Module()
        model.dense_head = torch.nn.Linear(2, 2)
        model.pano_camera_head = torch.nn.Linear(2, 2)
        model.camera_head = torch.nn.Linear(2, 2).requires_grad_(False)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        sum(p.sum() for p in model.parameters() if p.requires_grad).backward()
        captured = capture_update_paths(model, optimizer)
        optimizer.step()
        result = finish_update_paths(model, optimizer, captured)
        self.assertTrue(result["fixed_camera_unchanged"])
        self.assertGreater(result["paths"]["dense_head."]["update_abs_max"], 0)
        self.assertGreater(result["paths"]["pano_camera_head."]["gradient_abs_max"], 0)

    def test_checkpoint_covers_dino_and_preserves_outputs_and_gradients(self):
        from vggt_omega.models.aggregator import Aggregator
        from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer, init_weights_vit
        torch.manual_seed(57)
        encoder = DinoVisionTransformer(patch_size=16, embed_dim=32, depth=2, num_heads=4,
                                        n_storage_tokens=1, drop_path_rate=0)
        encoder.init_weights()
        with patch("vggt_omega.models.aggregator._build_patch_embed", return_value=encoder):
            plain = Aggregator(patch_size=16, embed_dim=32, depth=2, num_heads=4,
                               num_register_tokens=1, register_attention_block_indices=(),
                               cached_layer_indices=(0, 1), enable_luna=False,
                               luna_patch_layers="none", luna_camera_layers="none",
                               enable_pano_global_token=False)
        plain.apply(init_weights_vit)
        plain.rope_embed._init_weights()
        checked = copy.deepcopy(plain)
        checked.use_checkpoint = True
        calls = []
        checked.patch_embed.blocks[0].register_forward_pre_hook(lambda *unused: calls.append(1))
        images = torch.rand(1, 4, 3, 32, 32)
        def forward_with_saved_activation_bytes(model):
            parameters = {p.untyped_storage().data_ptr() for p in model.parameters()}
            saved = {}
            def pack(tensor):
                storage = tensor.untyped_storage()
                if storage.data_ptr() not in parameters:
                    saved[storage.data_ptr()] = storage.nbytes()
                return tensor
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                outputs, _ = model(images)
            return outputs, sum(saved.values())
        baseline, plain_bytes = forward_with_saved_activation_bytes(plain)
        actual, checked_bytes = forward_with_saved_activation_bytes(checked)
        self.assertLess(checked_bytes, plain_bytes)
        print(f"[CHECKPOINT ACTIVATIONS] tiny DINO+aggregator saved non-parameter storage: {plain_bytes} -> {checked_bytes} bytes")
        for left, right in zip(baseline, actual):
            torch.testing.assert_close(left, right)
        sum(x.square().mean() for x in baseline).backward()
        sum(x.square().mean() for x in actual).backward()
        self.assertGreaterEqual(len(calls), 2, "DINO was not recomputed during backward")
        for (name, p), (other_name, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertEqual(name, other_name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad)

    def test_camera_slice_before_float_preserves_outputs_and_gradients(self):
        from vggt_omega.models.heads.camera_head import CameraHead
        from vggt_omega.models.layers.vision_transformer import init_weights_vit
        head = CameraHead(dim_in=32).apply(init_weights_vit)
        other = copy.deepcopy(head)
        x = torch.randn(1, 4, 20, 32, dtype=torch.bfloat16, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        old = head([x.float()], patch_token_start=3)
        new = other([y], patch_token_start=3)
        torch.testing.assert_close(old, new)
        old.square().mean().backward()
        new.square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad)
        for p, q in zip(head.parameters(), other.parameters()):
            torch.testing.assert_close(p.grad, q.grad)

    def test_unused_fixed_pose_graph_preserves_all_returned_gradients(self):
        from vggt_omega.models.layers.vision_transformer import init_weights_vit
        args = train.parse_args([])
        args.smoke, args.window_size, args.num_yaw = True, 32, 2
        model = train.build_model(args).apply(init_weights_vit)
        model.aggregator.rope_embed._init_weights()
        model.camera_head.requires_grad_(False)
        other = copy.deepcopy(model)
        grad_modes = []
        other.camera_head.register_forward_pre_hook(lambda *unused: grad_modes.append(torch.is_grad_enabled()))
        images = torch.rand(1, 2, 3, 32, 64)
        old = model(pano_images=images, return_window_pose=False, window_camera_grad_enabled=True)
        new = other(pano_images=images, return_window_pose=False)
        self.assertFalse(grad_modes[-1])
        for key in old:
            if torch.is_tensor(old[key]):
                torch.testing.assert_close(old[key], new[key])
        (old['depth'].square().mean() + old['pano_pose_enc'].square().mean()).backward()
        (new['depth'].square().mean() + new['pano_pose_enc'].square().mean()).backward()
        for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad, name)
            else:
                torch.testing.assert_close(p.grad, q.grad, msg=name)
        returned_pose = other(pano_images=images, return_window_pose=True)
        self.assertTrue(grad_modes[-1])
        self.assertTrue(returned_pose['pose_enc'].requires_grad)


if __name__ == "__main__":
    unittest.main()
