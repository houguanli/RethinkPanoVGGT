# A1_no_geora

Source: multipano-work-pro@3ea1294, 4 x RTX 5000, 2h+8h+2h ERP completion.
All Patch AND Camera GeoRA parameters are zero and frozen.
Historical ablation table values do not apply to this new experiment.

All code is local to this directory. Edit configs/train.yaml for model/loss/
sampling; configs/pipeline.yaml for server paths, four GPUs, duration and eval.

```bash
bash run.sh --run-dir logs/run_001
bash run.sh --warmup-checkpoint /path/last.pt --run-dir logs/run_002
bash run.sh --no-auto-eval --run-dir logs/run_003
python scripts/run_ablation_full_eval.py CKPT OUTPUT_DIR
python tests/run_tests.py
python tests/smoke_ablation.py --device cuda --output-dir logs/smoke_check
```

The final completion checkpoint resolves its Omega teacher automatically. Keep
all stage output folders together. Do not overwrite existing run directories.
