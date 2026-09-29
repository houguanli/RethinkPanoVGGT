# Verification — 2026-09-29

Upstream source: multipano-work-pro@3ea1294d82d702279719a744fa3d5d26016f964a.
Local environment: Ubuntu-22.04 WSL, existing RethinkPanoVGGT_omega conda,
one NVIDIA RTX 4090. No packages were installed.

| Experiment | Regression tests | CUDA synthetic smoke | Updated active parameter tensors |
|---|---|---|---:|
| Full reference | 103 passed | completed before test-initialization hardening | — |
| A1 no GeoRA | 103 passed | passed | 934 |
| A2 no Camera GeoRA | 103 passed | passed | 955 |
| A3 no Patch GeoRA | 103 passed | passed | 941 |
| A4 random Patch Bank | 103 passed | passed | 962 |

Every A1–A4 smoke run performed 3 small-Omega optimizer updates, teacher
checkpoint save/reload, one completion-main update, one refinement update,
completion save/reload and learned ERP metric calculation. Test-only dimensions
were reduced, but the 6 yaw x 2 pitch window layout was retained. Explicit test
initialization supplies the attention bias-mask buffers otherwise loaded from
foundation weights. These are NOT accuracy measurements.

Automated assertions cover:

- every disabled adapter parameter remains exactly zero and requires_grad=False
  after nonzero checkpoint loading, trainability-stage transitions and updates;
- disabled adapters are exact identity maps, with no gradients;
- random bank vectors are independent of input features, not trainable, same
  within an ERP cell, distinct across panoramas, repeatable by seed, and do not
  advance the global RNG;
- warmup skipping, checkpoint handoff, auto-eval trigger, no eval after training
  failure, propagated eval error code, and explicit auto-eval opt-out;
- two-argument evaluation correctly chooses the completion checkpoint's Omega
  teacher and preserves native window geometry;
- all four folders have independent physical files; all non-ablation configs
  and source files match the Full reference (README and variant setting aside).

Local evidence:
`logs/verification_20260929.json` and each experiment's
`logs/verification_20260929/{tests.log,smoke.log,smoke/smoke_summary.json}`.
These logs and test checkpoints are ignored by Git.

No 4-GPU RTX 5000 job, long training, or full dataset evaluation was launched
locally. Server readiness was checked via configs, command plans and tests;
server filesystem availability cannot be verified from this WSL host.
Historical no-camera checkpoint and merged summary remain in their original
Rethink_pano_new_exp_omega/logs directory.
