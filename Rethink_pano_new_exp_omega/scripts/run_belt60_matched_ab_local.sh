#!/usr/bin/env bash
# Matched foundation -> each arm's 2h full warm-up -> 8h head -> 2h refinement.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/aoki/miniconda3/envs/RethinkPanoVGGT_omega/bin/python}"
DATASET_ROOT="${PANOVGGT_ROOT:-/mnt/e/PanoVGGT_minimal_datasets/datasets}"
FOUNDATION="${FOUNDATION_CHECKPOINT:-/home/aoki/RethinkPanoVGGT_omega_compare_methods_only/ckpt/VGGT-Omega/vggt_omega_1b_512.pt}"
OUTPUT="$PROJECT_ROOT/logs/${RUN_NAME:-belt60_matched_foundation_ab_20260924}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 SKIP_INDEX_BUILD=1
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/pipeline.lock"
flock -n 9 || { echo 'Another launcher owns this run'; exit 2; }
exec > >(tee -a "$OUTPUT/pipeline.log") 2>&1
monitor_pid=""
cleanup() {
  local code=$?
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  if (( code != 0 )); then
    echo "[FAILED] exit=$code; no automatic training restart"
    printf '%s\n' "exit=$code phase=$(cat "$OUTPUT/current_phase.txt" 2>/dev/null || true)" > "$OUTPUT/FAILED"
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
[[ ! -e "$OUTPUT/COMPLETE" ]] || exit 0
[[ -s "$FOUNDATION" && -d "$DATASET_ROOT" ]]
phase() { printf '%s\n' "$*" | tee "$OUTPUT/current_phase.txt"; }
gpu_gate() {
  local compute
  compute="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "$compute" ]] || { echo "[BLOCKED] existing GPU compute PID(s): $compute"; return 2; }
  free -b
  nvidia-smi
}
check_status() {
  "$PYTHON_BIN" - "$1" "$2" "$3" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); mode=sys.argv[2]; minutes=float(sys.argv[3])
assert s['state']=='completed',s
if mode in ('preflight','stability'):
    assert s['step']==(3 if mode=='stability' else 1) and s['stop_reason']=='max_steps',s
    inputs=s['input_metadata']; arm='B' if 'preflight_B' in sys.argv[1] else 'A'
    assert inputs['pano_count']==2 and inputs['window_shape']==[1,24 if arm=='B' else 8,3,384,384],s
    assert inputs['degrees']['pitch']==([-25,25] if arm=='B' else [-15]),s
    assert inputs['degrees']['fov_x']==[75] and inputs['degrees']['fov_y']==[75],s
    if mode=='stability':
        records=s['update_path_audit']
        assert len(records)==3 and all(r['fixed_camera_unchanged'] for r in records),s
        assert len({r['fixed_camera_sha256'] for r in records})==1,s
        for prefix in ('aggregator.patch_embed.','aggregator.frame_blocks.','aggregator.inter_frame_blocks.','dense_head.','pano_camera_head.'):
            assert any(r['paths'].get(prefix,{}).get('update_abs_max',0)>0 for r in records),prefix
        import pathlib,torch
        checkpoint=pathlib.Path(sys.argv[1]).with_name('last.pt')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        tensors=payload['model_delta']
        assert payload['step']==3 and payload['args']['matched_warmup_arm']==arm
        assert sum(t.numel() for t in tensors.values())==s['trainable_parameters']
        assert all(bool(torch.isfinite(t).all()) for t in tensors.values()),'Nonfinite checkpoint tensor'
        integrity={'cpu_load':'passed','all_tensors_finite':True,'step':payload['step'],
                   'parameters':s['trainable_parameters'],'tensor_count':len(tensors),'bytes':checkpoint.stat().st_size}
        checkpoint.with_name('checkpoint_integrity.json').write_text(json.dumps(integrity,indent=2))
else:
    assert s['stop_reason'] in ('duration','max_duration'),s
    elapsed=s.get('elapsed_seconds',s.get('elapsed_minutes',0)*60)
    assert elapsed>=minutes*60,s
print('[VERIFIED]',sys.argv[1],s)
PY
}
fresh_stage() {
  [[ ! -e "$1/loss.csv" && ! -e "$1/interrupted.pt" && ! -e "$1/last.pt.tmp" && ! -e "$1/status.json" ]] || {
    echo "[BLOCKED] partial stage needs explicit inspection: $1"; return 2;
  }
  mkdir -p "$1"
}
if [[ ! -e "$OUTPUT/protocol.json" ]]; then
  cp configs/multipano_4090_mixed4_omega_canonical_warmup_2h.yaml "$OUTPUT/A.yaml"
  cp configs/multipano_4090_mixed4_belt60_completion.yaml "$OUTPUT/B.yaml"
  "$PYTHON_BIN" - "$OUTPUT" "$FOUNDATION" "$DATASET_ROOT" <<'PY'
import hashlib,json,pathlib,subprocess,sys,time
out=pathlib.Path(sys.argv[1]); foundation=pathlib.Path(sys.argv[2])
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
p={'protocol':'matched_foundation_belt60_v1','created_utc':time.strftime('%FT%TZ',time.gmtime()),
   'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
   'foundation':str(foundation),'foundation_sha256':digest(foundation),'dataset_root':sys.argv[3],
   'seed':57,'warmup_minutes':120,'main_minutes':480,'refine_minutes':120,
   'arm_order':['A','B'],'train_panos':2,'window_resolution':384,'fov':[75,75],
   'A':{'pitch':[-15],'num_yaw':4,'windows_per_pano':4,'trusted_latitude':90},
   'B':{'pitch':[-25,25],'num_yaw':6,'windows_per_pano':12,'trusted_latitude':60},
   'config_sha256':{arm:digest(out/f'{arm}.yaml') for arm in ('A','B')},
   'eval':{'sets':{'Stanford2D3DS':216,'Matterport3D':891,'Structured3D':1662,'Panocity':6064},
           'pano_policy':'panovggt','camera_max_panos':3,'alignment_domain':'canonical_pitch15_fov75x75'},
   'timing':'equal stage wall budgets; record optimizer steps, throughput and CUDA event step time separately',
   'historical_run':'belt60_completion_ab_20260924 is a frozen cross-sampling transfer diagnostic'}
(out/'protocol.json').write_text(json.dumps(p,indent=2))
PY
fi
"$PYTHON_BIN" - "$OUTPUT" "$FOUNDATION" "$DATASET_ROOT" <<'PY'
import hashlib,json,pathlib,sys
out=pathlib.Path(sys.argv[1]); p=json.loads((out/'protocol.json').read_text())
assert p['foundation']==sys.argv[2] and p['dataset_root']==sys.argv[3], 'Run provenance changed'
for arm in ('A','B'):
    assert hashlib.sha256((out/f'{arm}.yaml').read_bytes()).hexdigest()==p['config_sha256'][arm], 'Saved config changed'
with open(sys.argv[2],'rb') as f:
    assert hashlib.file_digest(f,'sha256').hexdigest()==p['foundation_sha256'], 'Foundation content changed'
PY
[[ "${PREPARE_ONLY:-0}" != 1 ]] || { phase PREPARED; exit 0; }
(
  while true; do
    date -u +%FT%TZ
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    free -b
    sleep 30
  done
) >> "$OUTPUT/resources.log" 2>&1 &
monitor_pid=$!

# Inspect the optimized complete B recheck before any stability/formal stage.
[[ -s "$OUTPUT/RESOURCE_RECHECK_ACCEPTED" ]] || {
  echo '[BLOCKED] optimized B resource recheck has not been accepted'; exit 2;
}
check_status "$OUTPUT/preflight_B_recheck/status.json" preflight 0
for arm in ${PREFLIGHT_ARMS:-B A}; do
  [[ "$arm" == A || "$arm" == B ]] || exit 2
  dest="$OUTPUT/preflight_${arm}_stability"
  phase "preflight_${arm}_stability"
  if [[ ! -s "$dest/status.json" ]]; then
    fresh_stage "$dest"
    gpu_gate
    "$PYTHON_BIN" training/train_pano_omega.py --config "$OUTPUT/$arm.yaml" \
      --dataset-root "$DATASET_ROOT" --checkpoint "$FOUNDATION" --base-checkpoint "$FOUNDATION" \
      --matched-warmup-arm "$arm" --no-inherit-checkpoint-training-defaults --verify-update-paths \
      --output-dir "$dest" --tensorboard-dir "$dest/tensorboard" --debug-dir "$dest/debug" \
      --max-steps 3 --max-duration-minutes 120 --num-workers 0 --save-last \
      --save-every-steps 2000 --no-progress-bar 2>&1 | tee -a "$dest/train.log"
  fi
  test -s "$dest/last.pt"
  check_status "$dest/status.json" stability 0
done
[[ "${PREFLIGHT_ONLY:-0}" != 1 ]] || { phase PREFLIGHT_COMPLETE; exit 0; }
[[ -s "$OUTPUT/FORMAL_TRAINING_ACCEPTED" ]] || { phase WAITING_FOR_FORMAL_ACCEPTANCE; exit 2; }
for arm in A B; do check_status "$OUTPUT/preflight_${arm}_stability/status.json" stability 0; done
phase test_index_audit
"$PYTHON_BIN" scripts/validate_eval_cardinality.py --config "$OUTPUT/A.yaml" \
  --dataset-root "$DATASET_ROOT" --index-audit "$OUTPUT/test_index_audit.json"

for arm in A B; do
  config="$OUTPUT/$arm.yaml"
  warm="$OUTPUT/$arm/warmup_2h"
  main="$OUTPUT/$arm/completion_8h"
  refine="$OUTPUT/$arm/refine_2h"
  cutoff=90; [[ "$arm" != B ]] || cutoff=60
  phase "$arm/warmup_2h"
  if [[ ! -s "$warm/last.pt" ]]; then
    fresh_stage "$warm"
    gpu_gate
    "$PYTHON_BIN" training/train_pano_omega.py --config "$config" \
      --dataset-root "$DATASET_ROOT" --checkpoint "$FOUNDATION" --base-checkpoint "$FOUNDATION" \
      --matched-warmup-arm "$arm" --no-inherit-checkpoint-training-defaults --seed 57 \
      --output-dir "$warm" --tensorboard-dir "$warm/tensorboard" --debug-dir "$warm/debug" \
      --max-duration-minutes 120 --max-steps 1000000 --num-workers 0 --save-every-steps 2000 \
      --progress-bar 2>&1 | tee -a "$warm/train.log"
  fi
  check_status "$warm/status.json" duration 120
  if [[ "$arm" == B ]]; then
    "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
a=json.loads((out/'A/warmup_2h/status.json').read_text())
b=json.loads((out/'B/warmup_2h/status.json').read_text())
assert a['foundation_new_state_sha256']==b['foundation_new_state_sha256'], 'New Omega parameters initialized differently'
PY
  fi
  for stage in main refine; do
    dest="$main"; minutes=480; lr=2e-4; resume=()
    if [[ "$stage" == refine ]]; then dest="$refine"; minutes=120; lr=5e-5; resume=(--resume "$main/last.pt"); fi
    phase "$arm/$stage"
    if [[ ! -s "$dest/last.pt" ]]; then
      fresh_stage "$dest"
      gpu_gate
      "$PYTHON_BIN" training/train_erp_completion.py --config "$config" --dataset-root "$DATASET_ROOT" \
        --omega-checkpoint "$warm/last.pt" --base-checkpoint "$FOUNDATION" --matched-warmup-arm "$arm" \
        --output-dir "$dest" --duration-minutes "$minutes" --stage "$stage" --lr "$lr" \
        --core-latitude-degrees "$cutoff" --head-width 32 --height 256 --width 512 --seed 57 \
        --num-workers 0 --save-every 2000 --progress-bar "${resume[@]}" 2>&1 | tee -a "$dest/train.log"
    fi
    check_status "$dest/status.json" duration "$minutes"
  done
  if [[ "$arm" == B ]]; then
    "$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json,pathlib,sys
out=pathlib.Path(sys.argv[1])
a=json.loads((out/'A/completion_8h/status.json').read_text())
b=json.loads((out/'B/completion_8h/status.json').read_text())
assert a['initial_head_sha256']==b['initial_head_sha256'], 'Completion heads initialized differently'
PY
  fi
  phase "$arm/loss_analysis"
  "$PYTHON_BIN" scripts/analyze_full_erp_training_losses.py \
    --series "warmup=$warm/loss.csv" --series "completion=$main/loss.csv" --series "refine=$refine/loss.csv" \
    --output-dir "$OUTPUT/$arm/loss_analysis"
done

for arm in A B; do
  config="$OUTPUT/$arm.yaml"; warm="$OUTPUT/$arm/warmup_2h"; refine="$OUTPUT/$arm/refine_2h"
  "$PYTHON_BIN" scripts/validate_eval_cardinality.py --config "$config" \
    --dataset-root "$DATASET_ROOT" --index-audit "$OUTPUT/test_index_audit.json"
  dest="$OUTPUT/$arm/full8833"; mkdir -p "$dest"
  phase "$arm/full8833"
  if [[ ! -s "$dest/summary.json" ]]; then
    gpu_gate
    "$PYTHON_BIN" scripts/evaluate_mixed4_depth_checkpoint.py --config "$config" --dataset-root "$DATASET_ROOT" \
      --checkpoint "$warm/last.pt" --erp-completion-checkpoint "$refine/last.pt" \
      --output "$dest/summary.json" --per-sample-csv "$dest/per_sample.csv" --camera-pair-csv "$dest/camera_pairs.csv" \
      --progress-file "$dest/progress.json" --datasets all --limit-per-dataset 0 --sample-policy anchor \
      --pano-count-policy panovggt --camera-eval-max-panos 3 --device cuda --num-workers 0 \
      --amp-dtype bfloat16 --seed 123 --resume --no-print-each-sample 2>&1 | tee -a "$dest/eval.log"
  fi
  "$PYTHON_BIN" scripts/validate_eval_cardinality.py "$dest/summary.json"
  for dataset in stanford2d3ds matterport3d structured3d panocity; do
    phase "$arm/preview_$dataset"
    "$PYTHON_BIN" scripts/reconstruct_pano_omega.py --dataset-root "$DATASET_ROOT" --dataset-format pano_minimal \
      --minimal-datasets "$dataset" --dataset-split test --sample-index 0 --checkpoint "$warm/last.pt" \
      --erp-completion-checkpoint "$refine/last.pt" --output-dir "$OUTPUT/$arm/preview_${dataset}_test0" --device cuda --seed 123
  done
done
phase TRAINING_AND_EVAL_COMPLETE_REPORT_PENDING
date -u +%FT%TZ > "$OUTPUT/TRAINING_AND_EVAL_COMPLETE"
echo '[HANDOFF] Full per-dataset interpretation and matched preview review remain for the supervising task.'
