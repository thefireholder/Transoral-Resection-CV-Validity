#!/usr/bin/env bash
# =============================================================================
# run_local.sh -- run every GPU job of scripts/rerun/ on ONE local GPU
# (e.g. a 6 GB laptop), no SLURM. Resumable: each finished step leaves a
# marker in scripts/rerun/logs/local/<step>.done and is skipped next time.
#
#   bash scripts/rerun/run_local.sh              # sam3 -> maskrcnn -> yolo -> monai  (+ figures)
#   bash scripts/rerun/run_local.sh --sweep      # ... then the dataset-size sweep (10 trainings, long)
#   bash scripts/rerun/run_local.sh --only yolo  # one step: sam3 (=GT box) | maskrcnn | yolo | monai | sam3_text | sam3_yolobox | sweep
#   bash scripts/rerun/run_local.sh --dry        # print the plan only
#
# Parallelism on a single GPU:
#   GPU jobs are strictly SERIAL (VRAM). CPU post-processing of a finished
#   step (rescoring, collect_results, figures) runs in the BACKGROUND while
#   the next GPU job trains; the script waits for everything at the end.
#   On the cluster the three sbatch files are independent and run in parallel.
#
# One-time setup on another machine (laptop):
#   1. conda env with: torch+torchvision (CUDA), ultralytics, transformers>=5 (Sam3Model),
#      pycocotools, opencv-python, matplotlib, pyyaml, pillow, numpy, monai (==1.5.1 for torch 2.6;
#      any monai>=1.3 that matches your torch), scikit-learn, tqdm.
#      e.g.  conda create -n tors python=3.11 && conda activate tors
#            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#            pip install ultralytics transformers pycocotools opencv-python matplotlib pyyaml "monai==1.5.1" scikit-learn tqdm
#   2. copy from the cluster, keeping the layout:
#        <TORS_SHARED>/processed_data/coco_format/1200images/        (images + annotations, ~? GB)
#        <TORS_SHARED>/processed_data/yolo_format/1200images/
#        <TORS_SHARED>/result/                                       (this folder)
#      optional (only for scripts/export_yolo_predictions.py of the OLD model):
#        <TORS_SHARED>/train_script/yolov12_project/runs/segment/1200images/1200images_train/weights/best.pt
#      NOTE processed_data/yolo_format/1200images/data.yaml has an absolute `path:` -> edit it to your copy.
#   3. huggingface-cli login   (facebook/sam3 is gated) -- or export HF_HOME to a copy of the cluster cache
#   4. export TORS_SHARED=/path/to/your/copy ;  cd $TORS_SHARED/result ;  bash scripts/rerun/run_local.sh
# =============================================================================
set -uo pipefail

# ---- knobs (override with env vars: MRCNN_BATCH=4 bash run_local.sh) ----------------------
CONDA_ENV="${CONDA_ENV:-pytorch_env}"          # on the laptop: the env you created (e.g. tors)
MRCNN_BATCH="${MRCNN_BATCH:-2}"                # cluster recipe = 4 (needs >~10 GB); 2 fits 6 GB
MRCNN_LR="${MRCNN_LR:-0.0025}"                 # linear-scaling rule: 0.005 * (batch/4)
MRCNN_EPOCHS="${MRCNN_EPOCHS:-12}"
MRCNN_LR_STEPS="${MRCNN_LR_STEPS:-8 11}"
YOLO_BATCH="${YOLO_BATCH:-16}"                 # yolo11n @640 fits 6 GB at 16
YOLO_EPOCHS="${YOLO_EPOCHS:-60}"
MONAI_EPOCHS="${MONAI_EPOCHS:-100}"             # project recipe: 100 max, early stopping patience 15
MONAI_BATCH="${MONAI_BATCH:-4}"                 # 512x512 UNet fits 6 GB at 4
SAM3_ARGS="${SAM3_ARGS:---fp16}"               # fp32 SAM3 is ~3.5 GB + activations; fp16 is safer on 6 GB. "" for fp32
DEVICE="${DEVICE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ---- locate ----------------------------------------------------------------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT="$(cd "$HERE/../.." && pwd)"
export TORS_SHARED="${TORS_SHARED:-$(cd "$RESULT/.." && pwd)}"
LOG="$HERE/logs/local"; mkdir -p "$LOG"
cd "$RESULT"

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then source "$HOME/miniforge3/etc/profile.d/conda.sh";
elif command -v conda >/dev/null; then eval "$(conda shell.bash hook)"; fi
conda activate "$CONDA_ENV" || { echo "cannot activate conda env $CONDA_ENV"; exit 1; }

DO_SWEEP=0; ONLY=""; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --sweep) DO_SWEEP=1;; --only) ONLY="$2"; shift;; --dry) DRY=1;; *) echo "unknown arg $1"; exit 1;; esac; shift; done

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG/run_local.log"; }
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
done_() { [ -f "$LOG/$1.done" ]; }
run_step() {   # run_step <name> <command...>   (serial, GPU)
  local name="$1"; shift
  if done_ "$name"; then say "skip $name (done marker exists: $LOG/$name.done)"; return 0; fi
  say "START $name :: $*"
  [ $DRY = 1 ] && return 0
  if "$@" > "$LOG/$name.out" 2> "$LOG/$name.err"; then
    touch "$LOG/$name.done"; say "DONE  $name  (log: $LOG/$name.out)"
  else
    say "FAILED $name  -> see $LOG/$name.err"; FAILED+=("$name"); return 1
  fi
}
post() {       # post <name> <command...>   (CPU, background, does not block the next GPU job)
  local name="$1"; shift
  say "post-processing $name in background :: $*"
  [ $DRY = 1 ] && return 0
  ( "$@" > "$LOG/post_$name.out" 2>&1 && say "post-processing $name finished" || say "post-processing $name FAILED (see $LOG/post_$name.out)" ) &
  BG+=($!)
}
FAILED=(); BG=()

say "=== run_local.sh  RESULT=$RESULT  TORS_SHARED=$TORS_SHARED  env=$CONDA_ENV  device=$DEVICE ==="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" | tee -a "$LOG/run_local.log"
say "plan: [GPU serial] sam3 GT-box (~1-3 h) -> maskrcnn curves (~3-5 h @bs$MRCNN_BATCH) -> yolo curves (~3-5 h) -> monai (~2-4 h) -> sam3_text (~1-2 h) -> sam3_yolobox (~1 h)$( [ $DO_SWEEP = 1 ] && echo ' -> sweep (10 trainings, ~1-2 days)')"
say "      [CPU background] after each: compute_mask_metrics / collect_results / make_figures"

# ---------------------------------------------------------------- 1. SAM3 (shortest, unlocks Fig 1b + 2)
if want sam3; then
  run_step sam3 python scripts/rerun/sam3/sam3_gt_box_prompt.py --device cuda $SAM3_ARGS \
  && post sam3 bash -c "python scripts/compute_mask_metrics.py sam3_gtbox && python scripts/collect_results.py && python plot/make_figures.py 3b"
fi

# ---------------------------------------------------------------- 2. Mask R-CNN 12-epoch retrain (adopted as THE Mask R-CNN model)
if want maskrcnn; then
  run_step maskrcnn python scripts/rerun/maskrcnn_train_curves.py --epochs "$MRCNN_EPOCHS" --lr-steps $MRCNN_LR_STEPS \
      --batch-size "$MRCNN_BATCH" --lr "$MRCNN_LR" --seed 42 --num-workers 4 --adopt \
  && post maskrcnn bash -c "python scripts/compute_mask_metrics.py maskrcnn && python scripts/collect_results.py && python plot/make_figures.py"
fi

# ---------------------------------------------------------------- 3. YOLO retrain + per-epoch eval
if want yolo; then
  # yolo11n-seg.pt (pretrained) is auto-downloaded by ultralytics into the cwd if absent
  run_step yolo python scripts/rerun/yolo_train_curves.py --epochs "$YOLO_EPOCHS" --device "$DEVICE" \
  && post yolo bash -c "python scripts/collect_results.py && python plot/make_figures.py 1a"
fi

# ---------------------------------------------------------------- 4. MONAI UNet retrain on the COCO-derived masks (adopted as THE MONAI model)
if want monai; then
  run_step monai_masks python scripts/rerun/monai/prepare_masks.py \
  && run_step monai python scripts/rerun/monai/monai_train_curves.py --epochs "$MONAI_EPOCHS" --batch-size "$MONAI_BATCH" --seed 42 --num-workers 4 \
  && post monai bash -c "python scripts/compute_mask_metrics.py monai && python scripts/collect_results.py && python plot/make_figures.py"
fi

# ---------------------------------------------------------------- 5. SAM3 text-prompt (zero-shot) and YOLO-box prompt  (Fig 3b + main figures)
if want sam3_text; then
  run_step sam3_text python scripts/rerun/sam3/sam3_text_prompt.py --device cuda $SAM3_ARGS \
  && post sam3_text bash -c "python scripts/compute_mask_metrics.py sam3_text && python scripts/collect_results.py && python plot/make_figures.py"
fi
if want sam3_yolobox; then
  run_step sam3_yolobox python scripts/rerun/sam3/sam3_yolo_box_prompt.py --device cuda $SAM3_ARGS \
  && post sam3_yolobox bash -c "python scripts/compute_mask_metrics.py sam3_yolobox && python scripts/collect_results.py && python plot/make_figures.py 3b"
fi

# ---------------------------------------------------------------- 6. optional dataset-size sweep (Figure 5)
if want sweep && [ $DO_SWEEP = 1 -o "$ONLY" = sweep ]; then
  run_step sweep_subsets python scripts/rerun/dataset_size_sweep/make_subsets.py
  for n in 75 150 300 600 732; do        # small -> large so partial results are useful early
    run_step "sweep_yolo_n$n"     python scripts/rerun/dataset_size_sweep/yolo_sweep.py --n $n --seed 0 --epochs "$YOLO_EPOCHS" --device "$DEVICE"
    run_step "sweep_maskrcnn_n$n" python scripts/rerun/dataset_size_sweep/maskrcnn_sweep.py --n $n --seed 42 \
        --epochs "$MRCNN_EPOCHS" --lr-steps $MRCNN_LR_STEPS --batch-size "$MRCNN_BATCH" --lr "$MRCNN_LR"
    post "sweep_n$n" bash -c "python scripts/rerun/dataset_size_sweep/collect_sweep.py && python plot/make_figures.py 5"
  done
fi

# ---------------------------------------------------------------- wait for background CPU work, final figures
say "waiting for background post-processing (${#BG[@]} jobs)..."
for pid in "${BG[@]}"; do wait "$pid" 2>/dev/null; done
[ $DRY = 1 ] || { python scripts/collect_results.py > "$LOG/final_collect.out" 2>&1; python plot/make_figures.py > "$LOG/final_figures.out" 2>&1; }
if [ ${#FAILED[@]} -gt 0 ]; then say "FINISHED WITH FAILURES: ${FAILED[*]}   (fix, then re-run; done steps are skipped)"; exit 1; fi
say "ALL DONE. figures in $RESULT/figures, provenance in $RESULT/data/SOURCES.md"
