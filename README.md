# result/ — unified numbers + figures for the 4-model tonsillectomy segmentation comparison

Models: **Mask R-CNN**, **YOLO11n-seg**, **SAM3** (**YOLO-box-prompted** two-stage pipeline in the main figures; zero-shot text and GT-box oracle in the Fig 3b/3c ablation — see §5e), **MONAI UNet** (semantic; retrained here on the COCO-derived masks — see §5c).
The folder is a git repo: `github.com/thefireholder/Transoral-Resection-CV-Validity` (checkpoints / masks / image dumps are git-ignored).
Dataset "1200images" = 1048 frames (train 732 / val 157 / test 158), 21-class vocabulary, 14 classes present in the test split.
**All metrics are mask metrics** (never bounding box). All tables/figures use the **test** split unless stated.

My original notes are in [README_original_notes.md](README_original_notes.md).

---

## 0. TL;DR — what to run

```bash
source /u/sl257/miniforge3/etc/profile.d/conda.sh
conda activate pytorch_env                    # <- the ONLY env used (see §2)
cd /u/sl257/shared_data/result

# already run for you (CPU, seconds–minutes). Re-run whenever a source file changes:
python scripts/collect_results.py             # gathers every existing number  -> data/*.csv, data/SOURCES.md
python scripts/export_maskrcnn_predictions.py # saved Mask R-CNN masks         -> data/predictions/maskrcnn_test.json
python scripts/export_yolo_predictions.py     # best.pt on CPU (~30 s)         -> data/predictions/yolo_test.json
python scripts/compute_mask_metrics.py --all  # one scoring code for all models -> data/computed/, data/pr_curves/
python plot/make_figures.py                   # all figures                    -> figures/

# NOT run (need a GPU) — fill the n/p gaps. Cluster version (the 3 jobs are independent -> run in parallel):
sbatch scripts/rerun/sam3/sam3_gt_box_prompt.slurm       # ~1 h  : SAM3 GT-box masks+scores (needs the HF cache re-downloaded on the cluster)
sbatch scripts/rerun/maskrcnn_train_curves.sbatch        # ~4 h  : 12-epoch retrain, per-epoch train/val P,R -> Fig 1a; becomes THE Mask R-CNN model (--adopt)
sbatch scripts/rerun/yolo_train_curves.slurm             # ~3 h  : per-epoch train/val P,R -> Fig 1a
sbatch scripts/rerun/monai/monai_train_curves.slurm      # ~3 h  : UNet retrain on COCO-derived masks (§5c) -> all MONAI n/p
python scripts/rerun/dataset_size_sweep/make_subsets.py  # done  : seeded subsets (75/150/300/600/732)
sbatch --array=0-9 scripts/rerun/dataset_size_sweep/sweep.slurm   # 10 GPU-jobs : Fig 5 (not needed now)

# Same jobs WITHOUT slurm, on one local GPU (e.g. 6 GB laptop) -- serial, resumable, waits for results:
bash scripts/rerun/run_local.sh            # sam3 -> maskrcnn -> yolo -> monai, figures refresh after each step
bash scripts/rerun/run_local.sh --only monai   # just the MONAI retrain (the other three are done)
bash scripts/rerun/run_local.sh --sweep    # ... + the sweep afterwards
bash scripts/rerun/run_local.sh --only yolo
```

Each sbatch / run_local step ends by re-running `compute_mask_metrics.py` / `collect_results.py` / `make_figures.py`,
so the figures update themselves. Laptop setup (what to copy, env, `TORS_SHARED`) is in the header of
[scripts/rerun/run_local.sh](scripts/rerun/run_local.sh) and §2.

---

## 1. Folder map

```
result/
├── README.md                     this file
├── README_original_notes.md      my original notes (moved, unchanged)
├── some disorganized results/    untouched. Everything in it was traced back to its original run dir (see data/SOURCES.md)
│
├── data/                         >>> THE ONE PLACE FOR NUMBERS <<<  (all generated; do not hand-edit — edit the script or source instead)
│   ├── metrics_per_class.csv     model,split,source,class, AP50,AP50_95,precision,recall,iou,dice
│   ├── metrics_aggregate.csv     model,split,source, mAP50,mAP50_95,precision,recall,iou,dice,n_classes
│   ├── training_curves/<model>.csv          epoch,train_loss,val_loss,train_precision,train_recall,val_precision,val_recall,val_mAP50,val_mAP50_95
│   ├── pr_curves/<model>.json               per-class precision on a 1000-pt recall grid (mask IoU 0.5)
│   ├── predictions/<model>_test.json        every predicted mask (COCO RLE) + score — the raw material for overlays/curves/metrics
│   ├── computed/<model>_test.json           metrics recomputed with ONE common implementation (scripts/seglib.py)
│   ├── dataset_size_sweep.csv               Figure 5 input (header only until the sweep is run)
│   └── SOURCES.md                           WHERE EACH NUMBER CAME FROM (file path + how it was computed) — auto-generated
│
├── scripts/                      CPU scripts (run now) + GPU re-runs (you run)
│   ├── seglib.py                 shared: RLE, GT loading, greedy IoU matching, AP (COCO or ultralytics rule), PR curves, F1-max P/R
│   ├── collect_results.py        parse original run dirs / logs / gathered csvs  -> data/metrics_*.csv, training_curves, SOURCES.md
│   ├── export_maskrcnn_predictions.py   run_output/mask_metrics/mask_metrics_test.json -> uniform prediction file
│   ├── export_yolo_predictions.py       best.pt -> uniform prediction file (CPU ok)
│   ├── compute_mask_metrics.py          uniform prediction file -> data/computed + data/pr_curves
│   ├── original/                 verbatim copies of the training/eval code that produced the numbers (reference; rerun/ imports from here)
│   └── rerun/                    GPU jobs, each = <name>.py + <name>.slurm/.sbatch, all seeded
│       ├── run_local.sh                 NO-SLURM runner: serial GPU steps + background CPU post-processing, resumable
│       ├── maskrcnn_train_curves.*      retrain 12 ep (LR x0.1 @8,11; seed 42) logging train+val mask P/R every epoch; --adopt
│       ├── yolo_train_curves.*          retrain (seed 0) with save_period=1, then eval every epoch ckpt on train+val
│       ├── sam3/                        sam3_gt_box_prompt.* (oracle), sam3_text_prompt.py + sam3_yolo_box_prompt.py (built on the laptop, see HANDOFF), prompts.json
│       ├── monai/                       prepare_masks.py (COCO polygons -> semantic PNGs, done), monai_train_curves.py + .slurm, data/
│       └── dataset_size_sweep/          make_subsets.py, yolo_sweep.py, maskrcnn_sweep.py, collect_sweep.py, sweep.slurm
│
├── plot/
│   ├── plot_config.py            EVERY knob: model list/order/colours/labels, class order, metric source, layouts, fonts, dpi
│   └── make_figures.py           fig1a fig1b fig2 fig3 fig4 fig5 — each an independent function reading only data/
└── figures/                      fig1a_training_curves, fig1b_qualitative, fig2_pr_curves_{per_model,per_class},
                                  fig3_map_table (+ .csv/.md/.tex), fig3b_sam3_prompt_ablation_per_class (+ .csv/.md/.tex),
                                  fig3c_sam3_prompt_ablation_aggregate (+ .csv/.md),
                                  fig4_aggregate_bars, fig5_dataset_size   (.png + .pdf)
```

---

## 2. Conda environment

Everything (CPU scripts, plotting, and all GPU re-runs) uses the existing env
**`pytorch_env`** (`/u/sl257/miniforge3/envs/pytorch_env`, python 3.11): matplotlib 3.11, numpy 2.4, torch 2.6+cu124,
torchvision 0.21, ultralytics 8.4, pycocotools, opencv 5, transformers 5.14 (has `Sam3Model`), pyyaml,
**monai 1.5.1 + scikit-learn + tensorboard + tqdm (pip-installed 2026-09-17 for the MONAI retrain; monai 1.6 needs
torch ≥ 2.8 so it must stay pinned at 1.5.1 with this torch)**.
No new env was created; pandas is not installed there so the code deliberately uses only `csv`/`json`.

The original Mask R-CNN job was run by **am232** in their env `tors_env` (module anaconda3/2024.10); the re-run scripts here
use `pytorch_env` instead, which has everything `train_maskrcnn.py` imports.
SAM3 needs `HF_HOME=/u/sl257/scratch/huggingface` + `HF_HUB_OFFLINE=1` (weights already cached) — set in the slurm file.

**On another machine** (laptop, 6 GB GPU): create any env with torch+torchvision(CUDA), ultralytics, transformers≥5,
pycocotools, opencv-python, matplotlib, pyyaml; copy `processed_data/{coco_format,yolo_format}/1200images` and this
`result/` folder keeping the same relative layout; `export TORS_SHARED=<that folder>` (every script reads it via
`scripts/seglib.py`); edit the absolute `path:` in the copied `yolo_format/1200images/data.yaml`; `huggingface-cli login`
for the gated `facebook/sam3`. `run_local.sh` defaults to Mask R-CNN batch 2 / lr 0.0025 (linear scaling of the cluster's
4 / 0.005) and SAM3 fp16 so they fit 6 GB — override with `MRCNN_BATCH=4 MRCNN_LR=0.005 SAM3_ARGS= bash …`.
`collect_results.py` still works there: anything that only exists in the cluster's `train_script/` becomes n/p.

---

## 3. Status: what exists, what is n/p, what fills it

| Figure | Mask R-CNN | YOLO11n-seg | SAM3 | MONAI |
|---|---|---|---|---|
| **1a** training curve: loss | train ✔ val ✔ (12-epoch rerun) | train ✔ val ✔ (60 epochs) | — (not trained) | train ✔ val ✔ (98 epochs, §5c) |
| **1a** training curve: train-split P/R | ✔ (12-ep rerun) | ✔ (rerun) | — | ✔ (retrain, §5c) |
| **1a** training curve: val-split P/R | ✔ | ✔ | — | ✔ |
| **1b** overlay pred vs GT | ✔ | ✔ | ✔ YOLO-box (§5e) | ✔ (retrain) |
| **2** mask PR curve per class | ✔ (original curves were **box**-based; these are mask) | ✔ | ✔ YOLO-box | ✔ (retrain) |
| **3** mAP table per class + aggregate | ✔ | ✔ | ✔ YOLO-box; Fig 3b/3c: all 3 conditions | ✔ (all 14 classes after retrain) |
| **4** mAP / Dice / IoU bars | ✔ | ✔ | ✔ YOLO-box | ✔ (all `recomputed`, same definition) |
| **5** dataset-size sweep | **n/p** → `dataset_size_sweep/` | **n/p** → same | n/a (no training) | n/a |

"n/p" panels/cells are drawn automatically; nothing crashes when data is missing.

---

## 4. Two kinds of numbers: `source = reported` vs `recomputed`

`data/metrics_*.csv` carry a `source` column and `plot_config.METRIC_SOURCE` picks which one Figures 3/4 use.

* **reported** — each model's own evaluation code, numbers copied verbatim from the run directories (this is what the
  gathered folder contains). Problem: they are **not computed the same way**:
  * Mask R-CNN & MONAI use the **COCO** 101-point AP (precision drops to 0 beyond the max recall).
  * YOLO & SAM3 use the **ultralytics** AP (linearly ramps precision to 0 at recall 1 → *higher* AP when recall < 1;
    e.g. Mask R-CNN `cut` = 0.644 COCO-rule vs 0.822 ultralytics-rule on the very same predictions).
  * Dice/IoU: Mask R-CNN & SAM3 = mean over instances (Mask R-CNN over matched instances only, SAM3 over all GT
    instances since every GT box is prompted); MONAI = per-frame **semantic** Dice; YOLO = none.
  * P/R: YOLO at its F1-max confidence; MONAI pixel-level; others none.
* **recomputed** — `scripts/compute_mask_metrics.py` scores every model's `data/predictions/<model>_test.json` with
  `scripts/seglib.py`: same greedy IoU≥0.5 per-class matching, COCO AP rule (`--ap-method ultralytics` for the other),
  P/R at the single F1-maximising confidence (ultralytics convention), instance-matched Dice/IoU.
  Validation: Mask R-CNN recomputed = COCOeval to 4 decimals (0.5032 / 0.3274); YOLO recomputed with
  `--ap-method ultralytics` = 0.591 vs ultralytics' 0.576 (residual: full-res masks & conf≥0.05 here vs 640-px masks &
  conf≥0.001 inside `model.val`).
  Available now for Mask R-CNN and YOLO; SAM3 after its rerun; MONAI never (no masks on this computer).

**Default is now `METRIC_SOURCE = "recomputed"`** with `FILL_FROM_OTHER_SOURCE = True`: cells that only exist as
`reported` (MONAI always; SAM3 until its rerun) are filled from there and marked `*` / hatched. MONAI's AP is already
COCO-rule so it is comparable; its Dice/IoU are semantic (footnote it).

---

## 5. Caveats you should know before quoting numbers

1. **"1200images" is 1048 images** (732/157/158). Figure 5 sizes 900 and 1200 are therefore impossible; the sweep
   uses 75/150/300/600/732 (=all). Change `FIG5_SIZES` in `plot_config.py` if you want different ticks.
2. **MONAI used a different class merge**: no `bipolar`/`bot` (→ n/p rows), and it has `fat fascia` (33 test frames)
   which does not exist in the COCO test annotations the other three models were scored on. Its 13-class aggregate is
   `"AGGREGATE (all 13 classes with GT)"` from `_result_map_iou_dice_per_class.csv` (the `n=12 evaluable` variant is
   also in that file). MONAI Dice/IoU are semantic per-frame values, not instance-matched.
3. **SAM3 is not a comparable detector**: it is given the GT box of every instance, so AP50≈1 reflects mask quality
   given a perfect localisation, not detection. Its P/R and PR-curve are degenerate in the original run (conf=1.0
   everywhere); the rerun saves SAM3's own mask score so a real ranking exists.
4. **The original Mask R-CNN run (5 epochs, LR ×0.1 after epoch 3) is too short**: val AP was still rising when the LR
   was cut (val mask AP50 0.539→0.570→0.583→0.593→0.587), and it saw only ~900 iterations vs YOLO's 60 epochs.
   Decision: the rerun uses **12 epochs, LR ×0.1 after epochs 8 and 11** (`--epochs 12 --lr-steps 8 11`, same for the
   sweep) and is **adopted** as the Mask R-CNN model (`--adopt` writes `predictions/maskrcnn_test.json`; the collector
   then marks the old "reported" Mask R-CNN numbers as superseded — they stay in `run_output/` and in
   `README_original_notes.md`'s gathered folder). Until that job runs, fig 1a shows only the old train loss.
5. **Mask R-CNN's dataset copy `coco_format/maskrcnn` is owned by am232 and not readable by sl257**; the re-run uses
   `coco_format/1200images` (same 732/157/158 split, same 21 categories, 983 vs 984 test instances = 1 invalid
   polygon that both pipelines drop).
6. **YOLO per-class Mask mAP exists only in the stdout log** (`yolo11n_1200images_9808510.out`); the `results.csv` in
   `1200images_test/` has box mAP columns only. The collector parses the log.
7. **Training-set precision/recall does not exist for any model** and cannot be reconstructed from saved checkpoints
   (only best/last were kept). The reruns log it every epoch (Mask R-CNN on a fixed seeded 157-image train subset to
   keep eval time sane; `--train-eval-size 0` = all 732).
8. **SAM2** (`some disorganized results/1200 images/sam2-1200`, `train_script/SAM2_project/sam2/result/sam2/1200images`)
   is *not* collected. To add it: copy the `sam3()` block in `collect_results.py` pointing at the sam2 files, add a
   `"sam2"` entry to `plot_config.MODELS`.

---

## 5b. Status after the laptop reruns (2026-09-17)

All three GPU jobs ran on the laptop (Quadro RTX 3000, 6 GB; `run_local.sh`) and the small outputs were copied back
(`data/predictions/{sam3,maskrcnn,yolo_rerun}_test.json`, `data/training_curves/{maskrcnn,yolo}_rerun.csv`,
`data/computed/`, `data/pr_curves/`, `scripts/rerun/logs/local/`) plus `scripts/rerun/output/` (1.5 GB: Mask R-CNN `best.pth`, YOLO `best.pt`/`last.pt`/all 60 `epochN.pt`, ultralytics `results.csv` and plots).

* **SAM3 (GT-box, now key `sam3_gtbox`)** — 983 instances, mean IoU 0.7978 / Dice 0.8695 = the original run (0.7974 / 0.8692) ✔.
  Note the `recomputed` SAM3 IoU/Dice (0.831 / 0.904) are higher than `reported` because `seglib` averages over
  *matched* instances (IoU ≥ 0.5) like Mask R-CNN/YOLO, whereas the original SAM3 script averaged over *all* GT
  instances. Both are in `metrics_aggregate.csv`; say which one you quote.
* **Mask R-CNN, 12 epochs** — laptop recipe **batch 2 / lr 0.0025** (linear scaling of the cluster's 4 / 0.005;
  see `scripts/rerun/output/maskrcnn/recipe.txt`). Best val mask mAP50-95 at **epoch 6** (0.367); val loss rises and
  train precision saturates (~0.97) afterwards, i.e. it overfits — so 5 epochs was only slightly short. Adopted as the
  Mask R-CNN model: test mAP50 0.516 / mAP50-95 0.338 / IoU 0.806 / Dice 0.888 (old 5-epoch checkpoint: 0.503 / 0.327 /
  0.805 / 0.887). The old "reported" rows are n/p (superseded) in the CSVs.
* **YOLO, 60 epochs** — same seed-0 recipe; per-epoch train **and** val P/R now exist (`yolo_rerun.csv`; losses from the run's own `results.csv`).
  The metrics/predictions in the figures still come from the ORIGINAL cluster checkpoint (`yolo_test.json`);
  the laptop retrain's test masks are kept alongside as `yolo_rerun_test.json` (not adopted — the original run
  is the reference, the rerun only supplies the curve). Fixed after the run: ultralytics' `epochN.pt` is
  0-indexed vs 1-indexed `results.csv`, so the returned csv was re-aligned.
* **MONAI, retrained on the COCO-derived masks (laptop, 2026-09-17, 2 h 47 min)** — 98 epochs (early stop), best epoch 83
  (val mean Dice 0.653), adopted. Test (recomputed): mAP50 0.454 / mAP50-95 0.259 / IoU 0.754 / Dice 0.854, vs the old
  unified-vocabulary model's 0.384 / 0.203 (and semantic Dice 0.468, a different quantity). Old rows = superseded.
  `scripts/rerun/output/monai/best_model.pth` is on the laptop only (copy it over).
* Still **n/p**: only Figure 5 (sweep not run).

## 5c. MONAI: why it is retrained, and how (2026-09-17)

`train_script/monai-project` (code + `checkpoints/unified_best.pth`, the model behind the gathered `monai-results/`)
trained on its **own** re-parse of the raw VIA annotations with an 18-class "unified" vocabulary (`bipolar`→`maryland`,
`bot`→`base of tongue`, `pharyngeal`→`posterior pharyngeal wall`, a separate `fat fascia` with 33 test frames that the
COCO export has 0 annotations for). Its data folder and TensorBoard logs are not on this computer. Scoring that
checkpoint against our COCO GT would penalise it for label merges, and no training curve could be recovered — so the
decision (with the user) was to **retrain the same UNet recipe on the COCO-derived masks**:

* `scripts/rerun/monai/prepare_masks.py` (done, CPU, 7 s): COCO polygons → `uint8` semantic PNGs, 21 classes + bg,
  largest-area-first so small instances on top win the pixel; same 732/157/158 split. → `scripts/rerun/monai/data/`.
* `scripts/rerun/monai/monai_train_curves.py`: the project's recipe verbatim (UNet 32-512, 512×512, DiceFocal with
  inverse-frequency class weights, AdamW 1e-4, cosine over 100, grad-clip 1, batch 4, early stop 15 on val mean Dice,
  best = val mean Dice, seed 42) using the project's own `src/` (copied to `scripts/original/monai/`). Every epoch it
  also computes instance-level P/R/AP on val and a seeded 157-frame train subset through `seglib` → uniform curve csv.
  Instances of the semantic output = 8-connected components of the argmax map (≥ 20 px on the 512 grid), score = mean
  softmax of the class (the project's own `17_compute_instance_ap.py` convention), upsampled to full res as RLE.
* Output: `data/training_curves/monai_rerun.csv`, `data/predictions/monai_test.json` (adopted → the old gathered
  numbers become n/p "superseded" like Mask R-CNN's), `scripts/rerun/output/monai/{best_model.pth,recipe.txt,log.txt}`.
  Then `compute_mask_metrics.py monai` scores it with exactly the same code as the other three.
* Known differences that remain (inherent to a semantic model): touching same-class instances fuse into one
  component; masks are predicted at 512×512 (non-aspect-preserving) and upsampled.

## 5d. Incident 2026-09-17: `train_script/` was deleted

`/u/sl257/shared_data/train_script` (Mask R-CNN, YOLO, SAM2/3 and MONAI projects, their run outputs and logs) was
removed with `rm -rf`. `/projects` is NCSA Taiga (Lustre): no snapshots, no undelete — a help ticket to
help@ncsa.illinois.edu is the only route to the originals. **The figure pipeline is unaffected**: every input had
already been copied into `result/` (code → `scripts/original/`, numbers → gathered folder + `data/`, YOLO checkpoint →
gathered folder, verified identical). `collect_results.py` now falls back to the gathered copies automatically, and
[recovered/README.md](recovered/README.md) lists what survives where and what is truly lost (old 5-epoch Mask R-CNN
checkpoint, MONAI `unified_best.pth`/`jml619_best.pth` and the rest of that project, secondary YOLO/SAM runs).
Paths under `train_script/` quoted in `data/SOURCES.md` and elsewhere in this README are historical.
A backup of `result/` is at `/u/sl257/backups/result_2026-09-17.tgz` (home filesystem, separate from Lustre).

## 5e. SAM3: three prompting conditions (2026-09-17)

The original SAM3 evaluation prompted it with the **ground-truth box of every instance**, so it never had to
detect or classify anything — its 0.97 mAP50 is an upper bound on mask quality, not a competitor's score. The
comparison is now organised as:

| key | prompt | role |
|---|---|---|
| `sam3_yolobox` | YOLO11n's predicted boxes (+ YOLO class & score) | **the SAM3 in the main figures**: two-stage detector→SAM3 (decision 2026-09-18) |
| `sam3_gtbox` | GT boxes (the original run) | oracle upper bound (Fig 3b/3c only) |
| `sam3_text` | class-name text (`scripts/rerun/sam3/prompts.json`), zero-shot | fully self-contained zero-shot condition (Fig 3b/3c only; ~0.03 mAP, explained in the manuscript) |

**Figure 3b** (per-class AP for the three conditions + YOLO reference) and **Figure 3c** (one aggregate row each:
mAP, P, R, IoU, Dice + "what it tests") hold the ablation; `plot_config.MAIN_MODELS` picks which SAM3 enters
Figs 1b/2/3/4 (`sam3_yolobox` by default). The condition descriptions are kept in
`plot_config.FIG3B_NOTES` for the figure caption but not drawn (`FIG3B_SHOW_NOTES = False`).

Result (2026-09-18, laptop run): text-prompted SAM3 is essentially blind to this vocabulary — mAP50 **0.028**;
anatomy prompts return nothing usable, "suction tube" fires on every frame (2,547 instances), "base of tongue"
9,102 low-score instances. YOLO-box→SAM3 (0.499) is *below* YOLO alone (0.543): given YOLO's box and class, SAM3's
mask is on average no better than YOLO's own (IoU 0.809 vs 0.810) and worse for thin/diffuse classes (`cut`
0.62 vs 0.78, `soft palate` 0.18 vs 0.56 — inside a soft-palate box SAM3 prefers a smaller salient object).
GT-box SAM3 (0.970) remains the upper bound. Per-concept prediction counts: `data/computed/sam3_text_prompt_counts.csv`. The text-prompt design has one convention to
know: the label space contains the same concept under two names from the two annotated videos (`bot`/`base of
tongue`, `bipolar`/`maryland`; verified to be perfectly separated by source video), so each concept is prompted
once and written with the class name of the frame's source vocabulary. The scripts `sam3_text_prompt.py` and
`sam3_yolo_box_prompt.py` were built and run on the laptop (spec + the laptop session's notes in
`HANDOFF_FOR_CLAUDE.txt`; both move SAM3's outputs to CPU before post-processing to fit 6 GB).

## 6. Re-plotting / changing figures

Edit `plot/plot_config.py` (model order & colours, class order, `METRIC_SOURCE`, `FIG2_LAYOUT` per_model|per_class,
`FIG1B_IMAGES`/thresholds/crop, table metrics, bar metrics, fonts/dpi/formats) and run `python plot/make_figures.py [1a 1b 2 3 4 5]`.
Each `figN()` in `make_figures.py` is self-contained (reads csv/json from `data/` only), so changing one figure cannot
break another. Figure 3 also writes `.csv`, `.md`, `.tex` versions of the table next to the png.

---

## 7. Running the GPU jobs: what is parallel, what is serial

* **Cluster (slurm):** `sam3_eval_save_predictions`, `maskrcnn_train_curves`, `yolo_train_curves` are independent —
  submit all three at once. The sweep is a 10-task array (5 sizes × 2 models), also independent of the three.
  Each job's post-processing (rescore → collect → figures) runs at its end; if two finish simultaneously the last
  `collect_results.py` wins, which is harmless (it is deterministic from files on disk) — re-run it once at the end.
* **One local GPU (`run_local.sh`):** GPU steps are serial (VRAM), in the order SAM3 → Mask R-CNN → YOLO → sweep so the
  cheapest/most-blocking result lands first. The CPU post-processing of each finished step runs in the background
  while the next GPU step trains; the script `wait`s at the end and re-renders all figures. Every finished step leaves
  `scripts/rerun/logs/local/<step>.done`, so after a crash/reboot you re-run the same command and it resumes.
  Logs: `scripts/rerun/logs/local/<step>.{out,err}`; progress: `scripts/rerun/logs/local/run_local.log`.

## 8. Re-collecting numbers

`python scripts/collect_results.py` rebuilds `data/metrics_*.csv`, `data/training_curves/*.csv` and `data/SOURCES.md`
from the original locations every time. If a rerun has written `data/training_curves/<model>_rerun.csv`, it is used
instead of the log-parsed one (delete/rename to fall back). Provenance for every value is in `data/SOURCES.md`.
