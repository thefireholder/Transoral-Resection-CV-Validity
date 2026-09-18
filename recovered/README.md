# recovered/ — what survived the accidental `rm -rf /u/sl257/shared_data/train_script` (2026-09-17 ~21:35)

The filesystem (/projects = NCSA Taiga, Lustre) has no snapshots; a help ticket was the only route for the files themselves.
Everything the figures depend on had already been copied into result/, so the pipeline still runs:

| deleted original | surviving copy |
|---|---|
| train_script/*/ training + eval code (all 4 models) | scripts/original/ (verbatim copies) |
| yolov12_project/runs/segment/1200images/1200images_train/weights/best.pt, last.pt | recovered/yolo11n_1200images_{best,last}.pt  (from the gathered folder; verified to reproduce data/predictions/yolo_test.json exactly) |
| yolov12_project/runs/segment/1200images/1200images_train/results.csv | some disorganized results/1200 images/yolo-1200/val result/results.csv |
| yolov12_project/runs/segment/1200images/yolo11n_1200images_9808510.out (test table) | some disorganized results/1200 images/yolo-1200/test result/result.txt |
| SAM2_project/sam2/result/sam3/1200images/*, logs/sam3_eval_9794464.out | some disorganized results/1200 images/sam3-1200/  (per_class_ap.json, result.txt, curves, vis) |
| SAM2_project/sam2/result/sam2/1200images/* | some disorganized results/1200 images/sam2-1200/ |
| maskrcnn_project/run_output/mask_metrics/mask_metrics_summary.json (5-epoch model) | some disorganized results/1200 images/rcnn-1200/rcnn_result.txt (same numbers, text) |
| maskrcnn_project/slurm_9815937_full.out (5-epoch training log) | data/training_curves/maskrcnn_original5ep.csv (the per-epoch values, transcribed) |
| maskrcnn_project/run_output/maskrcnn_resnet50_fpn_best.pth (5-epoch) | LOST — superseded by scripts/rerun/output/maskrcnn/maskrcnn_resnet50_fpn_best.pth (12-epoch, adopted) |
| maskrcnn_project/run_output/curves/, mask_metrics_{val,test}.json | LOST (box-based curves / old predictions; not used by the figures) |
| monai-project/checkpoints/{unified_best,jml619_best}.pth, scripts 05-10/16, .git | LOST here — re-obtain from the original source; src/ + scripts 12/13/14/17 are in scripts/original/monai/ |
| yolov12_project other runs (600images, predict-*, train_vinay_*), SAM 600images results | LOST |
| SAM2 code base (facebook sam2 clone) | re-clone from GitHub; sam2.1_hiera_large.pt is still at /u/sl257/data/checkpoint/ |
