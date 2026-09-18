Verbatim copies (2026-09-16) of the scripts that produced the numbers in
data/. Kept for reference only -- the re-run scripts in scripts/rerun/ import
from these copies so this folder is self-contained.

maskrcnn/  from /u/sl257/shared_data/train_script/maskrcnn_project/   (written/run by am232, conda env tors_env)
yolo/      from /u/sl257/shared_data/train_script/yolov12_project/     (conda env pytorch_env)
sam/       from /u/sl257/shared_data/train_script/SAM2_project/sam2/   (conda env pytorch_env, HF_HOME=/u/sl257/scratch/huggingface)
monai/     from /u/sl257/shared_data/train_script/monai-project/         (src/ package + the unified prepare/train/eval/instance-AP scripts;
           checkpoints stay there: checkpoints/unified_best.pth = the model behind the gathered monai-results. Needs monai==1.5.1 with torch 2.6)
