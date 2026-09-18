from ultralytics import YOLO

from result_utils import save_test_results

dataPath = "/projects/illinois/cimed/bts/gayed/data/processed_data/yolo_format/1200images/data.yaml"
device = "cpu"

projectDir = "/u/sl257/shared_data/train_script/yolov12_project/runs/segment"
runName = "1200images"

weightsPath = (
    "/u/sl257/shared_data/train_script/yolov12_project/runs/segment/"
    "1200images/1200images_train/weights/best.pt"
)

model = YOLO(weightsPath)

metrics = model.val(
    data=dataPath,
    split="test",
    batch=32,
    device=device,
    project=projectDir,
    name=f"{runName}_test",
)

print("\n=== Test metrics ===")
for k, v in metrics.results_dict.items():
    print(f"{k}: {v:.4f}")

csv_path, png_path = save_test_results(metrics)
print(f"Saved {csv_path} and {png_path}")
