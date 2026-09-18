from ultralytics import YOLO

from result_utils import save_test_results

# real dataset 60ish images
dataPath = "/projects/illinois/cimed/bts/gayed/data/processed_data/yolo_format/1200images/data.yaml"
device = 0

projectDir = "/u/sl257/shared_data/train_script/yolov12_project/runs/segment"
runName = "1200images"

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo11n-seg.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(
    data=dataPath,
    epochs=60,
    imgsz=640,
    batch=16,     # or even 64 if memory allows
    device=device,
    workers=8,
    project=projectDir,
    name=f"{runName}_train",
    )

# Run inference with the YOLO12n model on the 'bus.jpg' image
# results = model("/u/slee32/data/yolosegdata/kvasir_yolo/images/val/ckcxaqyt900073b5yomyxi3bf.jpg")

print("predict on VAL SET")

model.predict(
    source="/projects/illinois/cimed/bts/gayed/data/processed_data/yolo_format/1200images/images/val/",
    batch=32,
    device=device,
    save=True,
    save_txt=True,
    project=projectDir,
    name=f"{runName}_val_predictions",
)

print("predict on TEST SET")

model.predict(
    source="/projects/illinois/cimed/bts/gayed/data/processed_data/yolo_format/1200images/images/test/",
    batch=32,
    device=device,
    save=True,
    save_txt=True,
    project=projectDir,
    name=f"{runName}_test_predictions",
)

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