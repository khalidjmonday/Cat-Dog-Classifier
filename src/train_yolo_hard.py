from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_YAML = Path(
    "data/yolo_hard/data.yaml"
)

BASE_MODEL = "yolo11n.pt"

PROJECT_DIR = Path(
    "runs/detect"
)

RUN_NAME = "cat_dog_hard_yolo11n"

EPOCHS = 60

# GTX 1650 4 GB:
# Start conservatively.
BATCH_SIZE = 4

IMAGE_SIZE = 960

DEVICE = 0

WORKERS = 0

PATIENCE = 15


# ============================================================
# VALIDATION
# ============================================================

if not DATASET_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATASET_YAML}"
    )

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected."
    )


# ============================================================
# DEVICE INFO
# ============================================================

print("=" * 70)
print("YOLO11 HARD-CASE TRAINING")
print("=" * 70)

print(
    f"\nGPU: "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"GPU Memory: "
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

print(
    f"\nDataset: "
    f"{DATASET_YAML}"
)

print(
    f"Base model: "
    f"{BASE_MODEL}"
)

print(
    f"Image size: "
    f"{IMAGE_SIZE}"
)

print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)

print(
    f"Epochs: "
    f"{EPOCHS}"
)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\nLoading pretrained YOLO11n...")

model = YOLO(
    BASE_MODEL
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...")

results = model.train(
    data=str(DATASET_YAML),

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    device=DEVICE,

    workers=WORKERS,

    patience=PATIENCE,

    project=str(PROJECT_DIR),

    name=RUN_NAME,

    exist_ok=True,

    pretrained=True,

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    optimizer="AdamW",

    lr0=0.0005,

    lrf=0.01,

    weight_decay=0.0005,

    warmup_epochs=3.0,

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------
    #
    # We deliberately disable Mosaic and MixUp here.
    #
    # Our objective is small / partially hidden animals.
    # Combining multiple images can make an already-small
    # animal even smaller.
    #

    mosaic=0.0,

    mixup=0.0,

    copy_paste=0.0,

    degrees=8.0,

    translate=0.05,

    scale=0.20,

    shear=2.0,

    perspective=0.0,

    fliplr=0.5,

    flipud=0.0,

    hsv_h=0.015,

    hsv_s=0.5,

    hsv_v=0.3,

    # --------------------------------------------------------
    # Saving / validation
    # --------------------------------------------------------

    save=True,

    save_period=10,

    val=True,

    plots=True,

    verbose=True,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("YOLO11 HARD-CASE TRAINING COMPLETE")
print("=" * 70)

print(
    "\nTraining output:"
)

print(
    f"{PROJECT_DIR / RUN_NAME}"
)

weights_dir = (
    PROJECT_DIR
    / RUN_NAME
    / "weights"
)

best_model = (
    weights_dir
    / "best.pt"
)

last_model = (
    weights_dir
    / "last.pt"
)


print(
    f"\nBest model:"
    f"\n{best_model}"
)

print(
    f"\nLast model:"
    f"\n{last_model}"
)

print(
    "\nNext step will be evaluating the trained detector"
    " on the untouched 84-image hard test set."
)

print(
    "We will also test blur12.jpg again."
)

print("\n" + "=" * 70)