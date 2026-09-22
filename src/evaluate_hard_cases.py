from pathlib import Path
import json

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/hard_cases/test")
MODEL_PATH = Path("models/best_model.pth")
RESULTS_DIR = Path("results")

IMAGE_SIZE = 224
BATCH_SIZE = 32

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("HARD-CASE TEST EVALUATION")
print("=" * 70)

print(f"\nDevice: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Check files
# ============================================================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Hard test dataset not found:\n{DATA_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )


# ============================================================
# Test transforms
# Must match current predict.py
# ============================================================

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Dataset
# ============================================================

print("\nLoading hard test dataset...")

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=test_transforms,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(f"Images: {len(dataset)}")
print(f"Classes: {dataset.classes}")
print(f"Mapping: {dataset.class_to_idx}")


# ============================================================
# Model
# ============================================================

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(
        model.classifier[1].in_features,
        2,
    ),
)


# ============================================================
# Load checkpoint
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)


if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(device)
model.eval()


# ============================================================
# Evaluation
# ============================================================

all_targets = []
all_predictions = []
all_probabilities = []
all_paths = []

print("\nRunning hard-case evaluation...")


with torch.no_grad():

    for images, targets in loader:

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        all_targets.extend(
            targets.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    all_targets,
    all_predictions,
)

precision = precision_score(
    all_targets,
    all_predictions,
    average="binary",
    zero_division=0,
)

recall = recall_score(
    all_targets,
    all_predictions,
    average="binary",
    zero_division=0,
)

f1 = f1_score(
    all_targets,
    all_predictions,
    average="binary",
    zero_division=0,
)

cm = confusion_matrix(
    all_targets,
    all_predictions,
)

report = classification_report(
    all_targets,
    all_predictions,
    target_names=dataset.classes,
    zero_division=0,
)


# ============================================================
# Find misclassified files
# ============================================================

misclassified = []

for index, (target, prediction) in enumerate(
    zip(all_targets, all_predictions)
):

    if target != prediction:

        image_path = dataset.samples[index][0]

        probabilities = all_probabilities[index]

        misclassified.append(
            {
                "image": image_path,
                "actual": dataset.classes[target],
                "predicted": dataset.classes[prediction],
                "cat_probability": round(
                    float(probabilities[0]) * 100,
                    2,
                ),
                "dog_probability": round(
                    float(probabilities[1]) * 100,
                    2,
                ),
            }
        )


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("HARD-CASE RESULTS")
print("=" * 70)

print(f"\nAccuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

print(
    f"Misclassified images: "
    f"{len(misclassified)} / {len(dataset)}"
)


# ============================================================
# Print misclassified images
# ============================================================

if misclassified:

    print("\nMisclassified examples:")

    for item in misclassified:

        print(
            f"\nImage      : {item['image']}"
            f"\nActual     : {item['actual']}"
            f"\nPredicted  : {item['predicted']}"
            f"\nCAT        : {item['cat_probability']:.2f}%"
            f"\nDOG        : {item['dog_probability']:.2f}%"
        )


# ============================================================
# Save results
# ============================================================

results = {
    "dataset": "data/hard_cases/test",
    "images": len(dataset),
    "classes": dataset.classes,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": cm.tolist(),
    "misclassified_count": len(misclassified),
    "misclassified": misclassified,
}


output_file = (
    RESULTS_DIR / "hard_case_baseline.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
    )


print(
    f"\nResults saved to:\n{output_file}"
)

print("\n" + "=" * 70)
print("BASELINE EVALUATION COMPLETE")
print("=" * 70)

print(
    "\nNo training data was modified."
)

print(
    "No model was modified."
)