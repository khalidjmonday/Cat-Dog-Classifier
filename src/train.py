import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    WeightedRandomSampler,
)
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights


# ============================================================
# CONFIGURATION
# ============================================================

CLEAN_DATA_DIR = Path("data/clean")
HARD_DATA_DIR = Path("data/hard_cases")

MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

# Hard examples receive extra probability during sampling.
# This makes the model see difficult images more often without
# physically duplicating files.
HARD_SAMPLE_MULTIPLIER = 5.0

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("CAT vs DOG CLASSIFIER")
print("=" * 70)

print(f"\nDevice: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )


# ============================================================
# DATA TRANSFORMS
# ============================================================

# ------------------------------------------------------------
# Normal training augmentation
# ------------------------------------------------------------
#
# Used for the original clean dataset.
#
# These augmentations improve robustness to:
# - crops
# - zoom
# - position changes
# - lighting
# - color
# - rotation
# ------------------------------------------------------------

train_transforms = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.70, 1.0),
    ),

    transforms.RandomHorizontalFlip(
        p=0.5,
    ),

    transforms.RandomRotation(
        15,
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ------------------------------------------------------------
# Hard-case training augmentation
# ------------------------------------------------------------
#
# IMPORTANT:
#
# Hard-case images often contain tiny or partially hidden
# animals. We therefore DO NOT use RandomResizedCrop here.
#
# Instead, preserve almost the entire image so that a small
# animal remains visible.
# ------------------------------------------------------------

hard_train_transforms = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(
        IMAGE_SIZE,
    ),

    transforms.RandomHorizontalFlip(
        p=0.5,
    ),

    transforms.RandomRotation(
        8,
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03,
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),

    # Small artificial occlusion.
    # This encourages the model not to depend on one specific
    # visible body region.
    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.08),
        ratio=(0.5, 2.0),
        value=0,
    ),
])


# ------------------------------------------------------------
# Validation / test transforms
# ------------------------------------------------------------

val_test_transforms = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(
        IMAGE_SIZE,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

clean_train_dir = CLEAN_DATA_DIR / "train"
clean_val_dir = CLEAN_DATA_DIR / "val"
clean_test_dir = CLEAN_DATA_DIR / "test"

hard_train_dir = HARD_DATA_DIR / "train"

required_dirs = [
    clean_train_dir,
    clean_val_dir,
    clean_test_dir,
    hard_train_dir,
]

for directory in required_dirs:
    if not directory.exists():
        raise FileNotFoundError(
            f"Required dataset directory not found:\n{directory}"
        )


# ------------------------------------------------------------
# Original clean training dataset
# ------------------------------------------------------------

clean_train_dataset = datasets.ImageFolder(
    clean_train_dir,
    transform=train_transforms,
)


# ------------------------------------------------------------
# Hard-case training dataset
# ------------------------------------------------------------

hard_train_dataset = datasets.ImageFolder(
    hard_train_dir,
    transform=hard_train_transforms,
)


# ------------------------------------------------------------
# Validation and normal test datasets
# ------------------------------------------------------------

val_dataset = datasets.ImageFolder(
    clean_val_dir,
    transform=val_test_transforms,
)

test_dataset = datasets.ImageFolder(
    clean_test_dir,
    transform=val_test_transforms,
)


# ------------------------------------------------------------
# Verify class mappings
# ------------------------------------------------------------

if clean_train_dataset.class_to_idx != hard_train_dataset.class_to_idx:
    raise RuntimeError(
        "Clean and hard training datasets have different "
        "class mappings."
    )

if clean_train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise RuntimeError(
        "Training and validation datasets have different "
        "class mappings."
    )

if clean_train_dataset.class_to_idx != test_dataset.class_to_idx:
    raise RuntimeError(
        "Training and test datasets have different "
        "class mappings."
    )


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

train_dataset = ConcatDataset([
    clean_train_dataset,
    hard_train_dataset,
])


print("\nDataset sizes:")

print(
    f"Normal training images : "
    f"{len(clean_train_dataset)}"
)

print(
    f"Hard training images   : "
    f"{len(hard_train_dataset)}"
)

print(
    f"Combined training      : "
    f"{len(train_dataset)}"
)

print(
    f"Validation images      : "
    f"{len(val_dataset)}"
)

print(
    f"Normal test images     : "
    f"{len(test_dataset)}"
)

print(
    f"Hard test images       : "
    f"{len(list((HARD_DATA_DIR / 'test').rglob('*')))}"
)

print(
    f"\nClasses: "
    f"{clean_train_dataset.classes}"
)

print(
    f"Class mapping: "
    f"{clean_train_dataset.class_to_idx}"
)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    MODEL_DIR / "class_names.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        clean_train_dataset.classes,
        f,
    )


# ============================================================
# TRAINING TARGETS
# ============================================================

clean_targets = list(
    clean_train_dataset.targets
)

hard_targets = list(
    hard_train_dataset.targets
)

combined_targets = (
    clean_targets
    + hard_targets
)


# ============================================================
# CLASS BALANCE + HARD CASE OVERSAMPLING
# ============================================================

class_counts = torch.bincount(
    torch.tensor(
        combined_targets,
        dtype=torch.long,
    ),
    minlength=len(
        clean_train_dataset.classes
    ),
).float()


print("\nCombined training class counts:")

for class_name, count in zip(
    clean_train_dataset.classes,
    class_counts,
):
    print(
        f"  {class_name}: "
        f"{int(count.item())}"
    )


# Inverse-frequency class weighting.
class_weights = 1.0 / class_counts


sample_weights = []

clean_count = len(
    clean_train_dataset
)

hard_count = len(
    hard_train_dataset
)

for index, label in enumerate(
    combined_targets
):

    weight = class_weights[label].item()

    # The first clean_count samples come from the normal dataset.
    # The remaining samples come from hard_cases/train.
    if index >= clean_count:
        weight *= HARD_SAMPLE_MULTIPLIER

    sample_weights.append(weight)


sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights,
)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

model.classifier[1] = nn.Sequential(
    nn.Dropout(
        p=0.3,
    ),

    nn.Linear(
        model.classifier[1].in_features,
        2,
    ),
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# TRAINING HISTORY
# ============================================================

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
}


# ============================================================
# TRAINING
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 70)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )


        scaler.scale(
            loss
        ).backward()

        scaler.step(
            optimizer
        )

        scaler.update()


        running_loss += (
            loss.item()
            * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1,
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100 * correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels,
                )


            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1,
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler.step(
        val_loss
    )


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nTrain Loss:       "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss:  "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Acc:   "
        f"{val_accuracy:.2f}%"
    )

    print(
        f"Learning Rate:    "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    history["val_loss"].append(
        val_loss
    )

    history["val_accuracy"].append(
        val_accuracy
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_names":
                    clean_train_dataset.classes,

                "image_size":
                    IMAGE_SIZE,

                "model_name":
                    "efficientnet_b0",

                "hard_sample_multiplier":
                    HARD_SAMPLE_MULTIPLIER,

                "normal_train_images":
                    len(clean_train_dataset),

                "hard_train_images":
                    len(hard_train_dataset),
            },

            MODEL_DIR / "best_model.pth",
        )

        print(
            "\n✓ New best model saved!"
        )

        print(
            f"  Validation accuracy: "
            f"{val_accuracy:.2f}%"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    RESULTS_DIR / "training_history.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        history,
        f,
        indent=4,
    )


# ============================================================
# TEST BEST MODEL ON NORMAL TEST SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL NORMAL TEST")
print("=" * 70)


checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_amp,
        ):

            outputs = model(images)


        _, predicted = torch.max(
            outputs,
            1,
        )

        test_total += labels.size(0)

        test_correct += (
            predicted == labels
        ).sum().item()


test_accuracy = (
    100 * test_correct / test_total
)

print(
    f"\nNormal Test Accuracy: "
    f"{test_accuracy:.2f}%"
)


# ============================================================
# FINAL HARD TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL HARD-CASE TEST")
print("=" * 70)


hard_test_dir = (
    HARD_DATA_DIR / "test"
)

if hard_test_dir.exists():

    hard_test_dataset = datasets.ImageFolder(
        hard_test_dir,
        transform=val_test_transforms,
    )

    if (
        hard_test_dataset.class_to_idx
        != clean_train_dataset.class_to_idx
    ):
        raise RuntimeError(
            "Hard test dataset has a different "
            "class mapping."
        )

    hard_test_loader = DataLoader(
        hard_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


    hard_correct = 0
    hard_total = 0

    with torch.no_grad():

        for images, labels in hard_test_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                outputs = model(images)


            _, predicted = torch.max(
                outputs,
                1,
            )

            hard_total += labels.size(0)

            hard_correct += (
                predicted == labels
            ).sum().item()


    hard_accuracy = (
        100 * hard_correct / hard_total
    )

    print(
        f"\nHard Test Accuracy: "
        f"{hard_accuracy:.2f}%"
    )

    print(
        f"Hard test images: "
        f"{hard_total}"
    )

else:

    hard_accuracy = None

    print(
        "\nHard test directory not found."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"\nBest Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"Normal Test Accuracy:     "
    f"{test_accuracy:.2f}%"
)

if hard_accuracy is not None:

    print(
        f"Hard Test Accuracy:       "
        f"{hard_accuracy:.2f}%"
    )

print(
    f"\nNormal training images: "
    f"{len(clean_train_dataset)}"
)

print(
    f"Hard training images:   "
    f"{len(hard_train_dataset)}"
)

print(
    f"Hard sample multiplier:  "
    f"{HARD_SAMPLE_MULTIPLIER}x"
)

print(
    f"\nBest model: "
    f"{MODEL_DIR / 'best_model.pth'}"
)

print(
    f"History:    "
    f"{RESULTS_DIR / 'training_history.json'}"
)

print("\nNo source dataset files were modified.")