import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/clean")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CAT vs DOG CLASSIFIER")
print("=" * 60)

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

# Training augmentation is intentionally stronger than before.
# The goal is to make the model robust to different:
# - image crops
# - zoom levels
# - positions
# - lighting
# - colors
# - orientations
#
# This helps reduce dependence on dataset-specific visual patterns.

train_transforms = transforms.Compose([
    transforms.Resize(256),

    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.70, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and test data are NOT randomly augmented.
# They use deterministic preprocessing.

val_test_transforms = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    DATA_DIR / "train",
    transform=train_transforms
)

val_dataset = datasets.ImageFolder(
    DATA_DIR / "val",
    transform=val_test_transforms
)

test_dataset = datasets.ImageFolder(
    DATA_DIR / "test",
    transform=val_test_transforms
)

print(f"Training images:   {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Test images:       {len(test_dataset)}")

print(f"\nClasses: {train_dataset.classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(MODEL_DIR / "class_names.json", "w") as f:
    json.dump(train_dataset.classes, f)


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

class_counts = torch.bincount(
    torch.tensor(train_dataset.targets)
)

print("\nTraining class counts:")

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):
    print(f"  {class_name}: {count.item()}")


# Give less frequent classes more sampling weight
class_weights = 1.0 / class_counts.float()

sample_weights = [
    class_weights[label].item()
    for label in train_dataset.targets
]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)


# ============================================================
# REPLACE FINAL CLASSIFIER
# ============================================================

model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(
        model.classifier[1].in_features,
        2
    )
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

# Label smoothing reduces extreme overconfidence and
# encourages better generalization.

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = torch.cuda.is_available()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("=" * 60)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):

            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=use_amp
            ):

                outputs = model(images)
                loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(f"\nTrain Loss:       {train_loss:.4f}")
    print(f"Train Accuracy:   {train_accuracy:.2f}%")

    print(f"Validation Loss:  {val_loss:.4f}")
    print(f"Validation Acc:   {val_accuracy:.2f}%")

    print(
        f"Learning Rate:    "
        f"{optimizer.param_groups[0]['lr']:.6f}"
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": train_dataset.classes,
                "image_size": IMAGE_SIZE,
                "model_name": "efficientnet_b0"
            },
            MODEL_DIR / "best_model.pth"
        )

        print(
            f"\n✓ New best model saved!"
            f" Validation accuracy: {val_accuracy:.2f}%"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(RESULTS_DIR / "training_history.json", "w") as f:
    json.dump(history, f, indent=4)


# ============================================================
# TEST BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST")
print("=" * 60)

checkpoint = torch.load(
    MODEL_DIR / "best_model.pth",
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=use_amp
        ):
            outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()


test_accuracy = 100 * test_correct / test_total

print(f"\nTest Accuracy: {test_accuracy:.2f}%")

print("\nTraining complete.")
print(f"Best model: {MODEL_DIR / 'best_model.pth'}")
print(f"History:    {RESULTS_DIR / 'training_history.json'}")