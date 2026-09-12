from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

PAIRS = [
    (
        "data/clean/train/cat/cat_000046.jpg",
        "data/clean/val/cat/cat_000241.jpg",
    ),
    (
        "data/clean/train/cat/cat_001672.jpg",
        "data/clean/val/cat/cat_000713.jpg",
    ),
    (
        "data/clean/train/cat/cat_001915.jpg",
        "data/clean/val/cat/cat_000289.jpg",
    ),
    (
        "data/clean/train/cat/cat_002040.jpg",
        "data/clean/val/cat/cat_000178.jpg",
    ),
    (
        "data/clean/train/cat/cat_002336.jpg",
        "data/clean/val/cat/cat_000241.jpg",
    ),
    (
        "data/clean/train/cat/cat_002648.jpg",
        "data/clean/val/cat/cat_000184.jpg",
    ),
    (
        "data/clean/train/cat/cat_003385.jpg",
        "data/clean/val/cat/cat_000799.jpg",
    ),
    (
        "data/clean/train/cat/cat_003413.jpg",
        "data/clean/val/cat/cat_000819.jpg",
    ),
    (
        "data/clean/train/cat/cat_003455.jpg",
        "data/clean/val/cat/cat_000514.jpg",
    ),
    (
        "data/clean/train/cat/cat_003624.jpg",
        "data/clean/val/cat/cat_001008.jpg",
    ),
    (
        "data/clean/train/cat/cat_004065.jpg",
        "data/clean/val/cat/cat_000291.jpg",
    ),
    (
        "data/clean/train/cat/cat_004188.jpg",
        "data/clean/val/cat/cat_000531.jpg",
    ),
    (
        "data/clean/train/cat/cat_004353.jpg",
        "data/clean/val/cat/cat_000152.jpg",
    ),
    (
        "data/clean/train/cat/cat_004733.jpg",
        "data/clean/val/cat/cat_000034.jpg",
    ),
    (
        "data/clean/train/cat/cat_004745.jpg",
        "data/clean/val/cat/cat_000785.jpg",
    ),
]


def show_pair(train_path, val_path, number):
    train_img = Image.open(train_path).convert("RGB")
    val_img = Image.open(val_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(train_img)
    axes[0].set_title(f"TRAIN\n{Path(train_path).name}")
    axes[0].axis("off")

    axes[1].imshow(val_img)
    axes[1].set_title(f"VALIDATION\n{Path(val_path).name}")
    axes[1].axis("off")

    fig.suptitle(f"Near-Duplicate Pair {number}", fontsize=14)
    plt.tight_layout()

    output = Path("results") / f"near_duplicate_{number}.png"
    output.parent.mkdir(exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.show()
    plt.close()


print("=" * 60)
print("NEAR-DUPLICATE VISUAL INSPECTION")
print("=" * 60)

for i, (train_path, val_path) in enumerate(PAIRS, start=1):

    if not Path(train_path).exists():
        print(f"\nMissing: {train_path}")
        continue

    if not Path(val_path).exists():
        print(f"\nMissing: {val_path}")
        continue

    print(f"\nPair {i}")
    print(f"TRAIN: {train_path}")
    print(f"VAL  : {val_path}")

    show_pair(train_path, val_path, i)

print("\nInspection complete.")
print("Images saved in: results/")