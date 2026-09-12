from pathlib import Path
import random
import shutil

# -----------------------------
# SETTINGS
# -----------------------------

CAT_SOURCE = Path("data/raw/cats")
DOG_SOURCE = Path("data/raw/dogs")
OUTPUT = Path("data/processed")

SEED = 42
random.seed(SEED)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def get_images(folder):
    """Return all image files inside a folder."""
    return [
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_images(images, destination):
    """Copy images to destination."""
    destination.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images):
        new_name = f"{index:05d}_{image.name}"
        shutil.copy2(image, destination / new_name)


def split_images(images, train_ratio=0.8, val_ratio=0.1):
    """Split images into train, validation and test."""
    images = images.copy()
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


# -----------------------------
# FIND CAT IMAGES
# -----------------------------

print("\nScanning cat dataset...")

cat_images = get_images(CAT_SOURCE)

print(f"Found {len(cat_images)} cat images.")


# -----------------------------
# FIND DOG IMAGES
# -----------------------------

print("\nScanning dog dataset...")

dog_train = get_images(DOG_SOURCE / "train")
dog_val = get_images(DOG_SOURCE / "valid")
dog_test = get_images(DOG_SOURCE / "test")

print(f"Dog train images: {len(dog_train)}")
print(f"Dog validation images: {len(dog_val)}")
print(f"Dog test images: {len(dog_test)}")


# -----------------------------
# SPLIT CATS
# -----------------------------

print("\nSplitting cat images...")

cat_train, cat_val, cat_test = split_images(cat_images)

print(f"Cat train: {len(cat_train)}")
print(f"Cat validation: {len(cat_val)}")
print(f"Cat test: {len(cat_test)}")


# -----------------------------
# CREATE OUTPUT DIRECTORIES
# -----------------------------

print("\nCreating processed dataset...")

for split in ["train", "val", "test"]:
    for animal in ["cat", "dog"]:
        folder = OUTPUT / split / animal
        folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# COPY CATS
# -----------------------------

print("\nCopying cat images...")

copy_images(cat_train, OUTPUT / "train" / "cat")
copy_images(cat_val, OUTPUT / "val" / "cat")
copy_images(cat_test, OUTPUT / "test" / "cat")


# -----------------------------
# COPY DOGS
# -----------------------------

print("Copying dog images...")

copy_images(dog_train, OUTPUT / "train" / "dog")
copy_images(dog_val, OUTPUT / "val" / "dog")
copy_images(dog_test, OUTPUT / "test" / "dog")


# -----------------------------
# FINAL SUMMARY
# -----------------------------

print("\n==============================")
print("DATASET PREPARATION COMPLETE")
print("==============================")

print(f"Train:")
print(f"  Cat: {len(cat_train)}")
print(f"  Dog: {len(dog_train)}")

print(f"\nValidation:")
print(f"  Cat: {len(cat_val)}")
print(f"  Dog: {len(dog_val)}")

print(f"\nTest:")
print(f"  Cat: {len(cat_test)}")
print(f"  Dog: {len(dog_test)}")

print("\nDataset location:")
print(OUTPUT)