from pathlib import Path
from PIL import Image
import imagehash
import shutil


# ============================================================
# CONFIG
# ============================================================

DATASET = Path("data/clean")

SPLITS = ["train", "val", "test"]

# Distance 0 means the perceptual hashes are identical.
# These are the safest candidates to remove automatically.
MAX_DISTANCE = 0


# ============================================================
# HASH IMAGE
# ============================================================

def get_hash(image_path):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(split, class_name):
    folder = DATASET / split / class_name

    if not folder.exists():
        return []

    return list(folder.glob("*.jpg"))


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("AUTOMATIC NEAR-DUPLICATE CLEANUP")
print("=" * 70)

print("\nOnly distance-0 duplicates will be removed.")
print("Validation/test copies will be kept.")
print("Training copies will be removed.\n")


# ------------------------------------------------------------
# Create hashes
# ------------------------------------------------------------

train_images = []

for class_name in ["cat", "dog"]:
    train_images.extend(
        [(path, class_name) for path in collect_images("train", class_name)]
    )

val_images = []

for class_name in ["cat", "dog"]:
    val_images.extend(
        [(path, class_name) for path in collect_images("val", class_name)]
    )

test_images = []

for class_name in ["cat", "dog"]:
    test_images.extend(
        [(path, class_name) for path in collect_images("test", class_name)]
    )


print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print(f"Test images: {len(test_images)}")


# ------------------------------------------------------------
# Hash validation and test first
# ------------------------------------------------------------

print("\nHashing validation images...")

val_hashes = []

for path, class_name in val_images:
    h = get_hash(path)

    if h is not None:
        val_hashes.append((path, class_name, h))


print("Hashing test images...")

test_hashes = []

for path, class_name in test_images:
    h = get_hash(path)

    if h is not None:
        test_hashes.append((path, class_name, h))


# ------------------------------------------------------------
# Compare TRAIN against VAL/TEST
# ------------------------------------------------------------

print("\nHashing train images...")

removed = []

for index, (train_path, train_class) in enumerate(train_images):

    train_hash = get_hash(train_path)

    if train_hash is None:
        continue

    duplicate_found = False

    # Compare with validation
    for val_path, val_class, val_hash in val_hashes:

        if train_class != val_class:
            continue

        distance = train_hash - val_hash

        if distance <= MAX_DISTANCE:
            print("\nDUPLICATE FOUND")
            print(f"Train: {train_path}")
            print(f"Val  : {val_path}")
            print(f"Distance: {distance}")

            removed.append(train_path)
            duplicate_found = True
            break

    if duplicate_found:
        continue

    # Compare with test
    for test_path, test_class, test_hash in test_hashes:

        if train_class != test_class:
            continue

        distance = train_hash - test_hash

        if distance <= MAX_DISTANCE:
            print("\nDUPLICATE FOUND")
            print(f"Train: {train_path}")
            print(f"Test : {test_path}")
            print(f"Distance: {distance}")

            removed.append(train_path)
            duplicate_found = True
            break


# ============================================================
# DELETE DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("REMOVING DUPLICATES")
print("=" * 70)

for path in removed:

    if path.exists():
        path.unlink()
        print(f"Removed: {path}")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANUP COMPLETE")
print("=" * 70)

print(f"\nDuplicates removed: {len(removed)}")

if len(removed) == 0:
    print("No distance-0 duplicates found.")

print("\nValidation and test images were NOT deleted.")
print("Only training copies were removed.")