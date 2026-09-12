from pathlib import Path
import hashlib
import shutil

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/clean")

SPLITS = ["train", "val", "test"]
CLASSES = ["cat", "dog"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

# New dataset split
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# FILE HASH
# ============================================================

def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# FIND ALL IMAGES
# ============================================================

def get_all_images(class_name):

    images = []

    for split in SPLITS:

        folder = SOURCE_DIR / split / class_name

        if not folder.exists():
            continue

        for path in folder.rglob("*"):

            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

    return images


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

print("=" * 70)
print("CREATING CLEAN DATASET")
print("=" * 70)

print("\nOld dataset:")
print(f"  {SOURCE_DIR}")

print("\nNew dataset:")
print(f"  {OUTPUT_DIR}")


if OUTPUT_DIR.exists():

    print(
        "\nRemoving previous clean dataset..."
    )

    shutil.rmtree(OUTPUT_DIR)


for split in ["train", "val", "test"]:

    for class_name in CLASSES:

        (OUTPUT_DIR / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_original = 0
total_unique = 0
total_duplicates = 0


for class_name in CLASSES:

    print("\n" + "=" * 70)
    print(f"PROCESSING: {class_name.upper()}")
    print("=" * 70)

    images = get_all_images(class_name)

    print(
        f"\nOriginal images found: {len(images)}"
    )

    total_original += len(images)


    # --------------------------------------------------------
    # REMOVE EXACT DUPLICATES
    # --------------------------------------------------------

    unique_images = []
    seen_hashes = {}

    duplicate_count = 0

    for path in images:

        try:
            image_hash = file_hash(path)

        except Exception as e:

            print(
                f"\nCould not read: {path}"
            )

            print(f"Reason: {e}")

            continue


        if image_hash in seen_hashes:

            duplicate_count += 1

        else:

            seen_hashes[image_hash] = path
            unique_images.append(path)


    print(
        f"Exact duplicates removed: "
        f"{duplicate_count}"
    )

    print(
        f"Unique images remaining: "
        f"{len(unique_images)}"
    )


    total_duplicates += duplicate_count
    total_unique += len(unique_images)


    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    train_images, temp_images = train_test_split(
        unique_images,
        test_size=(VAL_SIZE + TEST_SIZE),
        random_state=42,
        shuffle=True
    )


    val_images, test_images = train_test_split(
        temp_images,
        test_size=(
            TEST_SIZE / (VAL_SIZE + TEST_SIZE)
        ),
        random_state=42,
        shuffle=True
    )


    print("\nNew split:")

    print(
        f"  Train: {len(train_images)}"
    )

    print(
        f"  Validation: {len(val_images)}"
    )

    print(
        f"  Test: {len(test_images)}"
    )


    # --------------------------------------------------------
    # COPY FILES
    # --------------------------------------------------------

    split_data = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }


    for split, file_list in split_data.items():

        destination = (
            OUTPUT_DIR
            / split
            / class_name
        )


        for index, source in enumerate(file_list):

            # Give files unique names so different
            # source folders cannot collide.
            new_name = (
                f"{class_name}_{index:06d}"
                f"{source.suffix.lower()}"
            )

            destination_file = (
                destination / new_name
            )

            shutil.copy2(
                source,
                destination_file
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEAN DATASET COMPLETE")
print("=" * 70)

print(
    f"\nOriginal images: {total_original}"
)

print(
    f"Exact duplicates removed: "
    f"{total_duplicates}"
)

print(
    f"Unique images: {total_unique}"
)


# ============================================================
# COUNT OUTPUT DATASET
# ============================================================

print("\nFinal dataset:")

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}:")

    split_total = 0

    for class_name in CLASSES:

        folder = (
            OUTPUT_DIR
            / split
            / class_name
        )

        count = sum(
            1
            for path in folder.iterdir()
            if path.is_file()
        )

        split_total += count

        print(
            f"  {class_name}: {count}"
        )

    print(
        f"  Total: {split_total}"
    )


print("\nDataset location:")

print(
    f"  {OUTPUT_DIR}"
)

print("\nNext step:")

print(
    "Update train.py to use data/clean "
    "instead of data/processed."
)

print("\nDone.")