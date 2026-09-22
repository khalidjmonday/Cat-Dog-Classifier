from pathlib import Path
import hashlib
import random
import shutil

import pandas as pd


# ============================================================
# Configuration
# ============================================================

ANNOTATIONS_FILE = Path(
    r"C:\Users\yasha\fiftyone\open-images-v7\validation\labels\detections.csv"
)

IMAGE_DIR = Path(
    r"C:\Users\yasha\fiftyone\open-images-v7\validation\data"
)

OUTPUT_DIR = Path("data/hard_cases")

CAT_LABEL = "/m/01yrx"
DOG_LABEL = "/m/0bt9lr"

SEED = 42

# We want balanced classes
MAX_PER_CLASS = 214

# 80/20 hard train/test split
TRAIN_RATIO = 0.80

# Small-object thresholds
SMALL_AREA = 0.10
VERY_SMALL_AREA = 0.05


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    """Return SHA-256 hash for a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def find_images():
    """Map Open Images IDs to downloaded image paths."""
    image_map = {}

    print("Scanning downloaded Open Images...")

    for extension in ("*.jpg", "*.jpeg", "*.png"):
        for path in IMAGE_DIR.rglob(extension):
            image_map[path.stem] = path

    print(f"Downloaded images found: {len(image_map)}")

    if not image_map:
        raise RuntimeError(
            f"No images found in:\n{IMAGE_DIR}"
        )

    return image_map


# ============================================================
# Load annotations
# ============================================================

print("=" * 70)
print("BUILDING HARD-CASE DATASET")
print("=" * 70)

image_map = find_images()

print("\nReading annotations...")

usecols = [
    "ImageID",
    "LabelName",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsOccluded",
    "IsTruncated",
    "IsGroupOf",
    "IsDepiction",
    "IsInside",
]

df = pd.read_csv(
    ANNOTATIONS_FILE,
    usecols=usecols,
)

print(f"Total annotation rows loaded: {len(df)}")


# ============================================================
# Keep only Cat / Dog
# ============================================================

df = df[
    df["LabelName"].isin(
        [CAT_LABEL, DOG_LABEL]
    )
].copy()

df = df[
    df["ImageID"].isin(image_map.keys())
].copy()

print(f"Cat/Dog annotation rows: {len(df)}")


# ============================================================
# Remove group/depiction annotations
# ============================================================

df["IsGroupOf"] = (
    pd.to_numeric(
        df["IsGroupOf"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

df["IsDepiction"] = (
    pd.to_numeric(
        df["IsDepiction"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

df["IsOccluded"] = (
    pd.to_numeric(
        df["IsOccluded"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

df["IsTruncated"] = (
    pd.to_numeric(
        df["IsTruncated"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)

df = df[
    (df["IsGroupOf"] == 0)
    & (df["IsDepiction"] == 0)
].copy()


# ============================================================
# Calculate bounding-box size
# ============================================================

df["box_width"] = (
    df["XMax"] - df["XMin"]
)

df["box_height"] = (
    df["YMax"] - df["YMin"]
)

df["box_area_fraction"] = (
    df["box_width"] * df["box_height"]
)


# ============================================================
# Remove ambiguous images containing BOTH cat and dog
# ============================================================

class_counts_per_image = (
    df.groupby("ImageID")["LabelName"]
    .nunique()
)

ambiguous_ids = set(
    class_counts_per_image[
        class_counts_per_image > 1
    ].index
)

print(
    f"\nAmbiguous images containing both Cat and Dog: "
    f"{len(ambiguous_ids)}"
)

df = df[
    ~df["ImageID"].isin(ambiguous_ids)
].copy()


# ============================================================
# Determine image-level hard-case score
# ============================================================

df["hard_score"] = (
    (df["IsOccluded"] == 1).astype(int) * 4
    + (df["IsTruncated"] == 1).astype(int) * 2
    + (df["box_area_fraction"] <= VERY_SMALL_AREA).astype(int) * 4
    + (df["box_area_fraction"] <= SMALL_AREA).astype(int) * 2
)


# ============================================================
# One record per image
# ============================================================

image_records = []

for image_id, group in df.groupby("ImageID"):

    row = group.loc[
        group["hard_score"].idxmax()
    ]

    label = (
        "cat"
        if row["LabelName"] == CAT_LABEL
        else "dog"
    )

    image_records.append(
        {
            "ImageID": image_id,
            "label": label,
            "score": int(row["hard_score"]),
            "box_area_fraction": float(
                row["box_area_fraction"]
            ),
            "is_occluded": int(
                row["IsOccluded"]
            ),
            "is_truncated": int(
                row["IsTruncated"]
            ),
            "path": image_map[image_id],
        }
    )

records = pd.DataFrame(image_records)


# ============================================================
# Keep only genuinely hard images
# ============================================================

records = records[
    (records["is_occluded"] == 1)
    | (records["is_truncated"] == 1)
    | (records["box_area_fraction"] <= SMALL_AREA)
].copy()


print(
    f"\nHard images after filtering: {len(records)}"
)

print("\nAvailable hard cases:")

print(
    records["label"]
    .value_counts()
    .to_string()
)


# ============================================================
# Build balanced class pools
# ============================================================

random.seed(SEED)

cat_records = records[
    records["label"] == "cat"
].sort_values(
    by=[
        "score",
        "box_area_fraction",
    ],
    ascending=[
        False,
        True,
    ],
).copy()

dog_records = records[
    records["label"] == "dog"
].sort_values(
    by=[
        "score",
        "box_area_fraction",
    ],
    ascending=[
        False,
        True,
    ],
).copy()


available_per_class = min(
    len(cat_records),
    len(dog_records),
    MAX_PER_CLASS,
)

if available_per_class < 100:
    raise RuntimeError(
        "Not enough balanced hard cases were found."
    )


cat_records = cat_records.head(
    available_per_class
).copy()

dog_records = dog_records.head(
    available_per_class
).copy()


print(
    f"\nBalanced dataset size per class: "
    f"{available_per_class}"
)


# ============================================================
# Shuffle before splitting
# ============================================================

cat_records = cat_records.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)

dog_records = dog_records.sample(
    frac=1.0,
    random_state=SEED,
).reset_index(drop=True)


def split_records(records_df):
    train_count = int(
        len(records_df) * TRAIN_RATIO
    )

    return (
        records_df.iloc[:train_count].copy(),
        records_df.iloc[train_count:].copy(),
    )


cat_train, cat_test = split_records(cat_records)
dog_train, dog_test = split_records(dog_records)


train_records = pd.concat(
    [cat_train, dog_train],
    ignore_index=True,
)

test_records = pd.concat(
    [cat_test, dog_test],
    ignore_index=True,
)


# ============================================================
# Clear previous hard-case dataset
# ============================================================

if OUTPUT_DIR.exists():
    print("\nRemoving previous hard-case dataset...")
    shutil.rmtree(OUTPUT_DIR)


for split in ("train", "test"):
    for label in ("cat", "dog"):
        (
            OUTPUT_DIR
            / split
            / label
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Copy images
# ============================================================

def copy_records(records_df, split_name):
    copied = 0

    for _, row in records_df.iterrows():

        source = Path(row["path"])

        destination = (
            OUTPUT_DIR
            / split_name
            / row["label"]
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied += 1

    return copied


train_copied = copy_records(
    train_records,
    "train",
)

test_copied = copy_records(
    test_records,
    "test",
)


# ============================================================
# Save metadata
# ============================================================

metadata = pd.concat(
    [
        train_records.assign(split="train"),
        test_records.assign(split="test"),
    ],
    ignore_index=True,
)

metadata[
    [
        "ImageID",
        "split",
        "label",
        "score",
        "box_area_fraction",
        "is_occluded",
        "is_truncated",
    ]
].to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("HARD-CASE DATASET CREATED")
print("=" * 70)

print("\nTraining:")
print(
    train_records["label"]
    .value_counts()
    .to_string()
)

print("\nHard Test:")
print(
    test_records["label"]
    .value_counts()
    .to_string()
)

print(f"\nTraining images copied: {train_copied}")
print(f"Hard-test images copied: {test_copied}")

print(
    f"\nDataset location:\n{OUTPUT_DIR}"
)

print("\nMetadata:")
print(
    OUTPUT_DIR / "metadata.csv"
)

print("\nNo existing clean dataset was modified.")
print("No model was modified.")