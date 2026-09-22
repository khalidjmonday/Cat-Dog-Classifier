from pathlib import Path
import random
import shutil

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

HARD_DATASET = Path("data/hard_cases")
METADATA_FILE = HARD_DATASET / "metadata.csv"

YOLO_DATASET = Path("data/yolo_hard")

ANNOTATIONS_FILE = Path(
    r"C:\Users\yasha\fiftyone\open-images-v7\validation\labels\detections.csv"
)

CAT_LABEL = "/m/01yrx"
DOG_LABEL = "/m/0bt9lr"

# Use 20% of the 336 hard-training images as an internal
# validation set. The official 84-image hard test remains
# completely untouched.
VAL_RATIO = 0.20

SEED = 42


# ============================================================
# SETUP
# ============================================================

print("=" * 70)
print("BUILDING YOLO HARD-CASE DATASET")
print("=" * 70)

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(
        f"Open Images annotation file not found:\n{ANNOTATIONS_FILE}"
    )


# ============================================================
# LOAD HARD-CASE METADATA
# ============================================================

metadata = pd.read_csv(
    METADATA_FILE
)

print(
    f"\nHard-case metadata rows: "
    f"{len(metadata)}"
)

train_metadata = metadata[
    metadata["split"] == "train"
].copy()

test_metadata = metadata[
    metadata["split"] == "test"
].copy()

print(
    f"Hard training images: "
    f"{len(train_metadata)}"
)

print(
    f"Hard test images: "
    f"{len(test_metadata)}"
)


# ============================================================
# MAKE INTERNAL TRAIN / VALIDATION SPLIT
# ============================================================

random.seed(SEED)

train_ids = train_metadata[
    "ImageID"
].tolist()

random.shuffle(
    train_ids
)

val_count = int(
    len(train_ids) * VAL_RATIO
)

val_ids = set(
    train_ids[:val_count]
)

yolo_train_ids = set(
    train_ids[val_count:]
)

hard_test_ids = set(
    test_metadata["ImageID"].tolist()
)

print(
    f"\nYOLO training images: "
    f"{len(yolo_train_ids)}"
)

print(
    f"YOLO validation images: "
    f"{len(val_ids)}"
)

print(
    f"Untouched hard-test images: "
    f"{len(hard_test_ids)}"
)


# ============================================================
# IMAGE LOOKUP
# ============================================================

def find_hard_image(image_id, split):
    """
    Find an image in data/hard_cases/{split}/{cat|dog}.
    """

    root = (
        HARD_DATASET
        / split
    )

    matches = list(
        root.rglob(
            f"{image_id}.*"
        )
    )

    if not matches:
        return None

    return matches[0]


# ============================================================
# CREATE YOLO DIRECTORIES
# ============================================================

if YOLO_DATASET.exists():
    print(
        "\nRemoving previous YOLO hard dataset..."
    )

    shutil.rmtree(
        YOLO_DATASET
    )


for split in [
    "train",
    "val",
]:

    (
        YOLO_DATASET
        / "images"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        YOLO_DATASET
        / "labels"
        / split
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# READ OPEN IMAGES ANNOTATIONS
# ============================================================

print(
    "\nReading Open Images annotations..."
)

usecols = [
    "ImageID",
    "LabelName",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsGroupOf",
    "IsDepiction",
]

target_ids = (
    yolo_train_ids
    | val_ids
    | hard_test_ids
)

annotation_chunks = []

for chunk in pd.read_csv(
    ANNOTATIONS_FILE,
    usecols=usecols,
    chunksize=100_000,
):

    chunk = chunk[
        chunk["ImageID"].isin(
            target_ids
        )
    ]

    chunk = chunk[
        chunk["LabelName"].isin(
            [
                CAT_LABEL,
                DOG_LABEL,
            ]
        )
    ]

    if not chunk.empty:
        annotation_chunks.append(
            chunk.copy()
        )


if not annotation_chunks:
    raise RuntimeError(
        "No Cat/Dog annotations found."
    )


annotations = pd.concat(
    annotation_chunks,
    ignore_index=True,
)

print(
    f"Relevant annotations found: "
    f"{len(annotations)}"
)


# ============================================================
# REMOVE GROUP / DEPICTION OBJECTS
# ============================================================

annotations["IsGroupOf"] = (
    pd.to_numeric(
        annotations["IsGroupOf"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

annotations["IsDepiction"] = (
    pd.to_numeric(
        annotations["IsDepiction"],
        errors="coerce",
    )
    .fillna(0)
    .astype(int)
)

annotations = annotations[
    (annotations["IsGroupOf"] == 0)
    &
    (annotations["IsDepiction"] == 0)
].copy()


# ============================================================
# REMOVE ANY IMAGE CONTAINING BOTH CAT AND DOG
# ============================================================

per_image_classes = (
    annotations.groupby("ImageID")[
        "LabelName"
    ]
    .nunique()
)

ambiguous = set(
    per_image_classes[
        per_image_classes > 1
    ].index
)

if ambiguous:

    print(
        f"\nRemoving "
        f"{len(ambiguous)} "
        f"ambiguous mixed Cat/Dog images."
    )

    annotations = annotations[
        ~annotations["ImageID"].isin(
            ambiguous
        )
    ].copy()


# ============================================================
# YOLO LABEL CONVERSION
# ============================================================

class_map = {
    CAT_LABEL: 0,
    DOG_LABEL: 1,
}


def make_yolo_line(row):
    """
    Convert Open Images normalized XYXY box to YOLO:
        class x_center y_center width height
    """

    x_min = float(row["XMin"])
    x_max = float(row["XMax"])

    y_min = float(row["YMin"])
    y_max = float(row["YMax"])

    width = x_max - x_min
    height = y_max - y_min

    x_center = (
        x_min + x_max
    ) / 2.0

    y_center = (
        y_min + y_max
    ) / 2.0

    class_id = class_map[
        row["LabelName"]
    ]

    return (
        f"{class_id} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# ============================================================
# BUILD SPLIT
# ============================================================

def build_split(
    image_ids,
    source_split,
    output_split,
):

    image_ids = set(
        image_ids
    )

    subset = annotations[
        annotations["ImageID"].isin(
            image_ids
        )
    ].copy()

    grouped = subset.groupby(
        "ImageID"
    )

    copied_images = 0
    label_files = 0
    missing_images = 0

    print(
        f"\nBuilding {output_split}..."
    )

    for image_id in sorted(
        image_ids
    ):

        source_image = find_hard_image(
            image_id,
            source_split,
        )

        if source_image is None:

            print(
                f"WARNING: image not found: "
                f"{image_id}"
            )

            missing_images += 1
            continue


        if image_id not in grouped.groups:

            print(
                f"WARNING: no annotation found: "
                f"{image_id}"
            )

            continue


        image_destination = (
            YOLO_DATASET
            / "images"
            / output_split
            / source_image.name
        )

        label_destination = (
            YOLO_DATASET
            / "labels"
            / output_split
            / f"{source_image.stem}.txt"
        )


        # Copy image
        shutil.copy2(
            source_image,
            image_destination,
        )

        copied_images += 1


        # Write ALL valid Cat/Dog boxes
        rows = grouped.get_group(
            image_id
        )

        lines = []

        for _, row in rows.iterrows():

            lines.append(
                make_yolo_line(row)
            )


        if lines:

            with open(
                label_destination,
                "w",
                encoding="utf-8",
            ) as f:

                f.write(
                    "\n".join(lines)
                )

            label_files += 1


    print(
        f"Images copied: {copied_images}"
    )

    print(
        f"Label files created: {label_files}"
    )

    print(
        f"Missing images: {missing_images}"
    )

    return copied_images


# ============================================================
# BUILD TRAIN / VAL
# ============================================================

train_copied = build_split(
    yolo_train_ids,
    "train",
    "train",
)

val_copied = build_split(
    val_ids,
    "train",
    "val",
)


# ============================================================
# CREATE DATA YAML
# ============================================================

yaml_file = (
    YOLO_DATASET
    / "data.yaml"
)

yaml_content = f"""path: {YOLO_DATASET.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: cat
  1: dog
"""

with open(
    yaml_file,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        yaml_content
    )


# ============================================================
# SUMMARY
# ============================================================

train_label_count = len(
    list(
        (
            YOLO_DATASET
            / "labels"
            / "train"
        ).glob("*.txt")
    )
)

val_label_count = len(
    list(
        (
            YOLO_DATASET
            / "labels"
            / "val"
        ).glob("*.txt")
    )
)


print("\n" + "=" * 70)
print("YOLO HARD-CASE DATASET READY")
print("=" * 70)

print(
    f"\nTraining images : "
    f"{train_copied}"
)

print(
    f"Training labels : "
    f"{train_label_count}"
)

print(
    f"Validation images : "
    f"{val_copied}"
)

print(
    f"Validation labels : "
    f"{val_label_count}"
)

print(
    f"\nDataset:"
    f"\n{YOLO_DATASET}"
)

print(
    f"\nYAML:"
    f"\n{yaml_file}"
)

print(
    "\nClasses:"
)

print(
    "  0 = cat"
)

print(
    "  1 = dog"
)

print(
    "\nThe original 84-image hard test set "
    "was NOT copied into this training dataset."
)

print(
    "It remains untouched for final evaluation."
)

print(
    "\nNo existing clean classifier dataset was modified."
)