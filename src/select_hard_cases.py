from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

ANNOTATIONS_FILE = Path(
    r"C:\Users\yasha\fiftyone\open-images-v7\validation\labels\detections.csv"
)

IMAGE_DIR = Path(
    r"C:\Users\yasha\fiftyone\open-images-v7\validation\data"
)

OUTPUT_FILE = Path("results/open_images_hard_cases.csv")

# Open Images class IDs
CAT_LABEL = "/m/01yrx"
DOG_LABEL = "/m/0bt9lr"

# Small-object thresholds
SMALL_AREA = 0.10
VERY_SMALL_AREA = 0.05

# Process CSV in chunks to avoid unnecessary memory usage
CHUNK_SIZE = 100_000


# ------------------------------------------------------------
# Find downloaded Open Images files
# ------------------------------------------------------------

print("=" * 70)
print("OPEN IMAGES HARD-CASE SELECTOR")
print("=" * 70)

print("\nScanning downloaded images...")

downloaded_ids = set()

for extension in ("*.jpg", "*.jpeg", "*.png"):
    for image_path in IMAGE_DIR.rglob(extension):
        downloaded_ids.add(image_path.stem)

print(f"Downloaded image files found: {len(downloaded_ids)}")

if not downloaded_ids:
    raise RuntimeError(
        f"No downloaded images found in:\n{IMAGE_DIR}"
    )


# ------------------------------------------------------------
# Check annotation CSV
# ------------------------------------------------------------

if not ANNOTATIONS_FILE.exists():
    raise FileNotFoundError(
        f"Annotation file not found:\n{ANNOTATIONS_FILE}"
    )

print("\nReading Open Images annotations...")
print(f"CSV: {ANNOTATIONS_FILE}")


hard_rows = []

total_matching_rows = 0
occluded_rows = 0
truncated_rows = 0
small_rows = 0
very_small_rows = 0


# ------------------------------------------------------------
# Process annotations
# ------------------------------------------------------------

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

for chunk_number, chunk in enumerate(
    pd.read_csv(
        ANNOTATIONS_FILE,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
    ),
    start=1,
):
    print(f"Processing annotation chunk {chunk_number}...")

    # Keep only Cat and Dog
    chunk = chunk[
        chunk["LabelName"].isin(
            [CAT_LABEL, DOG_LABEL]
        )
    ]

    # Keep only images we actually downloaded
    chunk = chunk[
        chunk["ImageID"].isin(downloaded_ids)
    ]

    if chunk.empty:
        continue

    total_matching_rows += len(chunk)

    # Calculate bounding-box area
    chunk["box_width"] = (
        chunk["XMax"] - chunk["XMin"]
    )

    chunk["box_height"] = (
        chunk["YMax"] - chunk["YMin"]
    )

    chunk["box_area_fraction"] = (
        chunk["box_width"] * chunk["box_height"]
    )

    # Convert Open Images flags to numeric safely
    chunk["IsOccluded"] = (
        pd.to_numeric(
            chunk["IsOccluded"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    chunk["IsTruncated"] = (
        pd.to_numeric(
            chunk["IsTruncated"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    occluded_rows += int(
        (chunk["IsOccluded"] == 1).sum()
    )

    truncated_rows += int(
        (chunk["IsTruncated"] == 1).sum()
    )

    small_rows += int(
        (chunk["box_area_fraction"] <= SMALL_AREA).sum()
    )

    very_small_rows += int(
        (chunk["box_area_fraction"] <= VERY_SMALL_AREA).sum()
    )

    # Hard-case definition
    hard_mask = (
        (chunk["IsOccluded"] == 1)
        | (chunk["IsTruncated"] == 1)
        | (chunk["box_area_fraction"] <= SMALL_AREA)
    )

    hard_chunk = chunk[hard_mask].copy()

    if not hard_chunk.empty:
        hard_rows.append(hard_chunk)


# ------------------------------------------------------------
# Combine results
# ------------------------------------------------------------

if hard_rows:
    hard_cases = pd.concat(
        hard_rows,
        ignore_index=True
    )
else:
    hard_cases = pd.DataFrame(
        columns=usecols + [
            "box_width",
            "box_height",
            "box_area_fraction",
        ]
    )


# ------------------------------------------------------------
# Convert class IDs to readable labels
# ------------------------------------------------------------

if not hard_cases.empty:
    hard_cases["class"] = hard_cases["LabelName"].map(
        {
            CAT_LABEL: "cat",
            DOG_LABEL: "dog",
        }
    )

    # Put useful columns first
    hard_cases = hard_cases[
        [
            "ImageID",
            "class",
            "box_area_fraction",
            "IsOccluded",
            "IsTruncated",
            "XMin",
            "XMax",
            "YMin",
            "YMax",
            "IsGroupOf",
            "IsDepiction",
            "IsInside",
        ]
    ]

    # Remove duplicate annotations for the same class/image
    hard_cases = hard_cases.drop_duplicates(
        subset=["ImageID", "class"]
    )

    # Smallest objects first
    hard_cases = hard_cases.sort_values(
        by=["box_area_fraction", "class"],
        ascending=[True, True],
    )


# ------------------------------------------------------------
# Save report
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

hard_cases.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HARD-CASE ANALYSIS")
print("=" * 70)

print(f"\nDownloaded images checked : {len(downloaded_ids)}")
print(f"Cat/Dog annotation rows   : {total_matching_rows}")

print(f"\nOccluded annotations      : {occluded_rows}")
print(f"Truncated annotations     : {truncated_rows}")
print(f"Small objects <= 10%     : {small_rows}")
print(f"Very small objects <= 5% : {very_small_rows}")

print(
    f"\nUnique hard image/class pairs : {len(hard_cases)}"
)

if not hard_cases.empty:
    print("\nHard cases by class:")

    print(
        hard_cases["class"]
        .value_counts()
        .to_string()
    )

    print("\nHard cases by type:")

    print(
        f"  Occluded : "
        f"{(hard_cases['IsOccluded'] == 1).sum()}"
    )

    print(
        f"  Truncated: "
        f"{(hard_cases['IsTruncated'] == 1).sum()}"
    )

    print(
        f"  Small    : "
        f"{(hard_cases['box_area_fraction'] <= SMALL_AREA).sum()}"
    )

    print(
        f"  Very small: "
        f"{(hard_cases['box_area_fraction'] <= VERY_SMALL_AREA).sum()}"
    )

    print("\nSmallest examples:")

    print(
        hard_cases[
            [
                "ImageID",
                "class",
                "box_area_fraction",
                "IsOccluded",
                "IsTruncated",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

else:
    print("\nNo hard cases were found.")


print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nNo existing training data was modified.")
print("No model was modified.")