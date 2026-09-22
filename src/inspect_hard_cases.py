from pathlib import Path

import fiftyone as fo

CAT_DATASET = "open-images-cat-dog-hard-candidates-cats"
DOG_DATASET = "open-images-cat-dog-hard-candidates-dogs"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = RESULTS_DIR / "open_images_hard_cases_report.csv"


def normalize_bool(value):
    """Convert common Open Images attribute values into True/False/None."""
    if value is True:
        return True

    if value is False:
        return False

    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False

    if isinstance(value, str):
        value = value.strip().lower()

        if value in {"true", "1", "yes"}:
            return True

        if value in {"false", "0", "no"}:
            return False

    return None


def get_attribute(detection, wanted_name):
    """Find an attribute without depending on capitalization."""
    wanted_name = wanted_name.lower()

    if not detection.attributes:
        return None

    for key, value in detection.attributes.items():
        if key.lower() == wanted_name:
            return value

    return None


def inspect_dataset(dataset, expected_class):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset.name}")
    print(f"EXPECTED CLASS: {expected_class}")
    print(f"SAMPLES: {len(dataset)}")
    print("=" * 70)

    total_detections = 0
    occluded_count = 0
    truncated_count = 0
    small_count = 0
    very_small_count = 0

    rows = []

    printed_example_attributes = False

    for sample in dataset:
        detections = sample.ground_truth

        if detections is None:
            continue

        for detection in detections.detections:
            label = detection.label

            if label.lower() not in {"cat", "dog"}:
                continue

            total_detections += 1

            x, y, width, height = detection.bounding_box

            box_area = width * height

            occluded_raw = get_attribute(detection, "IsOccluded")
            truncated_raw = get_attribute(detection, "IsTruncated")

            occluded = normalize_bool(occluded_raw)
            truncated = normalize_bool(truncated_raw)

            if occluded is True:
                occluded_count += 1

            if truncated is True:
                truncated_count += 1

            if box_area <= 0.10:
                small_count += 1

            if box_area <= 0.05:
                very_small_count += 1

            is_hard = (
                occluded is True
                or truncated is True
                or box_area <= 0.10
            )

            if is_hard:
                rows.append(
                    {
                        "filepath": sample.filepath,
                        "label": label,
                        "box_area_fraction": round(box_area, 6),
                        "occluded": occluded,
                        "truncated": truncated,
                        "x": round(x, 6),
                        "y": round(y, 6),
                        "width": round(width, 6),
                        "height": round(height, 6),
                    }
                )

            if not printed_example_attributes:
                print("\nExample detection:")
                print("Label:", label)
                print("Bounding box:", detection.bounding_box)
                print("Attributes:", dict(detection.attributes))
                print()
                printed_example_attributes = True

    print(f"Total cat/dog detections : {total_detections}")
    print(f"Occluded                 : {occluded_count}")
    print(f"Truncated                : {truncated_count}")
    print(f"Small (<= 10% area)      : {small_count}")
    print(f"Very small (<= 5% area)  : {very_small_count}")
    print(f"Hard-case detections     : {len(rows)}")

    return rows


print("=" * 70)
print("OPEN IMAGES HARD-CASE INSPECTION")
print("=" * 70)

if CAT_DATASET not in fo.list_datasets():
    raise RuntimeError(
        f"Dataset '{CAT_DATASET}' was not found in FiftyOne."
    )

if DOG_DATASET not in fo.list_datasets():
    raise RuntimeError(
        f"Dataset '{DOG_DATASET}' was not found in FiftyOne."
    )

cat_dataset = fo.load_dataset(CAT_DATASET)
dog_dataset = fo.load_dataset(DOG_DATASET)

all_rows = []

all_rows.extend(inspect_dataset(cat_dataset, "cat"))
all_rows.extend(inspect_dataset(dog_dataset, "dog"))


# ------------------------------------------------------------
# Save CSV report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING REPORT")
print("=" * 70)

with open(REPORT_FILE, "w", encoding="utf-8", newline="") as f:
    import csv

    fieldnames = [
        "filepath",
        "label",
        "box_area_fraction",
        "occluded",
        "truncated",
        "x",
        "y",
        "width",
        "height",
    ]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(all_rows)

print(f"Hard-case report saved to:")
print(REPORT_FILE)

print("\nTotal hard-case detections:", len(all_rows))

print("\nNext step:")
print("We will use this report to select the genuinely difficult")
print("images for the new hard-case training/test dataset.")
