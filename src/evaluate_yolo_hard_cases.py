from pathlib import Path
import json

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/hard_cases/test")

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.05
IMAGE_SIZE = 960

RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "yolo_hard_case_evaluation.json"


# ============================================================
# SETUP
# ============================================================

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Hard test dataset not found:\n{DATA_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO11 HARD-CASE EVALUATION")
print("=" * 70)

print(
    f"\nModel: {MODEL_PATH}"
)

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD}"
)

print(
    f"Image size: "
    f"{IMAGE_SIZE}"
)

print("\nLoading YOLO11n...")

model = YOLO(
    MODEL_PATH
)


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

samples = []

for class_name in ["cat", "dog"]:

    class_dir = DATA_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class directory:\n{class_dir}"
        )

    for image_path in class_dir.iterdir():

        if (
            image_path.is_file()
            and image_path.suffix.lower()
            in image_extensions
        ):

            samples.append(
                {
                    "path": image_path,
                    "actual": class_name,
                }
            )


if not samples:
    raise RuntimeError(
        "No hard-test images were found."
    )


# Keep the result order deterministic
samples.sort(
    key=lambda item: str(item["path"]).lower()
)


print(
    f"\nHard-test images found: "
    f"{len(samples)}"
)


# ============================================================
# CLASS IDS
# ============================================================

# COCO / YOLO11 class mapping:
#
# 15 = cat
# 16 = dog

CAT_CLASS_ID = 15
DOG_CLASS_ID = 16


# ============================================================
# EVALUATION
# ============================================================

correct = 0
missed = 0
wrong_class = 0

cat_correct = 0
dog_correct = 0

cat_total = 0
dog_total = 0

details = []


print("\nRunning detection...")
print()


for index, sample in enumerate(samples, start=1):

    image_path = sample["path"]
    actual = sample["actual"]

    if actual == "cat":
        cat_total += 1
    else:
        dog_total += 1


    print(
        f"[{index:02d}/{len(samples)}] "
        f"{image_path.name} "
        f"(actual={actual})"
    )


    # --------------------------------------------------------
    # Run prediction
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        device=0,
        verbose=False,
        save=False,
    )

    result = results[0]


    # --------------------------------------------------------
    # Collect Cat / Dog detections
    # --------------------------------------------------------

    detected_cat = []
    detected_dog = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            if class_id == CAT_CLASS_ID:

                detected_cat.append(
                    confidence
                )

            elif class_id == DOG_CLASS_ID:

                detected_dog.append(
                    confidence
                )


    best_cat = (
        max(detected_cat)
        if detected_cat
        else 0.0
    )

    best_dog = (
        max(detected_dog)
        if detected_dog
        else 0.0
    )


    # --------------------------------------------------------
    # Determine detected class
    # --------------------------------------------------------

    if best_cat == 0.0 and best_dog == 0.0:

        detected_class = "none"

        missed += 1

        is_correct = False

        result_type = "miss"

    else:

        if best_dog >= best_cat:

            detected_class = "dog"
            detected_confidence = best_dog

        else:

            detected_class = "cat"
            detected_confidence = best_cat


        is_correct = (
            detected_class == actual
        )


        if is_correct:

            correct += 1

            result_type = "correct"

            if actual == "cat":
                cat_correct += 1
            else:
                dog_correct += 1

        else:

            wrong_class += 1

            result_type = "wrong"


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    if detected_class == "none":

        print(
            "    ❌ NO CAT/DOG DETECTION"
        )

    else:

        print(
            f"    "
            f"{'✅' if is_correct else '❌'} "
            f"detected={detected_class.upper()} "
            f"confidence={detected_confidence * 100:.2f}%"
        )

        print(
            f"    CAT={best_cat * 100:.2f}% "
            f"DOG={best_dog * 100:.2f}%"
        )


    # --------------------------------------------------------
    # Save details
    # --------------------------------------------------------

    details.append(
        {
            "image": str(image_path),
            "actual": actual,
            "detected": detected_class,
            "cat_confidence": round(
                best_cat,
                6,
            ),
            "dog_confidence": round(
                best_dog,
                6,
            ),
            "result": result_type,
        }
    )


# ============================================================
# METRICS
# ============================================================

total = len(samples)

accuracy = (
    correct / total
    if total
    else 0.0
)

cat_accuracy = (
    cat_correct / cat_total
    if cat_total
    else 0.0
)

dog_accuracy = (
    dog_correct / dog_total
    if dog_total
    else 0.0
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("YOLO11 HARD-CASE RESULTS")
print("=" * 70)

print(
    f"\nTotal images       : {total}"
)

print(
    f"Correct            : {correct}"
)

print(
    f"Missed             : {missed}"
)

print(
    f"Wrong class        : {wrong_class}"
)

print(
    f"\nOverall accuracy   : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Cat accuracy       : "
    f"{cat_accuracy * 100:.2f}% "
    f"({cat_correct}/{cat_total})"
)

print(
    f"Dog accuracy       : "
    f"{dog_accuracy * 100:.2f}% "
    f"({dog_correct}/{dog_total})"
)


# ============================================================
# LIST FAILURES
# ============================================================

failures = [
    item
    for item in details
    if item["result"] != "correct"
]


if failures:

    print(
        "\n" + "-" * 70
    )

    print(
        "FAILURES"
    )

    print(
        "-" * 70
    )

    for item in failures:

        print(
            f"\nImage: {item['image']}"
        )

        print(
            f"Actual: {item['actual']}"
        )

        print(
            f"Detected: {item['detected']}"
        )

        print(
            f"CAT: "
            f"{item['cat_confidence'] * 100:.2f}%"
        )

        print(
            f"DOG: "
            f"{item['dog_confidence'] * 100:.2f}%"
        )

else:

    print(
        "\n🎯 YOLO11 detected the correct class "
        "in every hard-test image."
    )


# ============================================================
# SAVE JSON
# ============================================================

evaluation = {
    "model": MODEL_PATH,
    "dataset": str(DATA_DIR),
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "image_size": IMAGE_SIZE,
    "total_images": total,
    "correct": correct,
    "missed": missed,
    "wrong_class": wrong_class,
    "accuracy": accuracy,
    "cat_total": cat_total,
    "cat_correct": cat_correct,
    "cat_accuracy": cat_accuracy,
    "dog_total": dog_total,
    "dog_correct": dog_correct,
    "dog_accuracy": dog_accuracy,
    "failures": failures,
    "details": details,
}


with open(
    RESULTS_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        evaluation,
        f,
        indent=4,
    )


print(
    f"\nResults saved to:\n{RESULTS_FILE}"
)

print(
    "\n" + "=" * 70
)

print(
    "YOLO11 HARD-CASE EVALUATION COMPLETE"
)

print(
    "=" * 70
)