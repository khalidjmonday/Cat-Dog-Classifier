from pathlib import Path
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

CLASSIFIER_WEIGHTS = Path("models/best_model.pth")
DETECTOR_WEIGHTS = Path("yolo11n.pt")

IMAGE_SIZE = 224
YOLO_IMAGE_SIZE = 960
YOLO_CONFIDENCE = 0.25
CROP_PADDING = 80

CLASS_NAMES = ["cat", "dog"]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD EFFICIENTNET CLASSIFIER
# ============================================================

def load_classifier():
    model = models.efficientnet_b0(weights=None)

    # This exactly matches the classifier architecture
    # used during training.
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            model.classifier[1].in_features,
            2,
        ),
    )

    checkpoint = torch.load(
        CLASSIFIER_WEIGHTS,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported classifier checkpoint format."
        )

    # Remove common prefixes if present.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        if new_key.startswith("model."):
            new_key = new_key[6:]

        cleaned_state_dict[new_key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORM
# ============================================================

classifier_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# CLASSIFY IMAGE / CROP
# ============================================================

def classify_crop(model, image):
    tensor = (
        classifier_transform(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    cat_probability = float(
        probabilities[0]
    )

    dog_probability = float(
        probabilities[1]
    )

    predicted_index = int(
        torch.argmax(probabilities)
    )

    return (
        CLASS_NAMES[predicted_index],
        cat_probability,
        dog_probability,
    )


# ============================================================
# MAIN HYBRID PIPELINE
# ============================================================

def predict(image_path):
    image_path = Path(image_path)

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not image_path.exists():
        print(
            f"Error: Image not found: {image_path}"
        )
        return 1

    if not CLASSIFIER_WEIGHTS.exists():
        print(
            "Error: Classifier weights not found: "
            f"{CLASSIFIER_WEIGHTS}"
        )
        return 1

    if not DETECTOR_WEIGHTS.exists():
        print(
            "Error: YOLO model not found: "
            f"{DETECTOR_WEIGHTS}"
        )
        return 1

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    print("=" * 60)
    print(
        "              HYBRID CAT vs DOG CLASSIFIER"
    )
    print("=" * 60)

    print(f"\nImage: {image_path}")
    print(f"Device: {DEVICE}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:
        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:
        print(
            f"Error: Could not open image: {error}"
        )
        return 1

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading YOLO11n...")

    detector = YOLO(
        str(DETECTOR_WEIGHTS)
    )

    print("Loading EfficientNet-B0...")

    try:
        classifier = load_classifier()

    except Exception as error:
        print(
            "\nError loading EfficientNet classifier:"
        )
        print(error)
        return 1

    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    print("\nRunning YOLO11n detection...")

    results = detector.predict(
        source=str(image_path),
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )

    boxes = results[0].boxes

    # --------------------------------------------------------
    # Find highest-confidence CAT/DOG detection
    # --------------------------------------------------------

    best_detection = None
    best_confidence = 0.0

    if boxes is not None and len(boxes) > 0:

        for (
            cls_tensor,
            conf_tensor,
            box_tensor,
        ) in zip(
            boxes.cls,
            boxes.conf,
            boxes.xyxy,
        ):

            class_id = int(
                cls_tensor
            )

            confidence = float(
                conf_tensor
            )

            detected_name = (
                detector.names[class_id]
                .lower()
            )

            # We only care about cats and dogs.
            if detected_name not in CLASS_NAMES:
                continue

            # Keep the highest-confidence
            # cat/dog detection.
            if confidence > best_confidence:

                best_confidence = confidence

                best_detection = (
                    detected_name,
                    box_tensor.tolist(),
                )

    # --------------------------------------------------------
    # No CAT/DOG detection
    # --------------------------------------------------------

    if best_detection is None:

        print(
            "\nYOLO11n did not detect a cat or dog."
        )

        print(
            "Falling back to full-image "
            "EfficientNet classification."
        )

        predicted, cat_prob, dog_prob = (
            classify_crop(
                classifier,
                image,
            )
        )

        print("\nFinal prediction:")
        print(
            f"  {predicted.upper()}"
        )

        print(
            f"  CAT : {cat_prob * 100:.2f}%"
        )

        print(
            f"  DOG : {dog_prob * 100:.2f}%"
        )

        print("\n" + "=" * 60)

        return 0

    # --------------------------------------------------------
    # Get detected animal and bounding box
    # --------------------------------------------------------

    detected_name, box = best_detection

    x1, y1, x2, y2 = map(
        int,
        box,
    )

    # --------------------------------------------------------
    # Add padding around detected animal
    # --------------------------------------------------------

    x1 -= CROP_PADDING
    y1 -= CROP_PADDING
    x2 += CROP_PADDING
    y2 += CROP_PADDING

    # Keep crop inside image boundaries.
    x1 = max(
        0,
        x1,
    )

    y1 = max(
        0,
        y1,
    )

    x2 = min(
        image.width,
        x2,
    )

    y2 = min(
        image.height,
        y2,
    )

    # --------------------------------------------------------
    # Crop detected animal
    # --------------------------------------------------------

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    crop_path = image_path.with_name(
        f"{image_path.stem}_hybrid_crop.jpg"
    )

    crop.save(
        crop_path
    )

    # --------------------------------------------------------
    # EfficientNet classification
    # --------------------------------------------------------

    print(
        "\nRunning EfficientNet-B0 "
        "on detected crop..."
    )

    predicted, cat_prob, dog_prob = (
        classify_crop(
            classifier,
            crop,
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\nYOLO11n detection")

    print(
        f"  Detected   : {detected_name.upper()}"
    )

    print(
        f"  Confidence : "
        f"{best_confidence * 100:.2f}%"
    )

    print(
        f"  Box        : "
        f"[{x1}, {y1}, {x2}, {y2}]"
    )

    print(
        "\nEfficientNet-B0 classification"
    )

    print(
        f"  Prediction : {predicted.upper()}"
    )

    print(
        f"  CAT        : "
        f"{cat_prob * 100:.2f}%"
    )

    print(
        f"  DOG        : "
        f"{dog_prob * 100:.2f}%"
    )

    print(
        "\nFinal hybrid prediction"
    )

    print(
        f"  >>> {predicted.upper()} <<<"
    )

    print(
        "\nSaved crop:"
    )

    print(
        f"  {crop_path}"
    )

    print("\n" + "=" * 60)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage:")
        print(
            "  python src/predict_hybrid.py "
            "<image_path>"
        )

        sys.exit(1)

    sys.exit(
        predict(
            sys.argv[1]
        )
    )