from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/best_model.pth")
CLASS_NAMES_PATH = Path("models/class_names.json")

IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CHECK IMAGE PATH
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python src/predict.py <image_path>")
    sys.exit(1)

image_path = Path(sys.argv[1])

if not image_path.exists():
    print(f"\nError: Image not found:")
    print(image_path)
    sys.exit(1)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

# MUST MATCH validation/test preprocessing used during training.

transform = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model = models.efficientnet_b0(weights=None)


# Same classifier structure used in train.py

model.classifier[1] = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(
        model.classifier[1].in_features,
        2
    )
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ============================================================
# LOAD IMAGE
# ============================================================

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"\nError loading image: {e}")
    sys.exit(1)


image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )[0]

    predicted_index = torch.argmax(
        probabilities
    ).item()


predicted_class = class_names[predicted_index]

confidence = probabilities[predicted_index].item() * 100

cat_probability = probabilities[0].item() * 100
dog_probability = probabilities[1].item() * 100


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("                 CAT vs DOG CLASSIFIER")
print("=" * 60)

print(f"\nImage: {image_path}")

if predicted_class == "cat":
    print(f"\n🐱 Prediction : CAT")
else:
    print(f"\n🐶 Prediction : DOG")

print(f"   Confidence : {confidence:.2f}%")

print("\nProbabilities")
print(f"   CAT : {cat_probability:.2f}%")
print(f"   DOG : {dog_probability:.2f}%")

print(f"\nDevice : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

print("\n" + "=" * 60)