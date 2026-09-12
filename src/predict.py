import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path("models/best_model.pth")
IMAGE_SIZE = 224


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_names = checkpoint["class_names"]

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        print(f"\n❌ Image not found: {image_path}")
        return

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"\n❌ Could not open image: {e}")
        return

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    # Get probabilities
    cat_probability = (
        probabilities[class_names.index("cat")].item() * 100
    )

    dog_probability = (
        probabilities[class_names.index("dog")].item() * 100
    )

    # Get prediction
    predicted_index = torch.argmax(probabilities).item()
    predicted_class = class_names[predicted_index]

    confidence = (
        probabilities[predicted_index].item() * 100
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("                 CAT vs DOG CLASSIFIER")
    print("=" * 60)

    print(f"\nImage: {image_path}")

    if predicted_class == "cat":
        print("\n🐱 Prediction : CAT")
    else:
        print("\n🐶 Prediction : DOG")

    print(f"   Confidence : {confidence:.2f}%")

    print("\nProbabilities")
    print(f"   CAT : {cat_probability:.2f}%")
    print(f"   DOG : {dog_probability:.2f}%")

    print(f"\nDevice : {device}")

    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    print("\n" + "=" * 60)


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) < 2:

    print("\nUsage:")
    print("python src\\predict.py path\\to\\image.jpg")

    print("\nExample:")
    print("python src\\predict.py test_cat.jpg")

else:

    predict(sys.argv[1])
