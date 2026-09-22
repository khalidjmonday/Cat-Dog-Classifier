# 🐱🐶 Cat vs Dog Image Classifier

A deep learning based image classification project that identifies whether an image contains a **cat** or a **dog** using **EfficientNet-B0** and PyTorch.

Built as a **team machine learning project** with a complete training, evaluation, prediction, dataset quality, hard-case training, and hybrid inference pipeline.

---

## 📌 Project Overview

The goal of this project is to build a reliable image classifier capable of distinguishing cats and dogs from images.

The project evolved beyond a basic image classifier and now includes a complete ML workflow:

- Dataset preparation
- Dataset inspection
- Exact duplicate detection
- Near-duplicate analysis
- Data leakage prevention
- Clean dataset creation
- EfficientNet-B0 transfer learning
- Data augmentation
- Hard-case training
- Hard-case evaluation
- YOLO11n object detection experiments
- Hybrid YOLO11n + EfficientNet-B0 inference
- Model evaluation
- Image prediction on unseen images

Rather than only training a model, this project focuses on **dataset quality, generalization, difficult examples, and practical inference behavior**.

---

## ✨ Features

- Binary image classification for Cats and Dogs
- EfficientNet-B0 transfer learning architecture
- GPU training with CUDA support
- Dataset preparation scripts
- Exact duplicate detection using SHA-256 hashing
- Near-duplicate inspection using perceptual hashing
- Automatic duplicate cleanup
- Cross-split leakage checking
- Data augmentation for improved generalization
- Dropout regularization
- Label smoothing
- Weighted sampling
- Dedicated hard-case dataset
- Increased sampling of difficult training examples
- Hard-case evaluation benchmark
- YOLO11n object detection experiments
- Custom YOLO hard-case dataset
- Hybrid YOLO11n + EfficientNet-B0 inference
- Automatic animal localization and cropping
- Full-image fallback when YOLO does not detect a cat or dog
- Confusion matrix generation
- Prediction script for custom images

---

## 🧠 Model Architecture

The primary classifier uses **EfficientNet-B0** implemented with PyTorch.

### Configuration

- Framework: PyTorch
- Architecture: EfficientNet-B0
- Image Size: 224 × 224
- Classes: Cat, Dog
- Optimizer: AdamW
- Loss Function: Cross-Entropy Loss with label smoothing
- Dropout: 0.3
- CUDA GPU acceleration supported

The training pipeline uses data augmentation techniques such as:

- Random resized cropping
- Horizontal flipping
- Rotation
- Color jitter
- Random affine transformations
- Random erasing

These augmentations are intended to improve robustness to variations in real-world images.

---

## 📂 Dataset

The project uses public cat and dog image datasets collected from Kaggle.

The original datasets contain multiple cat and dog breeds. During preparation, all cat breeds are mapped to the `cat` class and all dog breeds are mapped to the `dog` class.

After preparation, cleaning, and duplicate removal, the final clean dataset contains:

| Split | Images |
| --- | ---: |
| Training | 11,318 |
| Validation | 2,434 |
| Testing | 2,435 |
| **Total** | **16,187** |

The dataset is intentionally **not included** in this repository because of its size.

---

## 🔍 Dataset Cleaning & Leakage Prevention

One of the major parts of this project was validating the dataset before trusting the model's accuracy.

The workflow included:

1. Inspecting the original dataset structure
2. Mapping multiple breeds into the Cat/Dog classes
3. Creating clean train, validation, and test splits
4. Detecting exact duplicate images using SHA-256 hashing
5. Detecting perceptually similar images using perceptual hashing
6. Removing confirmed duplicate samples
7. Checking for cross-split image leakage
8. Retraining the model using the cleaned dataset
9. Evaluating on the held-out test set

Final verification confirmed:

- **0 exact cross-split duplicate groups**
- Validation and test sets remained separate during cleanup

This helped create a more trustworthy evaluation pipeline.

---

# 📊 Standard Model Results

The latest EfficientNet-B0 model was evaluated on **2,435 unseen test images**.

| Metric | Score |
| --- | ---: |
| Accuracy | **99.55%** |
| Precision | Not recalculated after the latest training run |
| Recall | Not recalculated after the latest training run |
| F1 Score | Not recalculated after the latest training run |

The latest training run achieved:

- **Best validation accuracy: 99.92%**
- **Test accuracy: 99.55%**

The precision, recall, and F1 values from the previous model version are intentionally not presented as metrics for the latest model.

### Previous Model Evaluation

For reference, the previous training configuration achieved:

| Metric | Score |
| --- | ---: |
| Accuracy | **99.88%** |
| Precision | **99.93%** |
| Recall | **99.86%** |
| F1 Score | **99.89%** |

Previous confusion matrix:

![Confusion Matrix](results/confusion_matrix.png)

| Actual / Predicted | Cat | Dog |
| --- | ---: | ---: |
| Cat | 1049 | 1 |
| Dog | 2 | 1383 |

These metrics correspond to the previous training configuration and are included for historical reference only.

---

# 🎯 Hard-Case Training

After evaluating the classifier on difficult images, a dedicated hard-case dataset was created.

The hard cases contain examples where the animal may be:

- Partially occluded
- Truncated
- Relatively small
- Surrounded by visual clutter
- Difficult to distinguish from the background

The hard-case training dataset contains:

| Class | Images |
| --- | ---: |
| Cat | 168 |
| Dog | 168 |
| **Total** | **336** |

A separate held-out hard-case test set contains:

| Class | Images |
| --- | ---: |
| Cat | 42 |
| Dog | 42 |
| **Total** | **84** |

The hard-case training images are combined with the clean training dataset.

Difficult examples receive additional sampling probability using:

```text
HARD_SAMPLE_MULTIPLIER = 5.0

📈 Hard-Case Evaluation

The dedicated 84-image hard-case benchmark showed:

Model	Accuracy
Baseline EfficientNet-B0	90.48%
Hard-case trained EfficientNet-B0	97.62%

Results:

Baseline:
76 / 84 correct

Hard-case trained:
82 / 84 correct

The hard-case training experiment therefore reduced errors on this held-out benchmark from:

8 errors → 2 errors


🤖 Hybrid YOLO11n + EfficientNet Inference

Difficult images can sometimes contain an animal that occupies only a small portion of the image or is surrounded by background clutter.

To address this type of case, the project includes a hybrid inference pipeline combining YOLO11n and EfficientNet-B0.

The pipeline is:

Input Image
     │
     ▼
  YOLO11n
     │
     │ Detect cat/dog
     ▼
Bounding Box
     │
     │ Add padding
     ▼
Animal Crop
     │
     ▼
EfficientNet-B0
     │
     ▼
CAT / DOG

The implementation is located at:

src/predict_hybrid.py
Pipeline behavior
YOLO11n receives the complete image.
The detector searches for cat and dog objects.
The highest-confidence cat/dog detection is selected.
Padding is added around the detected bounding box.
The animal region is automatically cropped.
The crop is passed to EfficientNet-B0.
EfficientNet produces the final Cat/Dog classification.
If YOLO11n does not detect a cat or dog, the system falls back to full-image EfficientNet classification.


▶️ Usage
Standard EfficientNet Prediction

Run the standard classifier:

python src/predict.py <image_path>

Example:

python src/predict.py test_dog4.jpeg

The script outputs:

Final prediction
Cat probability
Dog probability
Device used for inference



🤖 Hybrid YOLO + EfficientNet Prediction

Run the hybrid pipeline:

python src/predict_hybrid.py <image_path>

Example:

python src/predict_hybrid.py blur12.jpg

The pipeline automatically:

Detects the animal using YOLO11n
Finds its bounding box
Adds padding
Creates a crop
Runs EfficientNet-B0 on the crop
Produces the final Cat/Dog prediction

If no cat or dog is detected by YOLO11n, the pipeline falls back to full-image EfficientNet classification.



📁 Project Structure
Cat-Dog-Classifier/
│
├── src/
│   ├── prepare_dataset.py
│   ├── inspect_dataset.py
│   ├── create_clean_dataset.py
│   ├── check_duplicates.py
│   ├── inspect_near_duplicates.py
│   ├── remove_exact_near_duplicates.py
│   │
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── predict_hybrid.py
│   │
│   ├── download_hard_cases.py
│   ├── select_hard_cases.py
│   ├── inspect_hard_cases.py
│   ├── build_hard_case_dataset.py
│   ├── check_hard_case_overlap.py
│   ├── evaluate_hard_cases.py
│   │
│   ├── build_yolo_hard_dataset.py
│   ├── train_yolo_hard.py
│   └── evaluate_yolo_hard_cases.py
│
├── models/
│   └── class_names.json
│
├── results/
│   └── confusion_matrix.png
│
├── .gitignore
└── README.md



📌 Current Status

The project currently includes:

✅ Cleaned dataset
✅ Exact duplicate checking
✅ Near-duplicate analysis
✅ Cross-split leakage checking
✅ EfficientNet-B0 classifier
✅ Data augmentation
✅ Label smoothing
✅ Hard-case training
✅ Hard-case evaluation
✅ YOLO object detection experiments
✅ Custom YOLO hard-case dataset
✅ Hybrid YOLO11n + EfficientNet inference
✅ Automatic animal cropping
✅ Full-image fallback
✅ Standard test evaluation
Current reported results

Standard held-out test set:

2,435 images
99.55% accuracy

Hard-case test set:

84 images
97.62% accuracy

The hybrid inference pipeline has also demonstrated successful classification of a difficult image after YOLO11n localized the relevant animal region.






👨‍💻 Authors

Yashaswi and Team

B.Tech CSE (AI/ML)

A hands-on machine learning and computer vision project focused on building, evaluating, and improving an end-to-end image classification system.