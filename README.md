# 🐱🐶 Cat vs Dog Image Classifier

A deep learning based image classification project that identifies whether an uploaded image contains a **cat** or a **dog** using **EfficientNet-B0** and PyTorch.

Built as a **team machine learning project** with a complete training, evaluation, prediction, and dataset quality pipeline.

---

## 📌 Project Overview

The goal of this project is to build a reliable image classifier capable of distinguishing cats and dogs from images.

The project includes the complete ML workflow:

* Dataset preparation
* Dataset inspection
* Duplicate detection
* Data leakage prevention
* Model training using EfficientNet-B0
* Model evaluation
* Image prediction on unseen images

Rather than only training a model, this project also focuses on validating dataset quality before evaluating performance.

---

## ✨ Features

* Binary image classification for Cats and Dogs
* EfficientNet-B0 transfer learning architecture
* GPU training with CUDA support
* Dataset preparation scripts
* Exact duplicate detection across dataset splits
* Near-duplicate inspection utilities
* Automatic duplicate cleanup
* Model evaluation with multiple metrics
* Confusion matrix generation
* Prediction script for custom images

---

## 🧠 Model Architecture

The classifier uses **EfficientNet-B0** implemented with PyTorch.

**Configuration**

* Framework: PyTorch
* Architecture: EfficientNet-B0
* Image Size: 224 × 224
* Classes: Cat, Dog
* Optimizer and training pipeline implemented in `train.py`
* CUDA GPU acceleration supported

---

## 📂 Dataset

The dataset contains images of cats and dogs collected from public Kaggle datasets.

After preparation and cleaning, the final dataset contained:

| Split      |     Images |
| ---------- | ---------: |
| Training   |     11,318 |
| Validation |      2,434 |
| Testing    |      2,435 |
| **Total**  | **16,187** |

The dataset is intentionally **not included** in this repository because of its size.

---

## 🔍 Dataset Cleaning & Leakage Prevention

One of the major parts of this project was validating the dataset before trusting the model's accuracy.

The workflow included:

1. Inspecting dataset structure
2. Creating clean train, validation and test splits
3. Detecting exact duplicate images using SHA-256 hashing
4. Detecting perceptually similar images using perceptual hashing
5. Removing confirmed duplicate training samples
6. Re-training the model on the cleaned dataset
7. Performing final evaluation on the held-out test set

Final verification confirmed:

* **0 exact cross-split duplicate groups**
* Validation and test sets remained untouched during cleanup

This helped produce a more trustworthy evaluation pipeline.

---

## 📊 Final Results

The final model was evaluated on **2,435 unseen test images**.

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **99.88%** |
| Precision | **99.93%** |
| Recall    | **99.86%** |
| F1 Score  | **99.89%** |

### Confusion Matrix

![Confusion Matrix](results/confusion_matrix.png)

Confusion Matrix:

| Actual / Predicted |  Cat |  Dog |
| ------------------ | ---: | ---: |
| Cat                | 1049 |    1 |
| Dog                |    2 | 1383 |

Only **3 images** were misclassified out of **2,435** test samples.

---

## 📁 Project Structure

```text
Cat-Dog-Classifier/
│
├── src/
│   ├── prepare_dataset.py
│   ├── inspect_dataset.py
│   ├── create_clean_dataset.py
│   ├── check_duplicates.py
│   ├── inspect_near_duplicates.py
│   ├── remove_exact_near_duplicates.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── class_names.json
│
├── results/
│   └── confusion_matrix.png
│
├── .gitignore
└── README.md
```

> **Note:** Trained model weights (`best_model.pth`) and datasets are excluded from GitHub because of their large size.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/khalidjmonday/Cat-Dog-Classifier
cd Cat-Dog-Classifier
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required Python libraries:

```bash
pip install torch torchvision pillow scikit-learn matplotlib ImageHash
```

---

## 🚀 Training

To train the model:

```bash
python src/train.py
```

The best model checkpoint will be saved inside:

```text
models/best_model.pth
```

---

## 📈 Evaluation

Evaluate the trained model on the clean test dataset:

```bash
python src/evaluate.py
```

This generates:

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix
* Evaluation metrics JSON

---

## 🔮 Predict Your Own Image

Use any cat or dog image.

Example:

```bash
python src/predict.py test_cat.jpg
```

Example output:

```text
============================================================
                 CAT vs DOG CLASSIFIER
============================================================

Image: test_cat.jpg

🐱 Prediction : CAT
Confidence : 100.00%

Probabilities

CAT : 100.00%
DOG : 0.00%

============================================================
```

Another example:

```bash
python src/predict.py test_dog.jpg
```

The model correctly predicts the uploaded image along with confidence probabilities.

---

## 🛠 Technologies Used

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Pillow
* Scikit-learn
* Matplotlib
* ImageHash
* CUDA (GPU Training)

---

## 👥 Team Contributors

This project was developed collaboratively by:

* **Yashaswi**
* **Ankush**
* **Mayank**
* **Kunal**

---

## 🔮 Future Improvements

* Cat breed classification
* Dog breed classification
* Web interface for image upload
* Mobile application integration
* AI veterinary assistant for pet image analysis
* Nearby veterinary clinic integration
* Online appointment booking

---

## 📜 License

This project is created for educational and academic purposes.
