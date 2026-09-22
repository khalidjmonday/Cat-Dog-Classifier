from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATASET_NAME = "open-images-cat-dog-hard-candidates"

OUTPUT_DIR = Path("data/hard_cases_source")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_CLASS = 500
SEED = 42


# ------------------------------------------------------------
# Download / load Cat subset
# ------------------------------------------------------------

print("=" * 60)
print("OPEN IMAGES V7 HARD-CASE DATASET DOWNLOAD")
print("=" * 60)

print("\nLoading Cat candidates...")

cat_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Cat"],
    max_samples=SAMPLES_PER_CLASS,
    shuffle=True,
    seed=SEED,
    dataset_name=f"{DATASET_NAME}-cats",
    persistent=True,
)

print(f"Cat samples loaded: {len(cat_dataset)}")


# ------------------------------------------------------------
# Download / load Dog subset
# ------------------------------------------------------------

print("\nLoading Dog candidates...")

dog_dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Dog"],
    max_samples=SAMPLES_PER_CLASS,
    shuffle=True,
    seed=SEED,
    dataset_name=f"{DATASET_NAME}-dogs",
    persistent=True,
)

print(f"Dog samples loaded: {len(dog_dataset)}")


# ------------------------------------------------------------
# Create combined persistent dataset
# ------------------------------------------------------------

print("\nCreating combined dataset...")

if DATASET_NAME in fo.list_datasets():
    fo.delete_dataset(DATASET_NAME)

dataset = fo.Dataset(
    name=DATASET_NAME,
    persistent=True,
)

dataset.add_collection(cat_dataset)
dataset.add_collection(dog_dataset)

print(f"Combined samples: {len(dataset)}")


# ------------------------------------------------------------
# Show dataset fields
# ------------------------------------------------------------

print("\nDataset fields:")
print(dataset.get_field_schema())


# ------------------------------------------------------------
# Show basic information
# ------------------------------------------------------------

print("\nDataset information:")
print(dataset.info)


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print("\nWe now have:")
print(f"  Cats: approximately {len(cat_dataset)}")
print(f"  Dogs: approximately {len(dog_dataset)}")
print(f"  Total: {len(dataset)}")

print("\nThe datasets are now persistent.")

print("\nNext step:")
print("We will inspect the detection annotations and identify:")
print("  - occluded animals")
print("  - truncated animals")
print("  - very small animals")
print("  - cluttered scenes")
print("  - animals partially hidden by objects")

print("\nNo existing project dataset has been modified.")