from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
from scipy.fftpack import dct


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/clean")
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


# ============================================================
# GET IMAGES
# ============================================================

def get_images(split):

    folder = DATA_DIR / split

    return [
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# EXACT HASH
# ============================================================

def exact_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as file:

        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def perceptual_hash(path):

    try:

        image = Image.open(path).convert("L")

        image = image.resize((32, 32))

        pixels = np.asarray(image, dtype=np.float32)

        # 2D DCT
        dct_result = dct(
            dct(pixels, axis=0, norm="ortho"),
            axis=1,
            norm="ortho"
        )

        # Keep low-frequency information
        low_freq = dct_result[:8, :8]

        median = np.median(low_freq[1:])

        bits = low_freq > median

        return "".join(
            "1" if bit else "0"
            for bit in bits.flatten()
        )

    except Exception:

        return None


# ============================================================
# HAMMING DISTANCE
# ============================================================

def hamming_distance(hash1, hash2):

    return sum(
        bit1 != bit2
        for bit1, bit2 in zip(hash1, hash2)
    )


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("DATASET DUPLICATE / LEAKAGE CHECK")
print("=" * 70)


# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

images = {}

for split in SPLITS:

    images[split] = get_images(split)

    print(
        f"\n{split.upper():5}: "
        f"{len(images[split])} images"
    )


# ------------------------------------------------------------
# EXACT DUPLICATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING EXACT DUPLICATES")
print("=" * 70)

hashes = {}

for split in SPLITS:

    print(f"\nHashing {split} images...")

    for path in images[split]:

        file_hash = exact_hash(path)

        if file_hash not in hashes:

            hashes[file_hash] = []

        hashes[file_hash].append(
            (split, path)
        )


exact_cross_split = []

for file_hash, locations in hashes.items():

    split_names = {
        split
        for split, path in locations
    }

    # Only interested in duplicates across different splits
    if len(split_names) > 1:

        exact_cross_split.append(
            locations
        )


print(
    f"\nExact cross-split duplicate groups: "
    f"{len(exact_cross_split)}"
)


if exact_cross_split:

    print("\n⚠️ EXACT DUPLICATES FOUND:")

    for group in exact_cross_split[:20]:

        print("\nGroup:")

        for split, path in group:

            print(
                f"  {split}: {path}"
            )

else:

    print(
        "\n✅ No exact duplicates found "
        "across train/validation/test."
    )


# ------------------------------------------------------------
# PERCEPTUAL HASHES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING NEAR-DUPLICATES")
print("=" * 70)

print(
    "\nGenerating perceptual hashes..."
)

phashes = {}

for split in SPLITS:

    phashes[split] = []

    print(f"Processing {split}...")

    for path in images[split]:

        image_hash = perceptual_hash(path)

        if image_hash is not None:

            phashes[split].append(
                (path, image_hash)
            )


# ------------------------------------------------------------
# COMPARE SPLITS
# ------------------------------------------------------------

# Hamming distance <= 5 means the images are extremely similar.
THRESHOLD = 5

near_duplicates = []


for split_a, split_b in [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]:

    print(
        f"\nComparing {split_a} ↔ {split_b}..."
    )

    for path_a, hash_a in phashes[split_a]:

        for path_b, hash_b in phashes[split_b]:

            distance = hamming_distance(
                hash_a,
                hash_b
            )

            if distance <= THRESHOLD:

                near_duplicates.append({
                    "split_a": split_a,
                    "path_a": str(path_a),
                    "split_b": split_b,
                    "path_b": str(path_b),
                    "distance": distance
                })


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NEAR-DUPLICATE RESULTS")
print("=" * 70)

print(
    f"\nNear-duplicate pairs found: "
    f"{len(near_duplicates)}"
)


if near_duplicates:

    print(
        "\n⚠️ Potentially similar images found."
    )

    print(
        "\nShowing first 20 candidates:"
    )

    for item in near_duplicates[:20]:

        print(
            f"\nDistance: {item['distance']}"
        )

        print(
            f"  {item['split_a']}: "
            f"{item['path_a']}"
        )

        print(
            f"  {item['split_b']}: "
            f"{item['path_b']}"
        )

else:

    print(
        "\n✅ No suspicious near-duplicates "
        "found across dataset splits."
    )


# ============================================================
# FINAL VERDICT
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET CHECK")
print("=" * 70)

if exact_cross_split:

    print(
        "\n❌ EXACT DATA LEAKAGE DETECTED"
    )

    print(
        "Some identical images appear in "
        "multiple dataset splits."
    )

elif near_duplicates:

    print(
        "\n⚠️ POSSIBLE NEAR-DUPLICATE LEAKAGE"
    )

    print(
        "Some very similar images were found."
    )

    print(
        "These should be manually inspected."
    )

else:

    print(
        "\n✅ NO OBVIOUS DATA LEAKAGE DETECTED"
    )

    print(
        "The train/validation/test split "
        "appears clean."
    )

print("\nCheck complete.")