from pathlib import Path
import hashlib


# ============================================================
# Configuration
# ============================================================

CLEAN_DIR = Path("data/clean")
HARD_TRAIN_DIR = Path("data/hard_cases/train")

CLEAN_SPLITS = ("train", "val", "test")


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    """Calculate SHA-256 hash of an image file."""
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def collect_hashes(directory: Path):
    """Return {hash: path} for all image files in a directory."""
    image_hashes = {}

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            file_hash = sha256_file(path)
            image_hashes[file_hash] = path

    return image_hashes


# ============================================================
# Start
# ============================================================

print("=" * 70)
print("HARD-CASE DATASET OVERLAP CHECK")
print("=" * 70)


# ============================================================
# Validate directories
# ============================================================

if not CLEAN_DIR.exists():
    raise FileNotFoundError(
        f"Clean dataset not found:\n{CLEAN_DIR}"
    )

if not HARD_TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Hard training dataset not found:\n{HARD_TRAIN_DIR}"
    )


# ============================================================
# Collect hashes from clean dataset
# ============================================================

print("\nScanning clean dataset...")

clean_hashes = {}

for split in CLEAN_SPLITS:

    split_dir = CLEAN_DIR / split

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Missing clean split:\n{split_dir}"
        )

    print(f"  Scanning {split}...")

    split_hashes = collect_hashes(split_dir)

    for file_hash, path in split_hashes.items():
        clean_hashes[file_hash] = path


print(
    f"\nUnique clean image hashes: {len(clean_hashes)}"
)


# ============================================================
# Collect hashes from hard training set
# ============================================================

print("\nScanning hard training dataset...")

hard_hashes = collect_hashes(HARD_TRAIN_DIR)

print(
    f"Hard training image hashes: {len(hard_hashes)}"
)


# ============================================================
# Find exact overlaps
# ============================================================

overlaps = []

for file_hash, hard_path in hard_hashes.items():

    if file_hash in clean_hashes:

        overlaps.append(
            {
                "hard_path": hard_path,
                "clean_path": clean_hashes[file_hash],
                "sha256": file_hash,
            }
        )


# ============================================================
# Results
# ============================================================

print("\n" + "=" * 70)
print("OVERLAP RESULTS")
print("=" * 70)

print(
    f"\nHard training images checked : {len(hard_hashes)}"
)

print(
    f"Exact duplicates found       : {len(overlaps)}"
)


if overlaps:

    print("\nWARNING: Exact overlaps were found:")

    for item in overlaps:

        print("\nHard training:")
        print(f"  {item['hard_path']}")

        print("Clean dataset:")
        print(f"  {item['clean_path']}")

        print("SHA-256:")
        print(f"  {item['sha256']}")

    print(
        "\nThese images should NOT be added to the training set."
    )

else:

    print(
        "\n✅ No exact duplicate images were found."
    )

    print(
        "The hard training set is clean with respect to"
    )

    print(
        "the existing clean train, validation, and test sets."
    )


print("\n" + "=" * 70)
print("OVERLAP CHECK COMPLETE")
print("=" * 70)