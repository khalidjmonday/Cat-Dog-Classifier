from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def count_images(folder):
    counts = {}

    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            # First folder below the dataset root
            relative = path.relative_to(folder)
            category = relative.parts[0]
            counts[category] = counts.get(category, 0) + 1

    return counts


cat_path = Path("data/raw/cats")
dog_path = Path("data/raw/dogs")


print("\n========== CAT DATASET ==========")

cat_counts = count_images(cat_path)

for breed, count in sorted(cat_counts.items()):
    print(f"{breed}: {count}")

print(f"\nTotal cat images: {sum(cat_counts.values())}")
print(f"Cat categories found: {len(cat_counts)}")


print("\n========== DOG DATASET ==========")

dog_counts = count_images(dog_path)

for category, count in sorted(dog_counts.items()):
    print(f"{category}: {count}")

print(f"\nTotal dog images: {sum(dog_counts.values())}")
print(f"Dog categories found: {len(dog_counts)}")