import os
import shutil
import random

# ===== PATHS =====
PLANT_VILLAGE = "PlantVillage"
from pathlib import Path
import random
import re
import shutil


BASE_DIR = Path(__file__).resolve().parent
PLANT_VILLAGE = BASE_DIR / "PlantVillage" / "PlantVillage"
PLANT_DOC = BASE_DIR / "PlantDoc" / "PlantDoc-Dataset-master" / "train"
OUTPUT = BASE_DIR / "ml-model" / "dataset"

CLASS_ALIASES = {
    "Early_blight": ["Tomato_Early_blight", "Tomato Early blight leaf"],
    "Late_blight": ["Tomato_Late_blight", "Tomato leaf late blight"],
    "Leaf_Mold": ["Tomato_Leaf_Mold", "Tomato mold leaf"],
    "Healthy": ["Tomato_healthy", "Tomato leaf"],
}

MAX_IMAGES_PER_CLASS = 1500
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_source_folders(source_root: Path, alias_names):
    if not source_root.exists():
        return []

    candidates = {normalize_name(alias) for alias in alias_names}
    matches = []

    for folder in source_root.iterdir():
        if folder.is_dir() and normalize_name(folder.name) in candidates:
            matches.append(folder)

    return matches


def collect_images(folder: Path):
    files = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            files.append(path)
    return files


def create_output_dirs(class_names):
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            (OUTPUT / split / class_name).mkdir(parents=True, exist_ok=True)


def clear_existing_output():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)


def split_name(index: int, total: int):
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def copy_with_unique_name(src: Path, dst_dir: Path, sequence: int):
    stem = normalize_name(src.parent.name)[:20]
    ext = src.suffix.lower()
    unique_name = f"{stem}_{sequence:06d}{ext}"
    shutil.copy2(src, dst_dir / unique_name)


def main():
    random.seed(SEED)

    print("Using source folders:")
    print(f" - PlantVillage: {PLANT_VILLAGE}")
    print(f" - PlantDoc: {PLANT_DOC}")
    print(f" - Output: {OUTPUT}")

    if not PLANT_VILLAGE.exists() and not PLANT_DOC.exists():
        raise FileNotFoundError("Neither PlantVillage nor PlantDoc source paths were found.")

    clear_existing_output()
    create_output_dirs(CLASS_ALIASES.keys())

    all_data = {class_name: [] for class_name in CLASS_ALIASES}

    for class_name, aliases in CLASS_ALIASES.items():
        source_folders = []
        source_folders.extend(find_source_folders(PLANT_VILLAGE, aliases))
        source_folders.extend(find_source_folders(PLANT_DOC, aliases))

        if not source_folders:
            print(f"Warning: no source folders found for class '{class_name}'")
            continue

        image_paths = []
        for folder in source_folders:
            image_paths.extend(collect_images(folder))

        random.shuffle(image_paths)
        all_data[class_name] = image_paths[:MAX_IMAGES_PER_CLASS]

        print(
            f"Class {class_name}: found {len(image_paths)} images, "
            f"using {len(all_data[class_name])}"
        )

    for class_name, files in all_data.items():
        total = len(files)
        if total == 0:
            continue

        for idx, file_path in enumerate(files):
            split = split_name(idx, total)
            copy_with_unique_name(file_path, OUTPUT / split / class_name, idx)

    print("Dataset preparation complete.")
    for split in ["train", "val", "test"]:
        for class_name in CLASS_ALIASES:
            count = len(list((OUTPUT / split / class_name).glob("*")))
            print(f"{split}/{class_name}: {count}")


if __name__ == "__main__":
    main()