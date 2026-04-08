import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BACKBONES = {
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple model backbones")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="comparison")
    return parser.parse_args()


def build_generators(dataset_dir, img_size, batch_size):
    train_gen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=15, zoom_range=0.15)
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        Path(dataset_dir) / "train",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
    )
    val_data = val_gen.flow_from_directory(
        Path(dataset_dir) / "val",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )
    return train_data, val_data


def build_model(backbone_cls, num_classes, img_size):
    base = backbone_cls(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.25),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = build_generators(args.dataset_dir, args.img_size, args.batch_size)

    results = {}
    for name, cls in BACKBONES.items():
        print(f"Training {name}...")
        model = build_model(cls, train_data.num_classes, args.img_size)
        history = model.fit(train_data, validation_data=val_data, epochs=args.epochs, verbose=1)

        results[name] = {
            "best_val_accuracy": float(max(history.history.get("val_accuracy", [0]))),
            "best_val_loss": float(min(history.history.get("val_loss", [1e9]))),
        }

    names = list(results.keys())
    accs = [results[n]["best_val_accuracy"] for n in names]
    losses = [results[n]["best_val_loss"] for n in names]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(names, accs, color=["#2f7d32", "#0d47a1"])
    plt.ylim(0, 1)
    plt.title("Best Validation Accuracy")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.bar(names, losses, color=["#1b5e20", "#1565c0"])
    plt.title("Best Validation Loss")
    plt.ylabel("Loss")

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()

    with open(output_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved:", output_dir / "model_comparison.png")
    print("Saved:", output_dir / "model_comparison.json")


if __name__ == "__main__":
    main()
