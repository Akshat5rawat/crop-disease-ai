import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train crop disease classifier")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path containing train/ and val/")
    parser.add_argument("--output-dir", type=str, default=".", help="Directory to save model and labels")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        choices=["mobilenetv2", "efficientnetb0"],
        help="Feature extractor backbone",
    )
    return parser.parse_args()


def build_backbone(name, img_size):
    if name == "efficientnetb0":
        return EfficientNetB0(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))
    return MobileNetV2(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))


def build_model(backbone_name, num_classes, img_size, learning_rate):
    base_model = build_backbone(backbone_name, img_size)
    base_model.trainable = False

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_generators(dataset_dir, img_size, batch_size):
    train_dir = Path(dataset_dir) / "train"
    val_dir = Path(dataset_dir) / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("dataset/train and dataset/val must exist")

    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    return train_data, val_data


def save_training_plots(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history.get("accuracy", []), label="train")
    plt.plot(history.history.get("val_accuracy", []), label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history.get("loss", []), label="train")
    plt.plot(history.history.get("val_loss", []), label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data = create_generators(args.dataset_dir, args.img_size, args.batch_size)

    model = build_model(args.backbone, train_data.num_classes, args.img_size, args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
        ModelCheckpoint(output_dir / "best_model.h5", monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = model.evaluate(val_data, verbose=0)
    final_report = {"val_loss": float(metrics[0]), "val_accuracy": float(metrics[1])}

    model_path = output_dir / "model.h5"
    model.save(model_path)

    index_to_label = {str(index): label for label, index in train_data.class_indices.items()}
    with open(output_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    save_training_plots(history, output_dir)

    print(f"Saved model to: {model_path}")
    print(f"Validation accuracy: {final_report['val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
