import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-style Grad-CAM figure")
    parser.add_argument("image_path", type=str, help="Path to the leaf image")
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--labels", type=str, default="labels.json")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="gradcam_academic.png")
    return parser.parse_args()


def find_last_conv_name(backbone):
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model backbone")


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    model_path = Path(args.model)
    labels_path = Path(args.labels)
    output_path = Path(args.output)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    labels = {}
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

    model = tf.keras.models.load_model(model_path)

    # Expected layout from this project train.py:
    # Sequential([backbone, GlobalAveragePooling2D, Dropout, Dense, Dense])
    if not model.layers or not isinstance(model.layers[0], tf.keras.Model):
        raise ValueError("Expected first model layer to be a CNN backbone model")

    backbone = model.layers[0]
    classifier_layers = model.layers[1:]
    last_conv_name = find_last_conv_name(backbone)

    original = Image.open(image_path).convert("RGB")
    resized = original.resize((args.img_size, args.img_size))
    image_arr = np.array(resized, dtype=np.float32) / 255.0
    image_tensor = np.expand_dims(image_arr, axis=0)

    conv_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_name).output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_output, x = conv_model(image_tensor)
        for layer in classifier_layers:
            x = layer(x)
        predictions = x
        pred_index = tf.argmax(predictions[0])
        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = tf.where(max_val > 0, heatmap / max_val, heatmap)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(original.size, resample=Image.BILINEAR)
    heatmap_resized = np.array(heatmap_img, dtype=np.float32) / 255.0

    original_np = np.array(original, dtype=np.float32) / 255.0
    heatmap_color = plt.get_cmap("jet")(heatmap_resized)[..., :3]
    overlay = np.clip((1 - args.alpha) * original_np + args.alpha * heatmap_color, 0, 1)

    pred_vector = predictions.numpy()[0]
    class_idx = int(np.argmax(pred_vector))
    confidence = float(pred_vector[class_idx])
    pred_label = labels.get(str(class_idx), str(class_idx))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[0, 1])
    ax3 = fig.add_subplot(grid[0, 2])
    cax = fig.add_subplot(grid[0, 3])

    ax1.imshow(original_np)
    ax1.set_title("(a) Original Leaf")
    ax1.axis("off")

    img = ax2.imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    ax2.set_title("(b) Grad-CAM Heatmap")
    ax2.axis("off")

    ax3.imshow(overlay)
    ax3.set_title("(c) Overlay")
    ax3.axis("off")

    colorbar = fig.colorbar(img, cax=cax)
    colorbar.set_label("Activation Intensity")

    fig.suptitle(
        f"Grad-CAM Localization | Predicted: {pred_label} ({confidence * 100:.2f}%)",
        fontsize=13,
        y=1.02,
    )

    fig.savefig(output_path, dpi=360, bbox_inches="tight")
    plt.close(fig)

    print(f"Predicted class: {pred_label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Last conv layer: {last_conv_name}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()