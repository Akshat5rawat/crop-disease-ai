import argparse
import json
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_target(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model, layer.name

    # If model is Sequential with nested backbone, inspect sublayers.
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return layer, sublayer.name

    raise ValueError("No Conv2D layer found for Grad-CAM")


def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    resized = image.resize((img_size, img_size))
    arr = np.array(resized, dtype=np.float32) / 255.0
    return image, np.expand_dims(arr, axis=0)


def make_gradcam_heatmap(img_tensor, model, feature_model, last_conv_layer_name, pred_index=None):
    conv_layer = feature_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    denominator = tf.math.reduce_max(heatmap)
    heatmap = tf.where(denominator > 0, heatmap / denominator, heatmap)
    return heatmap.numpy()


def save_overlay(original_image, heatmap, out_path, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]

    jet_heatmap_img = Image.fromarray(np.uint8(jet_heatmap * 255))
    jet_heatmap_img = jet_heatmap_img.resize(original_image.size)

    blended = Image.blend(original_image, jet_heatmap_img, alpha=alpha)
    blended.save(out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualization")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--labels", type=str, default="labels.json")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output", type=str, default="gradcam_output.jpg")
    return parser.parse_args()


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    image, tensor = preprocess_image(args.image_path, args.img_size)

    feature_model, last_conv_layer_name = find_last_conv_target(model)
    heatmap = make_gradcam_heatmap(tensor, model, feature_model, last_conv_layer_name)

    out_path = Path(args.output)
    save_overlay(image, heatmap, out_path)

    pred = model.predict(tensor, verbose=0)[0]
    class_index = int(np.argmax(pred))
    confidence = float(pred[class_index])

    label = str(class_index)
    labels_path = Path(args.labels)
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
            label = labels.get(str(class_index), label)

    print(f"Predicted: {label} ({confidence * 100:.2f}%)")
    print(f"Grad-CAM saved to: {out_path}")


if __name__ == "__main__":
    main()
