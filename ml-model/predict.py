import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class CropDiseasePredictor:
    def __init__(self, model_path="model.h5", labels_path="labels.json"):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    @staticmethod
    def _preprocess_image(image_path, img_size=224):
        image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
        array = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(array, axis=0)

    def predict_image(self, image_path, top_k=3):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        inputs = self._preprocess_image(image_path)
        probs = self.model.predict(inputs, verbose=0)[0]

        top_indices = probs.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                "label": self.labels.get(str(int(idx)), str(int(idx))),
                "confidence": float(probs[idx]),
            }
            for idx in top_indices
        ]

        best = top_predictions[0]
        return {
            "disease": best["label"],
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop disease from one image")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--model", type=str, default="model.h5")
    parser.add_argument("--labels", type=str, default="labels.json")
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = CropDiseasePredictor(model_path=args.model, labels_path=args.labels)
    result = predictor.predict_image(args.image_path)

    print("Predicted disease:", result["disease"])
    print("Confidence:", f"{result['confidence'] * 100:.2f}%")
    for idx, item in enumerate(result["top_predictions"], start=1):
        print(f"Top {idx}: {item['label']} ({item['confidence'] * 100:.2f}%)")


if __name__ == "__main__":
    main()
