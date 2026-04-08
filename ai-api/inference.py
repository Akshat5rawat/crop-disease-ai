import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


class CropDiseaseService:
    def __init__(self, model_path, labels_path, img_size=224):
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.img_size = img_size

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.labels = json.load(f)

    def _preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB").resize((self.img_size, self.img_size))
        arr = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def predict_image(self, image_path, top_k=3):
        tensor = self._preprocess(image_path)
        probs = self.model.predict(tensor, verbose=0)[0]

        top_indices = probs.argsort()[-top_k:][::-1]
        top_predictions = [
            {
                "label": self.labels.get(str(int(i)), str(int(i))),
                "confidence": float(probs[i]),
            }
            for i in top_indices
        ]

        best = top_predictions[0]
        return {
            "disease": best["label"],
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
        }
