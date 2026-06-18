import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

MODEL_PATH = "inceptionv3_vit_lungcancer.h5"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Benign", "Malignant", "Normal"]

print(" Loading model...")
model = load_model(MODEL_PATH)
print(" Model loaded successfully!")

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f" Error: Image path '{img_path}' does not exist.")
        return

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = preds[0][pred_idx] * 100

    print("\n Prediction Results:")
    print(f" - Class: {CLASS_NAMES[pred_idx]}")
    print(f" - Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    img_path = r"Normal cases\Normal case (8).jpg"
    predict_image(img_path)
