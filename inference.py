import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("model/classifier.h5")


class_names = ['crack', 'dent', 'scratch', 'corrosion']

def predict_image(image_path):
    img = Image.open(image_path).convert("RGB").resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_class = class_names[np.argmax(pred)]

    return predicted_class