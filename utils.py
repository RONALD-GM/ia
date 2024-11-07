# utils.py
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > prediction[0][1]:
        return "Car"
    else:
        return "Motorcycle"
