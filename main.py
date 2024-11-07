import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model

def train_model():
    # Preprocesamiento de imágenes
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_data = datagen.flow_from_directory(
        'dataset/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    model = create_model()
    model.fit(train_data, validation_data=validation_data, epochs=10)
    model.save('model/vehicle_classifier.h5')

if __name__ == '__main__':
    train_model()




# main.py (continuación)
from utils import classify_image
import os

def load_and_classify_images():
    model = load_model('model/vehicle_classifier.h5')

    img_folder = 'images/'
    for img_file in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_file)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            result = classify_image(img_path, model)
            print(f"Image: {img_file} - Predicted: {result}")

if __name__ == '__main__':
    # Descomentar para entrenar el modelo
    # train_model()
    
    # Clasificar imágenes
    load_and_classify_images()

