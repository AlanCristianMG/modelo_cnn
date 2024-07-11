

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# Cargar el modelo previamente entrenado
model = tf.keras.models.load_model('main_model')

# Ruta del directorio que contiene las imágenes que deseas clasificar
image_dir = 'test/test_images/'  # Reemplaza con el directorio de tus imágenes

# Obtener la lista de todas las imágenes en el directorio
image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir) if img_name.endswith(('.png', '.jpg', '.jpeg'))]

# Crear una lista para almacenar los resultados
results = []

# Iterar sobre cada imagen y realizar la predicción
for image_path in image_paths:
    # Cargar la imagen usando PIL y redimensionarla a 150x150 píxeles
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión batch (batch_size = 1)

    # Normalizar la imagen (igual que en el entrenamiento)
    img_array /= 255.

    # Realizar la predicción
    predictions = model.predict(img_array)

    # Las predicciones son probabilidades para cada clase
    predicted_class = np.argmax(predictions[0])  # Obtiene el índice de la clase con mayor probabilidad

    # Almacenar el resultado en la lista
    results.append((image_path, predicted_class))

clases = ["cat", "dog", "horse"]
# Mostrar los resultados
for image_path, predicted_class in results:
    img = image.load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicción: Clase {clases[predicted_class]}')
    plt.show()
    print(f'Imagen: {image_path} -> Predicción: Clase {predicted_class} -> {clases[predicted_class]}')
