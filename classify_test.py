import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os
from PIL import Image, UnidentifiedImageError

# Verificar la versión de TensorFlow y los dispositivos disponibles
print("Versión de TensorFlow:", tf.__version__)
print("Dispositivos disponibles:", tf.config.list_physical_devices())

# Directorio del conjunto de datos
dataset_dir = 'dataset/'

# Función para verificar y eliminar imágenes corruptas
def verify_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            try:
                img = Image.open(os.path.join(root, file))  # Verifica si se puede abrir la imagen
                img.verify()  # Verifica si es una imagen válida
            except (IOError, SyntaxError, UnidentifiedImageError) as e:
                print(f'Corrupt image file: {os.path.join(root, file)}')
                os.remove(os.path.join(root, file))  # Opcional: elimina la imagen corrupta

# Llama a la función para verificar el directorio del conjunto de datos
verify_images(dataset_dir)

# Crear generadores de datos con aumentación
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Función para manejar excepciones en el generador de datos
def custom_flow_from_directory(directory, *args, **kwargs):
    generator = train_datagen.flow_from_directory(directory, *args, **kwargs)
    return generator

train_generator = custom_flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=64,  # Puedes ajustar el tamaño del lote aquí
    class_mode='categorical',
    subset='training'
)

validation_generator = custom_flow_from_directory(
    dataset_dir,
    target_size=(150, 150),
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# Configuración de GPU (si se usa)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to avoid allocating all the memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Verificar si existe un modelo guardado
if os.path.exists('main_model'):
    # Cargar el modelo previamente guardado
    print("Cargando el mejor modelo pre-entrenado...")
    model = tf.keras.models.load_model('main_model')
else:
    # Crear un nuevo modelo si no hay uno guardado
    with tf.device('/GPU:0'):  # Asegúrate de que el modelo se cree en el contexto de la GPU
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(3, activation='softmax')
        ])
    print("Creando un nuevo modelo...")

# Compilar el modelo
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Crear directorio de logs si no existe
log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Desactivar la trazabilidad de TensorBoard para evitar errores relacionados con CUPTI
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Solo muestra errores

# Definir callbacks para el entrenamiento
checkpoint_cb = callbacks.ModelCheckpoint("main_model", save_best_only=True, monitor='val_accuracy', mode='max', verbose=1, save_format='tf')
early_stopping_cb = callbacks.EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr_cb = callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir, update_freq='batch', profile_batch=0)

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,  # Puedes ajustar el número de épocas aquí
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb, tensorboard_cb],
    verbose=2  # Cambia a 2 para más detalles durante el entrenamiento
)

# Graficar resultados
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
