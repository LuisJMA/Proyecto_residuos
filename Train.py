# Archivo: entrenamiento.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Definir las categorías
CLASSES = ['Carton', 'Vidrio', 'Metal', 'Papel', 'Plastico']

# Validar el número de clases disponibles en el dataset
dataset_path = 'data/'
class_folders = [folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))]

if len(class_folders) != len(CLASSES):
    print(f"Error: Se encontraron {len(class_folders)} clases en el dataset, pero el modelo espera {len(CLASSES)}.")
    print("Asegúrate de que solo haya las siguientes carpetas:", CLASSES)
    exit()

# Crear MobileNetV2 preentrenado
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Agregar capas personalizadas
inputs = Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(len(CLASSES), activation='softmax')(x)
model = Model(inputs, outputs)

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Aumento de datos
data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    dataset_path,
    target_size=(128, 128),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Mensaje de inicio de entrenamiento
print("Inicio del entrenamiento del modelo...")

# Entrenamiento del modelo
model.fit(train_data, validation_data=val_data, epochs=15)

eval_loss, eval_acc = model.evaluate(val_data)

# Guardar el modelo
model.save("models/IA_ClasificacionAvanzada.keras")

# Mensaje de finalización
print("Modelo entrenado y guardado exitosamente en 'models/IA_ClasificacionAvanzada.keras'")