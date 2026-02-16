import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.keras.preprocessing import image

# Cargar el modelo
model = tf.keras.models.load_model("models/IA_ClasificacionAvanzada.keras")

CLASSES = ['Carton', 'Vidrio', 'Metal', 'Papel', 'Plastico']
input_folder = 'test_images/'  # Carpeta de imágenes para probar

# Verificar si la carpeta existe
if not os.path.exists(input_folder):
    print(f"Error: La carpeta {input_folder} no existe.")
    exit()

# Mostrar imágenes y sus predicciones
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    # Cargar y preprocesar la imagen
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer la predicción
    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    # Mostrar la imagen y su categoría
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (400, 400))
    label = f"Clase: {CLASSES[predicted_class]} ({confidence * 100:.2f}%)"
    cv2.putText(original_img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Prediccion de Imagen", original_img)

    # Esperar a que el usuario presione una tecla para continuar
    cv2.waitKey(0)
    cv2.destroyAllWindows()
