# Clasificacion de Residuos

- Este proyecto clasifica imagenes de residuos en cinco categorias (Carton, Vidrio, Metal, Papel, Plastico) utilizando un modelo de red neuronal convulcional (CNN) basado en MobileNetV2. Incluye scripts para entrenar el modelo y realizar predicciones a partir de nuevas imagenes.

## Estructura de carpetas del Proyecto
- models/
    - IA_ClasificacionAvanzada.keras
- data/
    - cardboard/ 
    - glass/
    - metal/
    - paper/
    - plastic/
- test_images/
- Train.py
- main.py

## Requisitos Previos para utilizar el programa
1. Python 3.7 o superior.
2. Instalacion de bibliotecas necesarias: 
`pip install tensorflow opencv-python numpy`

## Instrucciones de Ejecucion
### Entrenar el modelo
1. Organiza las imagenes de entrenamiento en sus respectivas carpetas dentro de data
2. Ejecutar el script de entrenamiento:
`python Train.py `

Durante el entrenamiento, el programa mostrara el proceso en cada epoca, junto con la precision y perdida de validacion. Al finalizar el modelo se guardara en models/IA_ClasificacionAvanzada.keras.

### Ejecutar Predicciones
1. Asegurarse de que hay imagenes en la carpeta test_images con formatos compatibles (.png, .jpg, .jpeg).

2. Ejecuta el script de prediccion:
` python main.py `

3. El programa abrira una ventana para cada imagen, en esta se mostrara la imagen, la clase y el nivel de confianza. Presiona cualquier tecla para cerrar la ventana y avanzar a la siguiente imagen.

## Importante
- El modelo espera imagenes de entrada con tamaño 128x128
- Si hay discrepancias en el numero de clases dentro de data/, el porgrama arrojara un error.
