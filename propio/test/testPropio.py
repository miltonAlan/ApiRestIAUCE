import json
import os
import cv2
import matplotlib.pyplot as plt

# Ruta del archivo JSON que contiene los resultados de detección
json_path = './propio/test'
json_filename = '123_jpg.rf.e18f0d9b32b083d1c988f58d59601fc5.jpg.json'  # Cambia esto al nombre correcto

# Ruta de la carpeta que contiene las imágenes originales
img_folder = './propio/test'

# Lee el archivo JSON
with open(os.path.join(json_path, json_filename), 'r') as json_file:
    detection_results = json.load(json_file)

# Itera a través de los resultados de detección
for result in detection_results:
    image_filename = result['image_filename']
    detections = result['detections']

    # Carga la imagen original
    image_path = os.path.join(img_folder, image_filename)
    print(f"Loading image from: {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Dibuja las cajas, etiquetas y puntuaciones en la imagen
    plt.imshow(image)
    ax = plt.gca()

    for detection in detections:
        x, y, w, h = detection['box']
        label = detection['label']
        score = detection['score']

        rect = plt.Rectangle(
            (x, y), w, h, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, f"{label}: {score:.2f}", color='red')

    plt.show()
