import json
import os
import cv2
import matplotlib.pyplot as plt

def draw_bounding_boxes(json_path, json_filename, img_folder, output_folder):
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

        # Guarda la imagen con las cajas delimitadoras en el mismo directorio
        output_filename = os.path.splitext(image_filename)[0] + '_box.jpg'
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path)
        plt.close()

# Si deseas llamar esta función desde otro módulo, podrías hacerlo de la siguiente manera:
# from nombre_del_modulo import draw_bounding_boxes
# draw_bounding_boxes(json_path, json_filename, img_folder, output_folder)
if __name__ == "__main__":
    # Ruta del archivo JSON que contiene los resultados de detección
    json_path = './propio/test'
    json_filename = 'IMG-20230822-WA0005.jpg.json'  # Cambia esto al nombre correcto

    # Ruta de la carpeta que contiene las imágenes originales
    img_folder = './propio/test'

    # Ruta de la carpeta donde se guardarán las imágenes con las cajas delimitadoras
    output_folder = './propio'

    draw_bounding_boxes(json_path, json_filename, img_folder, output_folder)