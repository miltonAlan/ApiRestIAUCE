import cv2
import numpy as np
import os


def detect_and_draw_aruco(image_path, output_folder, perimetro_reals):
    # Parámetros necesarios para usar aruco
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

    # Cargar imagen
    img = cv2.imread(image_path)

    # Detectar aruco
    corners, _, _ = cv2.aruco.detectMarkers(
        img, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        # Dibujar polígono en coordenadas detectadas
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

        # Aruco perímetro
        aruco_perimeter = cv2.arcLength(corners[0], True)
        # print('perimetro aruco: ' + str(aruco_perimeter))

        # ratio pixeles a cm
        ratio = aruco_perimeter / perimetro_real
        # print('ratio: ' + str(ratio))

        ancho_pixeles = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
        largo_pixeles = np.linalg.norm(corners[0][0][1] - corners[0][0][2])

        # Get Width and Height of the Objects by applying the Ratio pixel to cm
        object_width = round(ancho_pixeles / ratio, 1)
        object_height = round(largo_pixeles / ratio, 1)


        # Agregar texto en las coordenadas del primer marcador detectado
        marker_center = np.mean(corners[0][0], axis=0)
        ancho = f'ancho: {object_width}'
        cv2.putText(img, ancho, (int(marker_center[0]), int(marker_center[1]) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        largo = f'ancho: {object_height}'
        cv2.putText(img, largo, (int(marker_center[0]), int(marker_center[1]) + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        return ratio

    else:
        print("No se detectó ningún marcador ArUco en la imagen.")

    # Crear la carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)

    # Guardar la imagen procesada en la carpeta de detecciones sin importar si detecta o no
    output_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_path, img)



# Llamar a la función con la ruta de la imagen
image_path = './propio/aruco/2.jpeg'
output_folder = './propio/aruco/detecciones'
perimetro_real = 40
# ratio = detect_and_draw_aruco(image_path, output_folder, perimetro_real)
# print("ratio validacion:" + str(ratio))
