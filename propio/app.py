import torch
import torchvision.models as models
from torchvision.transforms import functional as F
import cv2
# import matplotlib.pyplot as plt
import matplotlib

from propio.comprobaciones import draw_bounding_boxes
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import numpy as np

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    raise TypeError(f"Type {type(obj)} not serializable")

def save_detection_results_to_json(json_path, json_filename, detection_results):
    if not json_filename.endswith('.json'):
        json_filename += '.json'
    
    json_file_path = os.path.join(json_path, json_filename)
    
    with open(json_file_path, 'w') as json_file:
        json.dump(detection_results, json_file, default=convert_to_serializable)

    return json_file_path

def perform_object_detection(image_filename, ratio, img_folder, model_weights_path, show_image=True, save_path=None, json_path=None):
    classes = ['rosas', '0', '1']
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, len(classes))
    model.load_state_dict(torch.load(model_weights_path,
                          map_location=torch.device('cpu')))
    model.eval()
    probabilidad = 0.60
    detection_results = []

    # for image_filename in os.listdir(img_folder):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png') or image_filename.endswith('.jpeg'):
        image_path = os.path.join(img_folder, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
         prediction = model(image_tensor)

         boxes = prediction[0]['boxes'].cpu().numpy()
         labels = prediction[0]['labels'].cpu().numpy()
         scores = prediction[0]['scores'].cpu().numpy()

         result = {
             'image_filename': image_filename,
             'ratio' : ratio,
             'detections': []
         }
         # paths imagenes
         # print('XXXXXXXXXXXXXXXXXXXX'+image_filename)

        for box, label, score in zip(boxes, labels, scores):
            if score > probabilidad:
                x, y, x_max, y_max = box
                w, h = x_max - x, y_max - y
                detection_info = {
                    'box': [x, y, w, h],
                    'label': classes[label],
                    'score': score
                }
                result['detections'].append(detection_info)
# Resto del código...

        for box1, label1, score1 in zip(boxes, labels, scores):
            if score1 > probabilidad:
                x1, y1, x_max1, y_max1 = box1
                w1, h1 = x_max1 - x1, y_max1 - y1

                for box2, label2, score2 in zip(boxes, labels, scores):
                    if label1 != label2 and score2 > probabilidad:
                        x2, y2, x_max2, y_max2 = box2
                        w2, h2 = x_max2 - x2, y_max2 - y2

                        # Comprueba si las cajas 1 y 2 se intersectan verticalmente
                        if max(x1, x2) < min(x_max1, x_max2):
                            # Calcula las coordenadas del nuevo 'box' que cubre ambas clases
                            new_x = min(x1, x2) - 20  # Resta 10 píxeles al límite izquierdo
                            new_y = min(y1, y2) - 20  # Resta 10 píxeles al límite superior
                            new_x_max = max(x_max1, x_max2) + 20  # Suma 10 píxeles al límite derecho
                            new_y_max = max(y_max1, y_max2) + 20  # Suma 10 píxeles al límite inferior
                            new_w = new_x_max - new_x
                            new_h = new_y_max - new_y


                            # Agrega el nuevo 'box' con label 'rosa'
                            detection_info = {
                                'box': [new_x, new_y, new_w, new_h],
                                'label': 'rosa',
                                'score': max(score1, score2)
                            }
                            result['detections'].append(detection_info)

        # Resto del código...

        detection_results.append(result)

            # if save_path:
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     save_filename = os.path.splitext(image_filename)[0]
            #     save_image_path = os.path.join(save_path, save_filename)
            #     plt.imshow(image)
            #     ax = plt.gca()

            #     for detection in result['detections']:
            #         x, y, w, h = detection['box']
            #         rect = plt.Rectangle(
            #             (x, y), w, h, fill=False, color='red', linewidth=2)
            #         ax.add_patch(rect)
            #         ax.text(x, y, f"{detection['label']}: {detection['score']:.2f}", color='red')

            #     plt.savefig(save_image_path)
        save_detection_results_to_json(json_path, image_filename, detection_results)
        # Reiniciar la lista detection_results para el siguiente ciclo
        detection_results = []
        result['detections'] = []  # Clear the list


            # if show_image:
            #     plt.show()
            # plt.clf()
            # se pintan las cajas x verificacion mientra no existe cliente...
            # Ruta del archivo JSON que contiene los resultados de detección
        json_path = './propio/rosasPost'
        json_filename = image_filename +'.json'  # Cambia esto al nombre correcto
        
            # Ruta de la carpeta que contiene las imágenes originales
        img_folder = './propio/rosasPre'
        
            # Ruta de la carpeta donde se guardarán las imágenes con las cajas delimitadoras
        output_folder = './propio/rosasPost'
        
        draw_bounding_boxes(json_path, json_filename, img_folder, output_folder)

# Si este módulo se ejecuta directamente
if __name__ == "__main__":
    img_folder = './propio/rosasPre'
    model_weights_path = './propio/modelo_entrenado.pth'
    show_image = True
    save_path = './propio/rosasPost'
    json_path = './propio/rosasPost'
    perform_object_detection(img_folder, model_weights_path, show_image, save_path, json_path)   
                
