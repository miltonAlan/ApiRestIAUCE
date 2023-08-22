import torch
import torchvision.models as models
from torchvision.transforms import functional as F
import cv2
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

def perform_object_detection(img_folder, model_weights_path, show_image=True, save_path=None, json_path=None):
    classes = ['rosas', '0', '1', '2']
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, len(classes))
    model.load_state_dict(torch.load(model_weights_path,
                          map_location=torch.device('cpu')))
    model.eval()
    probabilidad = 0.5
    detection_results = []

    for image_filename in os.listdir(img_folder):
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
                'detections': []
            }

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

            detection_results.append(result)

            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_filename = os.path.splitext(image_filename)[0] + '_processed.jpg'
                save_image_path = os.path.join(save_path, save_filename)
                plt.imshow(image)
                ax = plt.gca()

                for detection in result['detections']:
                    x, y, w, h = detection['box']
                    rect = plt.Rectangle(
                        (x, y), w, h, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, y, f"{detection['label']}: {detection['score']:.2f}", color='red')

                plt.savefig(save_image_path)
                save_detection_results_to_json(json_path, image_filename, detection_results)

                if show_image:
                    plt.show()
                # plt.clf()

# Si este módulo se ejecuta directamente
if __name__ == "__main__":
    img_folder = './propio/rosasPre'
    model_weights_path = './propio/modelo_entrenado.pth'
    show_image = False
    save_path = './propio/rosasPost'
    json_path = './propio/rosasPost'
    perform_object_detection(img_folder, model_weights_path, show_image, save_path, json_path)   
                