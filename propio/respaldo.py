import torch
import torchvision.models as models
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt
import os

# Definir las clases según las clases en tu conjunto de datos
classes = ['rosas', '0', '1', '2']

# Crear una instancia del modelo con la misma arquitectura
model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, len(classes))

# Cargar los pesos del modelo entrenado
# model.load_state_dict(torch.load('./modelo_entrenado.pth'))
model.load_state_dict(torch.load('./modelo_entrenado.pth',
                      map_location=torch.device('cpu')))

model.eval()  # Poner el modelo en modo de evaluación

# Carpeta que contiene las imágenes a inferir
img_folder = './rosas'

# Probabilidad mínima para considerar una detección
probabilidad = 0.5

# Iterar sobre todas las imágenes en la carpeta
for image_filename in os.listdir(img_folder):
    if image_filename.endswith('.jpg') or image_filename.endswith('.png'):
        image_path = os.path.join(img_folder, image_filename)

        # Preprocesar la imagen
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_tensor(image).unsqueeze(0)

        with torch.no_grad():
            prediction = model(image)

        boxes = prediction[0]['boxes'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()

        # Visualizar los resultados
        fig, ax = plt.subplots(1, figsize=(8, 6))
        ax.imshow(cv2.imread(image_path))
        for box, label, score in zip(boxes, labels, scores):
            if score > probabilidad:
                x, y, x_max, y_max = box
                w, h = x_max - x, y_max - y
                rect = plt.Rectangle(
                    (x, y), w, h, fill=False, color='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, y, f"{classes[label]}: {score:.2f}", color='red')
        plt.show()
