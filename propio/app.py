import torch
import torchvision.models as models
from torchvision.transforms import functional as F
import cv2
import matplotlib.pyplot as plt
import os

def perform_object_detection(img_folder, model_weights_path, show_image=True, save_path=None):
    # Definir las clases según las clases en tu conjunto de datos
    classes = ['rosas', '0', '1', '2']

    # Crear una instancia del modelo con la misma arquitectura
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, len(classes))

    # Cargar los pesos del modelo entrenado
    model.load_state_dict(torch.load(model_weights_path,
                          map_location=torch.device('cpu')))
    
    model.eval()  # Poner el modelo en modo de evaluación

    probabilidad = 0.5

    for image_filename in os.listdir(img_folder):
        if image_filename.endswith('.jpg') or image_filename.endswith('.png') or image_filename.endswith('.jpeg'):
            image_path = os.path.join(img_folder, image_filename)

            # Preprocesar la imagen
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = F.to_tensor(image).unsqueeze(
                0)  # Mantener el tensor para inferencia

            with torch.no_grad():
                prediction = model(image_tensor)

            boxes = prediction[0]['boxes'].cpu().numpy()
            labels = prediction[0]['labels'].cpu().numpy()
            scores = prediction[0]['scores'].cpu().numpy()

            # Visualizar los resultados
            fig, ax = plt.subplots(1, figsize=(8, 6))

            ax.imshow(image)  # Mostrar imagen RGB

            for box, label, score in zip(boxes, labels, scores):
                if score > probabilidad:
                    x, y, x_max, y_max = box
                    w, h = x_max - x, y_max - y
                    rect = plt.Rectangle(
                        (x, y), w, h, fill=False, color='red', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, y, f"{classes[label]}: {score:.2f}", color='red')


            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

            save_filename = os.path.splitext(image_filename)[0] + '_processed.jpg'
            save_image_path = os.path.join(save_path, save_filename)
            plt.savefig(save_image_path)                

            if show_image:
                plt.show()

# Si este módulo se ejecuta directamente
if __name__ == "__main__":
    img_folder = './propio/rosasPre'
    model_weights_path = './propio/modelo_entrenado.pth'
    show_image = False
    save_path = './propio/rosasPost'
    perform_object_detection(img_folder, model_weights_path, show_image, save_path)   
                
