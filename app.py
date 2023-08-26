from flask import Flask, request, Response, jsonify
from datetime import datetime
from PIL import Image
from io import BytesIO
import os
import json
from werkzeug.serving import WSGIRequestHandler
from propio.app import perform_object_detection
from propio.aruco.aruco import detect_and_draw_aruco

app = Flask(__name__)

# Método para invertir los colores de la imagen
def invert_colors(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    inverted_image = Image.eval(image, lambda x: 255 - x)
    return inverted_image

# Método para obtener la fecha y hora actual
def get_current_datetime():
    return datetime.now().isoformat()

def propio(image_bytes, save_path, original_filename):
    try:
        # guardamos la imagen para su procesado posterior
        image_path = os.path.join(save_path, original_filename)
        
        with open(image_path, 'wb') as image_file:
            image_file.write(image_bytes)

        print('save_path: ' + save_path)
        print('original_filename: ' + original_filename)
        print('combinacion: ' + save_path +'/'+ original_filename)

        path_rosa = save_path +'/'+ original_filename
        perimetro_real = 40

        ratio = detect_and_draw_aruco(path_rosa, save_path, perimetro_real)
        print("ratio validacion:" + str(ratio))

        # procesamos y guardamos el JSON
        img_folder = './propio/rosasPre'
        model_weights_path = './propio/modelo_entrenado.pth'
        show_image = False
        save_path = './propio/rosasPost'
        json_path = './propio/rosasPost'
        perform_object_detection(original_filename, ratio, img_folder, model_weights_path, show_image, save_path, json_path)   

        # obtenemos el json
        # Lee el archivo JSON

        json_filename = original_filename + ".json"
        with open(os.path.join(json_path, json_filename), 'r') as json_file:
            detection_results = json.load(json_file)

        # Imprimir los resultados de detección
        print("Detection Results:")
        # print(detection_results)

        # return jsonify({'message': 'Image saved successfully', 'image_path': image_path})
        return jsonify(detection_results)
    except Exception as e:
        return jsonify({'message': 'Error saving image', 'error': str(e)}), 500

# Ruta para el método POST de procesamiento de imagen y función 'propio'
@app.route('/propio', methods=['POST'])
def invoke_propio():
    try:
        image = request.files['img'].read()
        save_path = './propio/rosasPre'  # Ruta parametrizable
        original_filename = request.files['img'].filename
        response_json = propio(image, save_path, original_filename)
        return response_json
    except:
        return jsonify({'message': 'Error invoking propio function'}), 500

# Ruta principal para mostrar un mensaje
@app.route('/', methods=['GET'])
def index():
    return "¡La API se ha desplegado correctamentexxxxxxx!"

# Configura WSGIRequestHandler para protocolo HTTP/1.1
if __name__ == '__main__':
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(debug=True, host='0.0.0.0', port=5000)
