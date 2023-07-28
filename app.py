from flask import Flask, request, Response, jsonify
from datetime import datetime
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Método para invertir los colores de la imagen


def invert_colors(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    inverted_image = Image.eval(image, lambda x: 255 - x)
    buffered = BytesIO()
    inverted_image.save(buffered, format="PNG")
    return buffered.getvalue()

# Método para obtener la fecha y hora actual


def get_current_datetime():
    return datetime.now().isoformat()

# Ruta para el método POST de procesamiento de imagen


@app.route('/invert', methods=['POST'])
def process_image():
    try:
        image = request.files['img'].read()
        inverted_image_bytes = invert_colors(image)
        return Response(inverted_image_bytes, content_type='image/png')
    except:
        return jsonify({'message': 'Error processing image'}), 500

# Ruta para el método GET de fecha y hora


@app.route('/datetime', methods=['GET'])
def get_datetime():
    try:
        datetime_str = get_current_datetime()
        return jsonify({'datetime': datetime_str})
    except:
        return jsonify({'message': 'Error getting datetime'}), 500


# Ruta principal para mostrar un mensaje
@app.route('/', methods=['GET'])
def index():
    return "¡La API se ha desplegado correctamente!"


if __name__ == '__main__':
    app.run(debug=True)
