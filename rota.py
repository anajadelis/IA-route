import numpy as np
from PIL import Image
from flask import Flask, request, 
import joblib
import model.ipynb


# Carregar o modelo treinado
model = joblib.load('model.ipynb')

# Função de pré-processamento da imagem
def preprocess_image(image):
    # Redimensionar a imagem para o tamanho esperado pelo modelo
    image = image.resize((224, 224))
    # Converter a imagem para um array numpy
    image_array = np.array(image)
    # Normalizar os valores dos pixels entre 0 e 1
    image_array = image_array / 255.0
    # Adicionar uma dimensão extra para representar o batch (1 imagem no nosso caso)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def process_image():
    # Verificar se uma imagem foi enviada na requisição
    if 'image' not in request.files:
        return jsonify({'error': 'No image sent'})
    
    # Carregar a imagem enviada na requisição
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Pré-processar a imagem
    processed_image = preprocess_image(image)
    
    # Fazer a predição com o modelo
    prediction = model.predict(processed_image)
    
    # Retornar o resultado da predição
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run()
