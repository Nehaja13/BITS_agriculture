from flask import Flask, request, jsonify
import numpy as np
import os
import torch
import pickle
from PIL import Image
from torchvision import transforms
import io

# Import dictionaries and models
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# Initialize Flask app
app = Flask(__name__)

# Load trained models and classes
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', ..., 'Tomato___healthy']
disease_model_path = 'models/plant_disease_model.pth'
crop_recommendation_model_path = 'models/RandomForest.pkl'

# Load the disease prediction model
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Load the crop recommendation model
with open(crop_recommendation_model_path, 'rb') as f:
    crop_recommendation_model = pickle.load(f)

# Helper function to predict crop disease
def predict_disease(img_bytes):
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    img = Image.open(io.BytesIO(img_bytes))
    img_t = transform(img)
    img_t = torch.unsqueeze(img_t, 0)
    with torch.no_grad():
        prediction = disease_model(img_t)
        _, predicted_class = torch.max(prediction, 1)
        return disease_classes[predicted_class.item()]

# Endpoint to handle disease prediction
@app.route('/predict_disease', methods=['POST'])
def predict_disease_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = file.read()
    disease = predict_disease(img_bytes)
    return jsonify({'disease': disease, 'description': disease_dic[disease]})

# Endpoint for crop recommendation
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    data = request.get_json()
    N = data['N']
    P = data['P']
    K = data['K']
    ph = data['ph']
    rainfall = data['rainfall']
    temperature = data.get('temperature', 25)  # Example default values
    humidity = data.get('humidity', 60)

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    recommended_crop = crop_recommendation_model.predict(features)[0]
    return jsonify({'recommended_crop': recommended_crop})

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is up and running'}), 200

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

