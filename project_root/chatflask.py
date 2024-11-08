from flask import Flask, request, jsonify

from src.utils.disease import disease_dic

import sys
sys.path.append('/opt/render/project/src')


# Access disease_dic directly or call get_disease_info if needed

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
    try:
        transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # Ensure image is in RGB mode
        img_t = transform(img)
        img_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            prediction = disease_model(img_t)
            _, predicted_class = torch.max(prediction, 1)
            return disease_classes[predicted_class.item()]
    except Exception as e:
        print(f"Error in predicting disease: {e}")
        return None

# Endpoint to handle disease prediction
@app.route('/predict_disease', methods=['POST'])
def predict_disease_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = file.read()
    disease = predict_disease(img_bytes)
    
    if disease:
        return jsonify({'disease': disease, 'description': disease_dic.get(disease, "No description available")})
    else:
        return jsonify({'error': 'Failed to process image or predict disease'}), 500

# Endpoint for crop recommendation
@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    try:
        data = request.get_json()
        # Retrieve feature values from the request data
        N = data['N']
        P = data['P']
        K = data['K']
        ph = data['ph']
        rainfall = data['rainfall']
        temperature = data.get('temperature', 25)  # Default values if not provided
        humidity = data.get('humidity', 60)
        
        # Prepare features for prediction
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        recommended_crop = crop_recommendation_model.predict(features)[0]
        
        return jsonify({'recommended_crop': recommended_crop})
    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {e}'}), 400
    except Exception as e:
        print(f"Error in crop recommendation: {e}")
        return jsonify({'error': 'Failed to recommend crop'}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'API is up and running'}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
    print("Starting Flask app...")
