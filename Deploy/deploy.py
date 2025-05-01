from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import json
import os

# Create templates directory if it doesn't exist
templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
    print(f"Created templates directory at {templates_dir}")

app = Flask(__name__)

# Load your model
device = "cpu"  # Ensure this is set to "cpu"
num_classes = 4  # Adjust based on your number of classes

# Define the ResNet18 model with the EXACT same architecture as training
resnet18_model = models.resnet18(weights=None)
resnet18_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# This is the critical part - match the architecture EXACTLY with what was used in training
resnet18_model.fc = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, num_classes)
)

# Load the class names
class_names = ["adenocarcinoma_left.lower.lobe", "large.cell.carcinoma_left.hilum", 
               "normal", "squamous.cell.carcinoma_left.hilum"]

# Try to load class names from json file if it exists
class_names_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                               "chest.ipynb", "class_names.json")
if os.path.exists(class_names_path):
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        print(f"Loaded class names from {class_names_path}")
    except Exception as e:
        print(f"Error loading class names: {e}, using default class names")

# Load the saved model weights
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          "chest.ipynb", "chest-ctscan_model.pth")
try:
    resnet18_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

resnet18_model.to(device)
resnet18_model.eval()

# Define the data transformation - same as what was used for evaluation
data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the prediction function
def predict_image(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
    return {
        "class_idx": predicted_idx.item(),
        "confidence": confidence.item(),
        "probabilities": probabilities.squeeze().tolist()
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:   
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if not file:
            return jsonify({'error': 'No file provided'}), 400

        image = Image.open(file.stream).convert("RGB")
        result = predict_image(image, resnet18_model, data_transform, device)
        
        prediction = {
            'predicted_class': class_names[result["class_idx"]],
            'confidence': result["confidence"],
            'all_probabilities': {name: prob for name, prob in zip(class_names, result["probabilities"])}
        }
        
        return jsonify(prediction)
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
