from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, models
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch import nn
from PIL import Image
import io
import os
import json

app = Flask(__name__)

# Configuration
MODEL_PATH = "chest.ipynb/best_model.pth"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load class names (these should be identical to what was used during training)
class_names = ['adenocarcinoma_left.lower.lobe', 'large.cell.carcinoma_left.hilum', 'normal', 'squamous.cell.carcinoma_left.hilum']

# Define image transforms
transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """Load and prepare the model for inference"""
    # Initialize the model architecture
    model = resnet18(weights=None)  # No pre-trained weights needed here
    
    # Modify final layer for our classification task
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.Linear(256, len(class_names)),
    )
    
    # Load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict_image(image, model):
    """Predict class for a single image"""
    # Convert to RGB (in case the image is grayscale or RGBA)
    image = image.convert("RGB")
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        predicted_class = class_names[predicted_idx.item()]
        confidence = confidence.item()
        probs = probabilities.cpu().squeeze().tolist()
    
    # Return results
    return {
        'class': predicted_class,
        'confidence': confidence,
        'probabilities': dict(zip(class_names, probs))
    }

# Load model at startup
model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Read image
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))
        
        # Make prediction
        result = predict_image(image, model)
        return jsonify(result)
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
