import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import cv2

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
MODEL_PATH = 'models/crop_disease_model.h5'
model = None

# Disease classes - 38 classes from PlantVillage dataset
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Load disease information
with open('disease_info.json', 'r') as f:
    disease_info = json.load(f)

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully!")
    else:
        print("Model not found! Please train the model first.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess image for prediction"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def get_disease_info(class_name):
    """Get disease information and recommendations"""
    # Extract crop and disease name
    parts = class_name.split('___')
    crop = parts[0]
    disease = parts[1] if len(parts) > 1 else 'Unknown'
    
    # Get disease info from JSON
    info = disease_info.get(class_name, {
        'description': 'Information not available for this disease.',
        'symptoms': ['No symptoms information available'],
        'causes': ['No causes information available'],
        'prevention': ['Please consult an agricultural expert'],
        'treatment': ['Please consult an agricultural expert']
    })
    
    return {
        'crop': crop.replace('_', ' '),
        'disease': disease.replace('_', ' '),
        'info': info
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        if model is None:
            flash('Model not loaded. Please train the model first.', 'error')
            return redirect(url_for('index'))
        
        img_array = preprocess_image(filepath)
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        predicted_class = CLASS_NAMES[predicted_class_idx]
        disease_data = get_disease_info(predicted_class)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        for idx in top_3_idx:
            top_predictions.append({
                'class': CLASS_NAMES[idx],
                'confidence': float(predictions[0][idx]) * 100
            })
        
        return render_template('result.html',
                             filename=filename,
                             prediction=disease_data,
                             confidence=confidence,
                             top_predictions=top_predictions)
    else:
        flash('Invalid file type. Please upload PNG, JPG, or JPEG files only.', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load the model
    load_trained_model()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
