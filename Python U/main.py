from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "./rubberbugs_model.h5"  # Path to your saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ["Cockchafer Grubs","Mealy Bugs",]

# Upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# API Endpoint: Health Check
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200

# API Endpoint: Predict
@app.route('/bug', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image
    img = load_img(file_path, target_size=(224, 224))  # Resize image to 224x224
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch size

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[0][predicted_class_idx]

    # Get the class label
    predicted_class = class_names[predicted_class_idx]

    # Return prediction as JSON
    result = {
        'class': predicted_class,
        'confidence': float(confidence)
    }
    return jsonify(result)

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
