from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from flask_cors import CORS  # Import CORS
from PIL import Image  # For direct image processing

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "./rubberbugs_model.h5"  # Path to your saved model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = ["Cockchafer Grubs", "Mealy Bugs"]

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

    # Preprocess the image directly from memory
    try:
        img = Image.open(file.stream).convert('RGB')  # Open the uploaded image and ensure it's in RGB mode
        img = img.resize((224, 224))  # Resize image to 224x224
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the server
if __name__ == '__main__':
    app.run(debug=True)
