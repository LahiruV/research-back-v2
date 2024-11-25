from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from flask_cors import CORS  # Enable CORS
from PIL import Image  # For image processing

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
MODEL_PATH = "./leaf_disease_model.h5"  # Update with your model's path
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define class names
class_names = ["birdeyespot", "corynespora"]

# API Endpoint: Health Check
@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint to ensure the API is running.
    """
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200

# API Endpoint: Predict
@app.route('/disease', methods=['POST'])
def predict():
    """
    Predict the disease class for the uploaded leaf image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Process the image
    try:
        # Open the image file and preprocess
        img = Image.open(file.stream).convert('RGB')  # Ensure it's in RGB mode
        img = img.resize((224, 224))  # Resize to 224x224
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Perform prediction
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions)
        confidence = predictions[0][predicted_class_idx]

        # Get the class label
        predicted_class = class_names[predicted_class_idx]

        # Construct the result
        result = {
            'class': predicted_class,
            'confidence': float(confidence)
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handling for unsupported routes
@app.errorhandler(404)
def not_found(error):
    """
    Handle unsupported routes with a 404 error.
    """
    return jsonify({'error': 'Endpoint not found'}), 404

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5004)
