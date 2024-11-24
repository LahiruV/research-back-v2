from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and origins

# Load the models and scaler
models = {
    'linear_regression': joblib.load('linear_regression_modelp.pkl'),
    'gradient_boosting': joblib.load('gradient_boosting_modelp.pkl'),
    'random_forest': joblib.load('random_forest_modelp.pkl')
}
scaler = joblib.load('scalerp.pkl')

@app.route('/')
def home():
    return "Rubber Price Prediction API is running!"

@app.route('/price', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        input_data = request.json

        # Extract features from input
        year = input_data.get('Year')
        month = input_data.get('Month')

        # Validate inputs
        if year is None or month is None:
            return jsonify({'error': 'Missing required fields: Year and Month'}), 400
        if not isinstance(month, int) or month < 1 or month > 12:
            return jsonify({'error': f'Invalid month: {month}. It must be an integer between 1 and 12.'}), 400

        # Generate cyclical features
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        year_month = year + (month - 1) / 12

        # Prepare input features for prediction
        feature_data = pd.DataFrame({
            'Year_Month': [year_month],
            'Month_Sin': [month_sin],
            'Month_Cos': [month_cos]
        })

        # Scale the features
        feature_data_scaled = scaler.transform(feature_data)

        # Make predictions using ensemble
        predictions = {
            'Linear Regression': models['linear_regression'].predict(feature_data_scaled)[0],
            'Gradient Boosting': models['gradient_boosting'].predict(feature_data_scaled)[0],
            'Random Forest': models['random_forest'].predict(feature_data_scaled)[0]
        }

        # Ensemble prediction (weighted average)
        ensemble_prediction = (
            0.3 * predictions['Linear Regression'] +
            0.5 * predictions['Gradient Boosting'] +
            0.2 * predictions['Random Forest']
        )

        # Return predictions
        response = {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_prediction
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)