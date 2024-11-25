from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models and scaler
models = {
    "linear_regression": joblib.load("linear_regression_model.pkl"),
    "gradient_boosting": joblib.load("gradient_boosting_model.pkl"),
    "random_forest": joblib.load("random_forest_model.pkl")
}
scaler = joblib.load("scaler.pkl")  # Save and load the scaler separately when saving the models.

@app.route('/amount', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Extract and validate features
        required_fields = ['Year', 'Month']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        year = data['Year']
        month = data['Month']

        # Create feature set
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        year_month = year + (month - 1) / 12
        features = pd.DataFrame([[year_month, month_sin, month_cos]], 
                                 columns=['Year_Month', 'Month_Sin', 'Month_Cos'])
        
        # Scale features
        features_scaled = scaler.transform(features)

        # Generate predictions
        predictions = {
            "Linear Regression": float(models["linear_regression"].predict(features_scaled)),
            "Gradient Boosting": float(models["gradient_boosting"].predict(features_scaled)),
            "Random Forest": float(models["random_forest"].predict(features_scaled)),
        }
        
        # Ensemble prediction
        ensemble_prediction = (
            0.3 * predictions["Linear Regression"] +
            0.5 * predictions["Gradient Boosting"] +
            0.2 * predictions["Random Forest"]
        )
        predictions["Ensemble"] = ensemble_prediction
        
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)