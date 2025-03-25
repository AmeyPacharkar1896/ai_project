from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model and scaler from the model folder
try:
    model = joblib.load('model/rf_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
except Exception as e:
    print("Error loading model/scaler:", str(e))
    model = None
    scaler = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded.'}), 500

    try:
        data = request.get_json(force=True)
    except Exception as e:
        return jsonify({'error': 'Invalid JSON input.'}), 400

    try:
        input_features = np.array(data['features']).reshape(1, -1)
    except Exception as e:
        return jsonify({'error': 'Input data format error. Expected key "features" with a list of numerical values.'}), 400

    expected_features = scaler.n_features_in_
    if input_features.shape[1] != expected_features:
        return jsonify({
            'error': f"Incorrect number of features. Expected {expected_features}, but got {input_features.shape[1]}."
        }), 400

    try:
        input_features_scaled = scaler.transform(input_features)
    except Exception as e:
        return jsonify({'error': 'Error during feature scaling: ' + str(e)}), 500

    try:
        prediction = model.predict(input_features_scaled)
        probability = model.predict_proba(input_features_scaled).tolist()[0]
    except Exception as e:
        return jsonify({'error': 'Error during prediction: ' + str(e)}), 500

    # Create a user-friendly message based on the prediction
    if int(prediction[0]) == 0:
        message = "Credit Card Application Approved."
    else:
        message = "Credit Card Application Not Approved."

    result = {
        'message': message,
        'prediction': int(prediction[0]),  # optional; remove if not needed
    }
    return jsonify(result)

@app.route('/get_options', methods=['GET'])
def get_options():
    try:
        # Read the training data
        df = pd.read_csv("Training Data.csv")
        # For consistency, return only the top 30 most frequent professions and cities
        profession_options = df['profession'].value_counts().head(30).index.tolist()
        city_options = df['city'].value_counts().head(30).index.tolist()
        # Optionally sort them if desired
        profession_options.sort()
        city_options.sort()
        return jsonify({
            'profession_options': profession_options,
            'city_options': city_options
        })
    except Exception as e:
        return jsonify({'error': 'Error fetching options: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
