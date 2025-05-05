from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define a route to predict fraud
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.get_json()

        # Convert the data to a numpy array and reshape it for prediction
        input_data = np.array(data['input']).reshape(1, -1)

        # Standardize the data
        input_data = scaler.transform(input_data)

        # Make the prediction using the model
        prediction = model.predict(input_data)

        # Return the result as a JSON response
        return jsonify({'fraud': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
