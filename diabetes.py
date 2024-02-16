from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        new_patient = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(new_patient)

        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)

        return jsonify({"predictions": binary_predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
