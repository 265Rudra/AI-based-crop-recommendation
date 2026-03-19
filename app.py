from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

def predict_crop(features):
    rf_p = rf.predict([features])[0]
    svm_p = svm.predict([features])[0]
    xgb_p = xgb_model.predict([features])[0]
    # Majority vote
    predictions = [rf_p, svm_p, xgb_p]
    final_p = np.bincount(predictions).argmax()
    
    crop_name = le.inverse_transform([final_p])[0]
    rf_name = le.inverse_transform([rf_p])[0]
    svm_name = le.inverse_transform([svm_p])[0]
    xgb_name = le.inverse_transform([xgb_p])[0]
    
    return {
        'predicted_crop': crop_name,
        'rf': rf_name,
        'svm': svm_name,
        'xgb': xgb_name
    }

# Load models (without function)
models_dict = joblib.load('models.joblib')
rf = models_dict['rf']
svm = models_dict['svm']
xgb_model = models_dict['xgb']
le = models_dict['le']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])
        
        result = predict_crop(features[0])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
