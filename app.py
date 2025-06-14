from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from model import predict_mental_health

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Ensure all required fields are present
        required_fields = [
            'Age', 'Gender', 'Country', 'self_employed', 'family_history',
            'work_interfere', 'no_employees', 'remote_work', 'tech_company',
            'benefits', 'care_options', 'wellness_program', 'seek_help',
            'anonymity', 'leave', 'mental_health_consequence',
            'phys_health_consequence', 'coworkers', 'supervisor',
            'mental_health_interview', 'phys_health_interview',
            'mental_vs_physical', 'obs_consequence'
        ]
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                })
        
        # Convert Age to integer
        try:
            data['Age'] = int(data['Age'])
        except ValueError:
            return jsonify({
                'success': False,
                'error': 'Age must be a number'
            })
        
        # Make prediction
        result = predict_mental_health(data)
        
        return jsonify({
            'success': True,
            'prediction': str(result['prediction']),
            'probability': float(result['probability'])
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 