from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression  # Assuming you're using logistic regression as a model
import pickle

# Load the model
with open('models/coal_mine_model.pkl', 'rb') as model_file:
    lr = pickle.load(model_file)


app = Flask(__name__)
CORS(app)

# Define feature columns, this should match the order you expect from the form input
location_columns = ['Mine Location_Bhatgaon', 'Mine Location_Bijuri', 'Mine Location_Dipka', 
                    'Mine Location_Gevra', 'Mine Location_Korea', 'Mine Location_Kusmunda']

location_mapping = {
    "Bhatgaon": "Mine Location_Bhatgaon",
    "Bijuri": "Mine Location_Bijuri",
    "Dipka": "Mine Location_Dipka",
    "Gevra": "Mine Location_Gevra",
    "Korea": "Mine Location_Korea",
    "Kusmunda": "Mine Location_Kusmunda"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'POST': 
        # Print the incoming form data for debugging
            location = request.form['location']
            dpm = request.form['dpm']
            sub_level = float(request.form['sub_level'])
            supra_level = float(request.form['supra_level'])
            particle_count = float(request.form['particle_count'])
            analyzed_area = float(request.form['analyzed_area'])
            loading_density = float(request.form['loading_density'])
            abundance = float(request.form['abundance'])

        # Encode the location (one-hot encoding)
            location_encoding = [1 if location_mapping.get(location, "") == col else 0 for col in location_columns]

            # Encode DPM presence
            dpm_encoded = [1 if dpm == 'Yes' else 0]

            # Prepare the input data (combine features)
            x_data = [sub_level, supra_level, particle_count, analyzed_area, loading_density, abundance]
            x_data_encoded = np.array(x_data + dpm_encoded + location_encoding).reshape(1, -1)

            # Predict (this assumes you have already trained the model)
            prediction = lr.predict(x_data_encoded)  # Replace `lr` with your actual trained model

            # Return the result
            result = "Accident" if prediction[0] == 1 else "No Accident"
            return result
    
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True, use_reloader = False)


