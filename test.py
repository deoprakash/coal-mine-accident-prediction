import pickle
import numpy as np

# Load the model
with open('models/coal_mine_model.pkl', 'rb') as model_file:
    lr = pickle.load(model_file)

print("model loaded", lr)
# Example input data (as an array)
x_data = ['Bijuri', 'No', 24.163868, 29.656986, 368.522924, 739.974024, 0.354342, 28.008762]

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

# 1. Encode the location: This will turn the string location into the corresponding one-hot encoded vector
location_encoding = [1 if location_mapping.get(x_data[0], "") == col else 0 for col in location_columns]

# 2. Encode DPM Presence ('Yes' = 1, 'No' = 0)
dpm_encoded = [1 if x_data[1] == 'Yes' else 0]

# 3. Combine numerical data with encoded features (location + DPM)
x_data_encoded = np.array(x_data[2:] + dpm_encoded + location_encoding).reshape(1, -1)

# Make prediction
prediction = lr.predict(x_data_encoded)
print("Prediction:", prediction)
