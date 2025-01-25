import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

scaler = joblib.load('scaler.pkl')  # StandardScaler for years_of_experience
le_education = joblib.load('label_encoder.pkl')  # LabelEncoder for education_level
le_location = joblib.load('label_encoder_location.pkl')  # LabelEncoder for education_level

model = joblib.load('salary_model.pkl') # Load the trained model

def preprocess_input(input_data):

    years_of_experience = scaler.transform([[input_data[0]]])[0][0]


    # Convert education_level to numeric using the same LabelEncoder as used in training
    education_level = le_education.transform([input_data[1]])[0]
    
    # Standardize years_of_experience using the same scaler as used in training

    
    # Convert location to numeric using the same LabelEncoder as used in training
    location = le_location.transform([input_data[2]])[0]


    # Return preprocessed data
    return [years_of_experience, education_level,location]


def predict_salary(input_data):
    # Prepare input for prediction (assume data is preprocessed)
    # Features should match the training data (e.g., normalized years_of_experience, encoded education_level)
    ip = preprocess_input(input_data)
    prediction = model.predict([ip])
    return prediction[0]
    #return 2


