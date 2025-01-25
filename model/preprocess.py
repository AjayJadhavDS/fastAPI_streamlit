import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def preprocess_data(df):
    # Convert categorical features to numerical (e.g., encoding 'Education Level')
    le_education = LabelEncoder()
    df['education_level'] = le_education.fit_transform(df[['education_level']].values)
    joblib.dump(le_education, 'label_encoder.pkl')
    
    # Standardize numeric columns
    scaler = StandardScaler()
    df[['years_of_experience']] = scaler.fit_transform(df[['years_of_experience']].values)
    joblib.dump(scaler, 'scaler.pkl')

    le_location = LabelEncoder()
    df['location'] = le_location.fit_transform(df[['location']].values)
    joblib.dump(le_location, 'label_encoder_location.pkl')

    # Split data into features and target
    X = df.drop(columns=['salary'])
    y = df['salary']
    
    return X, y
