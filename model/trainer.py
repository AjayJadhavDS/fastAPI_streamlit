import joblib
import pandas as pd
from preprocess import preprocess_data
from model import create_model
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train_model(data_path):
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Initialize the model
    model = create_model()
    
    # Train the model
    model.fit(X, y)
    
    # Save the trained model to a file
    joblib.dump(model, 'salary_model.pkl')
    

    
    print("Model training complete and saved as salary_model.pkl")