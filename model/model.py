from sklearn.ensemble import RandomForestRegressor

def create_model():
    # Initialize and return the model (Random Forest in this case)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print('model has been built')
    return model
