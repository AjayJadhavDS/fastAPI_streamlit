from fastapi import FastAPI
from pydantic import BaseModel
from api.salary_prediction import predict_salary

app = FastAPI()

class InputData(BaseModel):
    years_of_experience: int
    education_level: str  # Assuming already encoded
    location: str  # You can use location if you plan to add a feature based on location


@app.get("/")
def hello_world():
    return {'Hello': 'world'}


@app.post("/predict")
async def predict(data: InputData):
    # Prepare data for prediction
    input_data = [data.years_of_experience, data.education_level,data.location]
    
    # Make prediction
    predicted_salary = predict_salary(input_data)
    
    return {"predicted_salary": predicted_salary}
