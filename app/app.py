import streamlit as st
import requests

st.title('Salary Prediction')

# Input fields for the user
years_of_experience = st.slider("Years of Experience", 0, 40)
education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
location = st.selectbox("Location",['California', 'New York', 'Texas'])

# Call the FastAPI backend for prediction when the user clicks the 'Predict' button
if st.button('Predict Salary'):
    # Convert education level to numeric (you need the same encoding logic used in training)
    # education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    # education_code = education_mapping.get(education_level, 0)  # Default to 'High School' encoding
    
    # location_mapping = {"California": 0, "New York": 1, "Texas": 2}
    # location_code = location_mapping.get(location, 0)  # Default to 'California' encoding


    response = requests.post(
        "http://127.0.0.1:8000/predict", 
        json={"years_of_experience": years_of_experience, "education_level": education_level, "location": location}
    )
    prediction = response.json()
    print(prediction)
    st.write(f"Predicted Salary: ${prediction['predicted_salary']}")


