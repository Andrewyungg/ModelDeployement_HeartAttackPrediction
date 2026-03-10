import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load preprocess and model from MLflow
# Load preprocessor
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def main():
    st.title('Machine Learning Heart Attack Prediction Model Deployment')

    # Add user input components for 5 features
    age = st.slider('Age', min_value=0, max_value=90, value=1)

    sex_option = st.selectbox("Sex", ("Male", "Female"))

    if sex_option == "Male":
        sex = 1
    else:
        sex = 0

    cp = st.selectbox("Chest Pain Type (cp)",[0,1,2,3])
    trestbps = st.slider("trestbps",min_value=80,max_value=200,value=120)

    chol = st.slider("Cholesterol (chol)",min_value=100,max_value=600,value=200)

    fbs = st.selectbox("Fasting Blood Sugar (fbs)",[0,1])
    restecg = st.selectbox("Rest ECG (restecg)",[0,1,2])
    thalach = st.slider("Maximum Heart Rate (thalach)",min_value=60,max_value=220,value=100)

    exang = st.selectbox("Exercise Induced Angina (exang)",[0,1])

    oldpeak = st.slider("Oldpeak",min_value=0.0,max_value=10.0,value=1.0,step=0.1)

    
    slope = st.selectbox("Slope",[0,1,2])

    ca = st.selectbox("Number of Major Vessels (ca)",[0,1,2,3])
    thal = st.selectbox("Thal",[0,1,2,3])


    if st.button('Make Prediction'):
        features = pd.DataFrame([{
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal
                }])
        result = make_prediction(features)
        st.success(f'Prediction: {result}')
        

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(input_array)
    prediction = model.predict(X_scaled)

    prob = model.predict_proba(X_scaled)
    st.write(f"Probability of Heart Attack: {prob[0][1]:.2f}")
    if prediction[0] == 1:
        return "1 (High Risk of Heart Attack)"
    else:
        return "0 (Low Risk of Heart Attack)"
    

if __name__ == '__main__':
    main()


