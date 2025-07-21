import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the final model pipeline and selected feature names
model_pipeline = joblib.load('../models/final_model.pkl')
selected_features = pd.read_csv('../data/heart_disease_selected_features.csv').columns.tolist()

# Streamlit UI
st.title('Heart Disease Prediction App')
st.write('Enter patient details to predict heart disease risk (0–4, where 0 is no disease and 4 is severe).')

# Input form for all features
age = st.number_input('Age', min_value=0, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=0, max_value=200, value=120)
chol = st.number_input('Cholesterol (mg/dl)', min_value=0, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T Abnormality', 'Left Ventricular Hypertrophy'])
thalach = st.number_input('Maximum Heart Rate', min_value=0, max_value=220, value=150)
exang = st.selectbox('Exercise Induced Angina', ['No', 'Yes'])
oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
ca = st.number_input('Number of Major Vessels (0–3)', min_value=0, max_value=3, value=0)
thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

# Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == 'Male' else 0],
    'cp': [{'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal Pain': 3, 'Asymptomatic': 4}[cp]],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [1 if fbs == 'Yes' else 0],
    'restecg': [{'Normal': 0, 'ST-T Abnormality': 1, 'Left Ventricular Hypertrophy': 2}[restecg]],
    'thalach': [thalach],
    'exang': [1 if exang == 'Yes' else 0],
    'oldpeak': [oldpeak],
    'slope': [{'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}[slope]],
    'ca': [ca],
    'thal': [{'Normal': 3, 'Fixed Defect': 6, 'Reversible Defect': 7}[thal]]
})

# Predict
if st.button('Predict'):
    # Preprocess input data using the pipeline's preprocessor
    preprocessor = model_pipeline.named_steps['preprocessor']
    X_preprocessed = preprocessor.transform(input_data)
    
    # Get feature names after preprocessing
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']
    feature_names = (numerical_cols + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_cols).tolist())
    
    # Convert to DataFrame and select the same features as in heart_disease_selected_features.csv
    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=feature_names)
    X_selected = X_preprocessed_df[selected_features]
    
    # Predict using the classifier
    prediction = model_pipeline.named_steps['classifier'].predict(X_selected)[0]
    probabilities = model_pipeline.named_steps['classifier'].predict_proba(X_selected)[0]
    
    st.write('**Prediction**: ', f'Heart Disease Level: {prediction} (0 = No Disease, 4 = Severe)')
    st.write('**Probabilities**:')
    for i, prob in enumerate(probabilities):
        st.write(f'Class {i}: {prob:.2%}')

# Visualization
st.subheader('Feature Distribution')
data = pd.read_csv('../data/heart_disease.csv')
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data['age'], kde=True, ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)