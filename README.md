# Heart Disease Prediction Project

This project implements a machine learning pipeline to predict heart disease severity (0–4) using the UCI Heart Disease Dataset. It includes data preprocessing, PCA, feature selection, supervised and unsupervised learning, hyperparameter tuning, and a Streamlit web app for real-time predictions.

## Project Structure
```
Heart_Disease_Project/
├── data/
│   ├── heart_disease.csv
│   ├── heart_disease_preprocessed.csv
│   ├── heart_disease_pca.csv
│   ├── heart_disease_selected_features.csv
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   ├── 07_model_export.ipynb
├── models/
│   ├── preprocessor.pkl
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   ├── svm_model.pkl
│   ├── random_forest_optimized.pkl
│   ├── svm_optimized.pkl
│   ├── final_model.pkl
├── ui/
│   ├── app.py
├── results/
│   ├── chi2_results.csv
│   ├── model_performance.csv
│   ├── clustering_results.csv
│   ├── hyperparameter_tuning_results.csv
├── requirements.txt
├── README.md
├── .gitignore
```

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Heart_Disease_Project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - Ensure `data/heart_disease.csv` contains the UCI Heart Disease Dataset.

4. **Run the Notebooks**:
   - Execute the notebooks in order (`01_data_preprocessing.ipynb` to `07_model_export.ipynb`) in a Jupyter environment to preprocess data, perform PCA, select features, train models, perform clustering, tune hyperparameters, and export the final model.

5. **Run the Streamlit App**:
   ```bash
   streamlit run ui/app.py
   ```
   - Access the app at `http://localhost:8501`.
   - Enter patient details to predict heart disease severity and view the age distribution.

## Dataset
The project uses the UCI Heart Disease Dataset (`data/heart_disease.csv`), which includes 13 features (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal) and a multiclass target (0–4, indicating no disease to severe).

## Deliverables
- Preprocessed dataset (`data/heart_disease_preprocessed.csv`)
- PCA-transformed dataset (`data/heart_disease_pca.csv`)
- Selected features dataset (`data/heart_disease_selected_features.csv`)
- Trained models (`models/*.pkl`)
- Final model pipeline (`models/final_model.pkl`)
- Performance metrics and clustering results (`results/*.csv`)
- Streamlit app for real-time predictions (`ui/app.py`)

## Notes
- The project assumes a multiclass target (0–4). Adjust the code if your dataset uses a binary target.
- Visualizations (e.g., ROC curves, elbow plots) are displayed in the notebooks but not saved to disk.
- For public deployment, consider using Ngrok (instructions available in `deployment/ngrok_setup.txt` if needed).