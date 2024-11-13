import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
hdbscan_clusterer = joblib.load('hdbscan_model.pkl')
mlp_model = load_model("mlp_electrification_classifier.h5")

# Streamlit app title and description
st.title("Electrification Solution Predictor")
st.write("""
Upload a CSV file containing data for new regions, and the app will predict whether grid extension or a wind microgrid is the best solution based on population density, grid value, and wind speed.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Prediction function
def predict_electrification_solution(new_data):
    # Preprocess the data
    new_data_scaled = scaler.transform(new_data)
    new_data_reduced = pca.transform(new_data_scaled)
    
    # Apply HDBSCAN clustering
    clusters = hdbscan_clusterer.fit_predict(new_data_reduced)
    new_data['Cluster'] = clusters
    
    # Predict with the MLP model
    new_data_classification = new_data[['PCA_Component_1', 'PCA_Component_2']]
    predictions = mlp_model.predict(new_data_classification)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Map predictions to solutions
    solutions = ["Grid Extension", "Wind Microgrid", "Further Investigation"]
    predicted_solutions = [solutions[label] for label in predicted_labels]
    
    new_data['Electrification_Solution'] = predicted_solutions
    return new_data[['Cluster', 'Electrification_Solution']]

# Process the uploaded file
if uploaded_file is not None:
    # Load the uploaded file
    new_data = pd.read_csv(uploaded_file)

    # Show the uploaded data
    st.write("Uploaded Data Preview:")
    st.write(new_data.head())

    # Ensure required columns are present in the uploaded data
    required_columns = ['Latitude', 'Longitude', 'Pop_Density_Mean', 'Pop_Density_Max', 'Pop_Density_Trend', 'Wind_Speed', 'Grid_Value']
    if all(col in new_data.columns for col in required_columns):
        # Make predictions
        predictions = predict_electrification_solution(new_data)
        
        # Display the predictions
        st.write("Electrification Solution Predictions:")
        st.write(predictions)
        
        # Provide a download link for the results
        csv = predictions.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name="electrification_predictions.csv",
            mime="text/csv",
        )
    else:
        st.write(f"Please make sure your file includes the following columns: {', '.join(required_columns)}")
