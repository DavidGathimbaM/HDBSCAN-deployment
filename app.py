import streamlit as st
import pandas as pd
import gdown
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load Models
hdbscan_model = joblib.load(open("hdbscan_model.pkl", "rb"))
pca_model = joblib.load(open("pca.pkl", "rb"))
mlp_model = tf.keras.models.load_model("mlp_electrification_classifier.h5")

# Retrieve Dataset from Google Drive
file_id = '1tA8MgnqH8pHQep6XGUoc1Wp9zy4RRzS0'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'dataset.csv'
gdown.download(url, output, quiet=False)
df = pd.read_csv(output)

# Prepare the Dropdown
county_options = [col.replace('Income_', '') for col in df.columns if 'Income_' in col]
county = st.selectbox("Select County", options=county_options)

# One-Hot Encode Selection
county_encoded = pd.DataFrame([[1 if f"Income_{county}" == col else 0 for col in df.columns if 'Income' in col]],
                              columns=[col for col in df.columns if 'Income_' in col])

# Filter Dataset by Selected County and Prepare for Prediction
county_data = df[(county_encoded.columns & df.columns).any(1)]

# PCA Visualization
st.subheader("PCA and HDBSCAN Clustering")
pca_result = pca_model.transform(county_data[['Latitude', 'Longitude', 'Wind_Speed', 'Pop_Density_2020']])
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=hdbscan_model.labels_, cmap='viridis')
st.pyplot(plt)

# MLP Prediction
st.subheader("Electricity Power Viability")
input_data = county_data[['Latitude', 'Longitude', 'Wind_Speed', 'Pop_Density_2020']].join(county_encoded)
prediction = mlp_model.predict(input_data)
if prediction[0] > 0.5:  # Adjust threshold as needed
    st.write("This area has electricity power.")
else:
    st.write("This area does not have electricity power. Wind microgrid might be viable.")
