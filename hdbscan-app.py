import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import hdbscan

# Load pre-trained components
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
hdbscan_clusterer = joblib.load('hdbscan_model.pkl')

# Streamlit App title and description
st.title("HDBSCAN Clustering App")
st.write("Upload your dataset, and the app will apply HDBSCAN clustering using pre-trained components.")

# File uploader for the new dataset
uploaded_file = st.file_uploader("Choose a CSV file for clustering", type="csv")

if uploaded_file is not None:
    # Load the uploaded data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Select features for clustering
    income_columns = [col for col in df.columns if col.startswith('Income_')]
    required_columns = ['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value'] + income_columns

    if all(feature in df.columns for feature in required_columns):
        # Extract and preprocess the features
        clustering_data = df[required_columns]
        clustering_data_scaled = scaler.transform(clustering_data)
        clustering_data_reduced = pca.transform(clustering_data_scaled)

        # Apply HDBSCAN clustering
        clusters = hdbscan_clusterer.fit_predict(clustering_data_reduced)
        df['Cluster'] = clusters

        # Display clustering results
        st.write("Clustering Results (with Cluster Labels):")
        st.write(df[['Cluster']].value_counts())

        # Filter out noise points (-1 label) before calculating metrics
        clustered_data = clustering_data_reduced[clusters != -1]
        valid_clusters = clusters[clusters != -1]
        
        # Calculate clustering metrics if there are enough valid clusters
        if len(set(valid_clusters)) > 1:
            db_index = davies_bouldin_score(clustered_data, valid_clusters)
            ch_index = calinski_harabasz_score(clustered_data, valid_clusters)
            st.write("Davies-Bouldin Index:", db_index)
            st.write("Calinski-Harabasz Index:", ch_index)
        else:
            st.write("Insufficient clusters for evaluation metrics (only one or no valid clusters).")

        # Visualization of clusters
        plt.figure(figsize=(10, 6))
        plt.scatter(clustering_data_reduced[:, 0], clustering_data_reduced[:, 1], c=clusters, cmap='viridis', marker='o')
        plt.title("HDBSCAN Clustering (PCA-reduced data)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        st.pyplot(plt)
        
        # Display cluster stability scores
        stability_scores = hdbscan_clusterer.probabilities_
        df['Stability_Score'] = stability_scores
        st.write("Cluster Stability Scores")
        st.write(df[['Cluster', 'Stability_Score']])
    else:
        st.write("The uploaded dataset is missing one or more required columns.")
