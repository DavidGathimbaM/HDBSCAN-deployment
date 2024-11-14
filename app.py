import streamlit as st
import gdown
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import hdbscan
import matplotlib.pyplot as plt

# Set Google Drive file ID for the CSV file
file_id = '1tA8MgnqH8pHQep6XGUoc1Wp9zy4RRzS0'
download_url = f'https://drive.google.com/uc?id={file_id}'
output_file = 'data.csv'

# Function to download and load data
@st.cache_data  # Cache download and processing for efficiency
def download_and_load_data(url):
    # Download file from Google Drive
    gdown.download(url, output_file, quiet=False)
    # Load CSV data into a DataFrame
    df = pd.read_csv(output_file)
    return df

# Function for data preprocessing and clustering
@st.cache_data
def preprocess_and_cluster_data(df):
    # Preprocess: Select required columns and scale data
    income_columns = [col for col in df.columns if col.startswith('Income_')]
    clustering_data = df[['Pop_Density_2020', 'Wind_Speed', 'Latitude', 'Longitude', 'Grid_Value'] + income_columns]
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)

    # Dimensionality reduction with PCA
    pca = PCA(n_components=2)
    clustering_data_reduced = pca.fit_transform(clustering_data_scaled)
    df['PCA_Component_1'] = clustering_data_reduced[:, 0]
    df['PCA_Component_2'] = clustering_data_reduced[:, 1]

    # HDBSCAN clustering
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=20, metric='manhattan')
    clusters = hdbscan_clusterer.fit_predict(clustering_data_reduced)
    df['Cluster'] = clusters
    df['Stability_Score'] = hdbscan_clusterer.probabilities_

    return df, clusters

# Streamlit app interface
st.title("Electrification Planning and Clustering Insights")
st.write("This app downloads a large dataset from Google Drive, processes it, applies clustering, and visualizes potential electrification options.")

# Step 1: Download and load data
st.write("Loading and processing data...")
try:
    data = download_and_load_data(download_url)
    st.write("Data loaded successfully!")
    
    # Step 2: Preprocess and cluster data
    data, clusters = preprocess_and_cluster_data(data)
    st.write("Data preprocessed and clustered successfully!")
    
    # Display clustering results
    st.write("Clustering Results Preview:")
    st.write(data[['Latitude', 'Longitude', 'Pop_Density_2020', 'Cluster', 'Stability_Score']].head())
    
    # Visualize clusters
    st.write("Cluster Visualization:")
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(data['PCA_Component_1'], data['PCA_Component_2'], c=clusters, cmap='viridis', s=5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('HDBSCAN Clustering Results')
    st.pyplot(fig)

    # Insights based on clustering
    st.write("**Insights:**")
    st.write("""
    - This analysis identifies clusters of regions based on population density, wind speed, and grid values.
    - Clusters with low population density and suitable wind conditions may be viable candidates for wind microgrids.
    - Regions with high population density close to existing grids could be more efficiently electrified through grid extensions.
    - Areas labeled as noise (-1) are unique or isolated regions that may require special considerations or further study.
    """)

except Exception as e:
    st.write("Error loading or processing data:", e)
