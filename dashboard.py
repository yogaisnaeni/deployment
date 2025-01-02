import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, label_binarize

# Function for Data Exploration
def data_exploration(df):
    st.subheader("Data Exploration")
    
    # Display general information
    st.write("### Dataset Information")
    st.write(f"Shape of dataset: {df.shape}")
    st.write("Columns:", list(df.columns))

    # Show missing values
    st.write("### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values[missing_values > 0])

    # Display Descriptive Statistics
    st.write("### Descriptive Statistics")
    st.write(df.describe())

    # Visualization for Numerical Columns
    numerical_cols = df.select_dtypes(include='number').columns
    if len(numerical_cols) > 0:
        st.write("### Numerical Data Distributions")
        for col in numerical_cols:
            st.write(f"#### Distribution of {col}")
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=20, color="blue")
            st.pyplot(plt)
    else:
        st.write("No numerical columns found.")

    # Correlation Heatmap
    if len(numerical_cols) > 1:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        st.pyplot(plt)
    else:
        st.write("Not enough numerical columns to calculate correlations.")

# Function for K-Means Clustering Visualization
def kmeans_visualization(df):
    st.subheader("K-Means Clustering")
    numerical_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    # PCA for Dimensionality Reduction
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    # Map Risk Levels
    risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    df['Risk Level'] = df['Cluster'].map(risk_mapping)

    # Visualization
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['Risk Level'], palette={'Low Risk': 'green', 'Medium Risk': 'cyan', 'High Risk': 'red'})
    plt.title('K-Means Clustering with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    st.pyplot(plt)

# Main Function
def main():
    st.title("Data Analysis Dashboard")
    st.sidebar.title("Menu")
    analysis_type = st.sidebar.radio("Choose Analysis", ["Data Exploration", "K-Means Clustering", "Logistic Regression", "ROC AUC Analysis"])
    st.write(f"Selected Analysis: {analysis_type}")

    # Upload Dataset
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())  # Tampilkan preview dataset
        st.write(f"Dataset shape: {df.shape}")  # Menampilkan ukuran dataset
        
        # Execute Analysis
        if analysis_type == "Data Exploration":
            data_exploration(df)
        elif analysis_type == "K-Means Clustering":
            kmeans_visualization(df)
        elif analysis_type == "Logistic Regression":
            logistic_regression_visualization(df)
        elif analysis_type == "ROC AUC Analysis":
            roc_auc_visualization(df)
    else:
        st.write("Please upload a dataset.")

if __name__ == '__main__':
    main()
