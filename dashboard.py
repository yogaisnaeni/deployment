import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Function for Data Exploration
def data_exploration(df):
    st.subheader("Data Exploration")
    st.write(f"### Dataset Information (Shape: {df.shape})")
    st.write("Columns:", list(df.columns))
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.write("### Missing Values")
        st.write(missing_values[missing_values > 0])
    else:
        st.write("No missing values found.")
    
    st.write("### Descriptive Statistics")
    st.write(df.describe())
    
    numerical_cols = df.select_dtypes(include='number').columns
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            st.write(f"#### Distribution of {col}")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=20, ax=ax)
            plt.xticks(rotation=45, ha="right")  # Rotasi agar lebih rapi
            st.pyplot(fig)
        
        if len(numerical_cols) > 1:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
    else:
        st.write("No numerical columns found.")

# Function for Logistic Regression
def logistic_regression_visualization(df):
    st.subheader("Logistic Regression")
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    if not numerical_cols:
        st.write("No numerical columns available for analysis.")
        return
    
    target = st.selectbox("Select Target Column", df.columns)
    
    X = df[numerical_cols]
    if target in numerical_cols:
        X = X.drop(columns=[target])
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    st.write(f"Model Accuracy: {score:.2f}")
    
    # Bar chart visualization
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=df[target], y=df[col], ax=ax)
        ax.set_title(f"Distribution of {col} by {target}")
        st.pyplot(fig)

# Function for ROC AUC Analysis
def roc_auc_analysis(df):
    st.subheader("ROC AUC Analysis")
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    if not numerical_cols:
        st.write("No numerical columns available for analysis.")
        return
    
    target = st.selectbox("Select Target Column", df.columns)
    
    X = df[numerical_cols]
    if target in numerical_cols:
        X = X.drop(columns=[target])
    y = df[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    st.write(f"ROC AUC Score: {auc:.2f}")
    
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)
    
    # Bar chart visualization
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=df[target], y=df[col], ax=ax)
        ax.set_title(f"Distribution of {col} by {target}")
        st.pyplot(fig)

# Main Function
def main():
    st.title("Data Analysis Dashboard")
    analysis_type = st.sidebar.radio("Choose Analysis", ["Data Exploration", "Logistic Regression", "ROC AUC Analysis"])
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"Dataset shape: {df.shape}")
        st.dataframe(df.head())
        
        if analysis_type == "Data Exploration":
            data_exploration(df)
        elif analysis_type == "Logistic Regression":
            logistic_regression_visualization(df)
        elif analysis_type == "ROC AUC Analysis":
            roc_auc_analysis(df)
    else:
        st.write("Please upload a dataset.")

if __name__ == '__main__':
    main()
