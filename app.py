import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os

# Function to load data
@st.cache_data
def load_data():
    file_path = "data/cleaned_employee_attrition.csv"
    if not os.path.exists(file_path):
        st.error(f"⚠️ File not found: {file_path}. Please check the path and try again.")
        return None
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    return df

# Load dataset
df = load_data()

if df is not None:
    st.title("📊 Employee Attrition Dashboard")

    # Sidebar for navigation
    st.sidebar.header("🔍 Navigation")
    options = ["Dataset Overview", "Attrition Analysis", "Feature Importance"]
    choice = st.sidebar.radio("📌 Select Analysis", options)

    # Dataset Overview
    if choice == "Dataset Overview":
        st.subheader("📂 Dataset Overview")
        with st.expander("👀 Preview Dataset"):
            st.write(df.head())

       

    # Attrition Analysis
    elif choice == "Attrition Analysis":
        st.subheader("📉 Employee Attrition Analysis")

        if "attrition" in df.columns:
            attrition_counts = df["attrition"].value_counts(normalize=True) * 100
            fig, ax = plt.subplots()
            sns.barplot(x=attrition_counts.index, y=attrition_counts.values, palette="coolwarm", ax=ax)
            ax.set_xlabel("Attrition")
            ax.set_ylabel("Percentage")
            ax.set_title("Attrition Distribution")
            for i, val in enumerate(attrition_counts.values):
                ax.text(i, val, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
            st.pyplot(fig)
        else:
            st.error("⚠️ Column 'Attrition' not found. Please check the dataset.")

    # Feature Importance
    elif choice == "Feature Importance":
        st.subheader("📊 Feature Importance in Attrition")

        # Convert categorical columns
        categorical_columns = df.select_dtypes(include=["object"]).columns
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in categorical_columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])

        if "attrition" in df_encoded.columns:
            # Compute correlation with attrition
            correlation = df_encoded.corr()["attrition"].dropna().sort_values(ascending=False)
            top_features = pd.concat([correlation.head(5), correlation.tail(5)])

            # Improved visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(y=top_features.index, x=top_features.values, palette="coolwarm", ax=ax)

            # Labels & Titles
            ax.set_xlabel("Correlation with Attrition", fontsize=12)
            ax.set_ylabel("Feature", fontsize=12)
            ax.set_title("🔝 Top Influential Features", fontsize=14, fontweight="bold")

            # Add value labels to bars
            for i, value in enumerate(top_features.values):
                ax.text(value, i, f"{value:.2f}", ha="left", va="center", fontsize=11, fontweight="bold", color="black")

            # Improve layout
            plt.grid(axis="x", linestyle="--", alpha=0.6)
            plt.xticks(fontsize=11)
            plt.yticks(fontsize=11)

            st.pyplot(fig)
        else:
            st.error("⚠️ Column 'Attrition' not found after encoding. Please verify the dataset.")

else:
    st.stop()
