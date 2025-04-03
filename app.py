import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Streamlit Title
st.title("📊 Employee Attrition Dashboard")

# Sidebar for Navigation
st.sidebar.header("🔍 Navigation")
options = ["Dataset Overview", "Attrition Analysis", "Feature Importance"]
choice = st.sidebar.radio("📌 Select Analysis", options)

# File Uploader (For Streamlit Cloud)
uploaded_file = st.file_uploader("📂 Upload Employee Attrition Data (CSV)", type=["csv"])

# Load Data Function
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    return df

# Load Dataset (Check Session State)
if "df" not in st.session_state:
    if uploaded_file is not None:
        st.session_state.df = load_data(uploaded_file)
    else:
        st.warning("⚠️ Please upload the dataset to proceed.")
        st.stop()

df = st.session_state.df

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
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=attrition_counts.index, y=attrition_counts.values, palette="coolwarm", ax=ax, edgecolor="black")
        ax.set_xlabel("Attrition", fontsize=12, fontweight="bold")
        ax.set_ylabel("Percentage", fontsize=12, fontweight="bold")
        ax.set_title("Attrition Distribution", fontsize=14, fontweight="bold")

        # Add data labels
        for i, val in enumerate(attrition_counts.values):
            ax.text(i, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
            
        st.pyplot(fig)
    else:
        st.error("⚠️ Column 'Attrition' not found. Please check the dataset.")

# Feature Importance
elif choice == "Feature Importance":
    st.subheader("📊 What Affects Employee Attrition?")

    # Convert categorical columns
    categorical_columns = df.select_dtypes(include=["object"]).columns
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in categorical_columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])

    if "attrition" in df_encoded.columns:
        # Compute correlation with attrition
        correlation = df_encoded.corr()["attrition"].dropna().sort_values(ascending=False)
        top_features = pd.concat([correlation.head(8), correlation.tail(8)]).reset_index()
        top_features.columns = ["Feature", "Correlation"]

        # Define colors: Blue (Positive Effect = More Attrition), Red (Negative Effect = Less Attrition)
        colors = ["#1F77B4" if x > 0 else "#D62728" for x in top_features["Correlation"]]

        # Create professional bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(y=top_features["Feature"], x=top_features["Correlation"], 
                    palette=colors, ax=ax, edgecolor="black", linewidth=1.5)

        # Format annotations in plain English
        for i, value in enumerate(top_features["Correlation"]):
            label = f"{value:.2f}"
            ax.text(value, i, label, ha="left" if value > 0 else "right", 
                    va="center", fontsize=12, fontweight="bold", 
                    color="white" if abs(value) > 0.1 else "black")

        # Labels & Titles with a simple explanation
        ax.set_xlabel("Effect on Employee Attrition", fontsize=14, fontweight="bold", color="#333333")
        ax.set_ylabel("Work & Personal Factors", fontsize=14, fontweight="bold", color="#333333")
        ax.set_title("📌 What Factors Increase or Reduce Attrition?", fontsize=16, fontweight="bold", color="#222222")

        # Enhance layout for simplicity
        plt.axvline(0, linestyle="--", color="gray", alpha=0.7)  # Reference line at zero
        plt.grid(axis="x", linestyle="--", alpha=0.5)  # Light gridlines
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Add a simple explanation
        st.markdown(
            """
            **📖 How to Read the Graph:**
            - **🔵 Blue Bars → Higher Attrition** (Employees are more likely to leave)  
            - **🔴 Red Bars → Lower Attrition** (Employees are more likely to stay)  
            - **Further from Zero = Stronger Impact**
            """
        )

        st.pyplot(fig)
    else:
        st.error("⚠️ Column 'Attrition' not found after encoding. Please verify the dataset.")
