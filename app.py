import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_employee_attrition.csv")  # Update with correct path
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    return df

df = load_data()

# Title
st.title("📊 Employee Attrition Analysis")

# Show Dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# Feature Importance Visualization
if st.button("Show Feature Importance"):
    # Convert categorical target variable if needed
    if df['attrition'].dtype == 'object':  
        df['attrition'] = df['attrition'].map({'Yes': 1, 'No': 0})
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Calculate correlation with Attrition
    feature_correlation = df.corr()['attrition'].dropna().sort_values(ascending=False)
    top_features = pd.concat([feature_correlation.head(5), feature_correlation.tail(5)])

    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm", ax=ax)
    plt.xlabel("Correlation with Attrition")
    plt.ylabel("Top Influential Features")
    plt.title("Top 10 Feature Importance in Employee Attrition")
    plt.grid(axis="x", linestyle="--", alpha=0.6)

    st.pyplot(fig)
