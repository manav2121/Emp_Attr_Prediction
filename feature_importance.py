import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/cleaned_employee_attrition.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Convert 'attrition' column to numerical (if it's categorical)
if df['attrition'].dtype == 'object':  
    df['attrition'] = df['attrition'].map({'Yes': 1, 'No': 0})

# Convert categorical features to numeric (one-hot encoding)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Compute correlation with Attrition
feature_correlation = df.corr()['attrition'].dropna().sort_values(ascending=False)

# ✅ Select **top 10 most important features** (positive & negative)
top_features = pd.concat([feature_correlation.head(5), feature_correlation.tail(5)])

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("visuals", exist_ok=True)

# Save feature importance to CSV
feature_correlation.to_csv("results/feature_importance.csv")

# 📊 **Improved Visualization: Clean & Readable Bar Chart**
plt.figure(figsize=(10, 5))
sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm")

plt.xlabel("Correlation with Attrition")
plt.ylabel("Top Influential Features")
plt.title("Top 10 Feature Importance in Employee Attrition")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.xticks(rotation=45)  # Rotate labels for better readability

# Save and show the plot
plt.savefig("visuals/feature_importance.png", bbox_inches="tight")
plt.show()

print("✅ Cleaned feature importance visualization saved at 'visuals/feature_importance.png'.")
