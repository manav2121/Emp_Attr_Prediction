import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/employee_attrition.csv")

# Standardize column names (remove spaces, lowercase)
df.columns = df.columns.str.strip().str.lower()

# Ensure 'attrition' exists
if 'attrition' not in df.columns:
    raise ValueError("Error: 'Attrition' column is missing in the dataset!")

# Drop unnecessary columns
df.drop(columns=['employeecount', 'over18', 'standardhours'], inplace=True, errors='ignore')

# Handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Save cleaned dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/cleaned_employee_attrition.csv", index=False)

print("✅ Data preprocessing complete! Cleaned dataset saved.")
