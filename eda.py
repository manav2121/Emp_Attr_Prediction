import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure visuals directory exists
os.makedirs("visuals", exist_ok=True)

# Load dataset
df = pd.read_csv("data/cleaned_employee_attrition.csv")

# Standardize column names
df.columns = df.columns.str.strip().str.lower()

# Attrition count plot
plt.figure(figsize=(6, 4))
sns.countplot(x='attrition', hue='attrition', data=df, palette='coolwarm', legend=False)
plt.title("Attrition Count")
plt.savefig("visuals/attrition_count.png")
plt.close()  # Close plot

# Salary vs Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(x='attrition', y='monthlyincome', data=df, palette='coolwarm')
plt.title("Salary vs Attrition")
plt.savefig("visuals/salary_vs_attrition.png")
plt.close()

# Department-wise attrition analysis
plt.figure(figsize=(10, 5))
sns.countplot(x='department', hue='attrition', data=df, palette='coolwarm')
plt.xticks(rotation=45)
plt.title("Attrition by Department")
plt.savefig("visuals/attrition_by_department.png")
plt.close()

# Work-life balance impact
plt.figure(figsize=(8, 5))
sns.boxplot(x='attrition', y='worklifebalance', data=df, palette='coolwarm')
plt.title("Work-Life Balance vs Attrition")
plt.savefig("visuals/work_life_balance.png")
plt.close()

print("✅ EDA complete! Check the 'visuals' folder for charts.")
