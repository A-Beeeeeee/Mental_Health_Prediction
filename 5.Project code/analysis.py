import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import preprocess_data

# Read and preprocess the dataset
raw_df = pd.read_csv('survey.csv')
df, y, label_encoders, scaler, le_target = preprocess_data(raw_df)
# For analysis, merge y back if needed:
df['treatment'] = y


# 1. Descriptive Statistics
print("\n=== Descriptive Statistics (Preprocessed Data) ===")
print("\nBasic Statistics:")
print(df.describe())

print("\nCategorical Variables Summary:")
print(df.describe(include=['object', 'int', 'float']))


# 2. Univariate Analysis
print("\n=== Univariate Analysis ===")

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', bins=30)
plt.title('Age Distribution')
plt.show()


# Gender Distribution
df = pd.read_csv("survey.csv")
plt.figure(figsize=(10, 6))
df['Gender'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Gender Categories')
plt.xticks(rotation=45)
plt.show()

# Country Distribution
plt.figure(figsize=(12, 6))
df['Country'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Countries')
plt.xticks(rotation=45)
plt.show()

# Family History of Mental Health
plt.figure(figsize=(8, 6))
df['family_history'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Family History of Mental Health')
plt.show()

# Treatment Status
plt.figure(figsize=(8, 6))
df['treatment'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Treatment Status')
plt.show()


# 3. Bivariate Analysis
print("\n=== Bivariate Analysis ===")


# Gender vs Treatment
plt.figure(figsize=(10, 6))
treatment_by_gender = pd.crosstab(df['Gender'], df['treatment'])
treatment_by_gender.plot(kind='bar', stacked=True)
plt.title('Treatment Status by Gender')
plt.xticks(rotation=45)
plt.show()

# Family History vs Treatment
plt.figure(figsize=(8, 6))
sns.heatmap(pd.crosstab(df['family_history'], df['treatment']), annot=True, fmt='d', cmap='YlOrRd')
plt.title('Treatment Status by Family History')
plt.show()

# Work Interference vs Treatment
plt.figure(figsize=(10, 6))
sns.heatmap(pd.crosstab(df['work_interfere'], df['treatment']), annot=True, fmt='d', cmap='YlOrRd')
plt.title('Treatment Status by Work Interference')
plt.show()

# 4. Correlation Analysis
print("\n=== Correlation Analysis ===")
numeric_columns = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()


# 5. Additional Insights
print("\n=== Additional Insights ===")

df = pd.read_csv("survey.csv")
df['treatment'] = df['treatment'].map({'Yes': 1, 'No': 0})
treatment_rate = df.groupby('Country')['treatment'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
treatment_rate.head(10).plot(kind='bar')
plt.title('Treatment Rate by Country (Top 10)')
plt.xticks(rotation=45)
plt.ylabel('Treatment Rate (Proportion)')
plt.tight_layout()
plt.show()

# Work Interference by Treatment Status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='work_interfere', hue='treatment')
plt.title('Work Interference by Treatment Status')
plt.xticks(rotation=45)
plt.show()