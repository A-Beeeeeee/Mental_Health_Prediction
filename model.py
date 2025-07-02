import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import KNNImputer
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('survey.csv')

def clean_gender(gender):
    gender = str(gender).strip().lower()
    if gender in ['male', 'm', 'male-ish', 'maile', 'mal', 'cis male', 'man', 'msle', 'mail', 'make', 'malr', 'cis man']:
        return 'Male'
    elif gender in ['female', 'f', 'cis female', 'woman', 'femake', 'female (cis)', 'femail', 'cis-female/femme', 'female ', 'femail']:
        return 'Female'
    else:
        return 'Other'
    

def preprocess_data(df):
    df_processed = df.copy()
    df_processed = df_processed[pd.to_numeric(df_processed['Age'], errors='coerce').notnull()]
    df_processed['Age'] = df_processed['Age'].astype(float)
    median_age = df_processed[(df_processed['Age'] >= 15) & (df_processed['Age'] <= 70)]['Age'].median()
    df_processed.loc[df_processed['Age'] < 15, 'Age'] = median_age
    df_processed.loc[df_processed['Age'] > 70, 'Age'] = median_age
    df_processed['Gender'] = df_processed['Gender'].apply(clean_gender)
    country_counts = df_processed['Country'].value_counts()
    rare_countries = country_counts[country_counts < 20].index
    df_processed['Country'] = df_processed['Country'].apply(lambda x: 'Other' if x in rare_countries else x)

    valid_family_history = ['Yes', 'No']
    valid_work_interfere = ['Never', 'Rarely', 'Sometimes', 'Often']
    valid_treatment = ['Yes', 'No']
    df_processed = df_processed[df_processed['family_history'].isin(valid_family_history)]
    df_processed = df_processed[df_processed['work_interfere'].isin(valid_work_interfere)]
    df_processed = df_processed[df_processed['treatment'].isin(valid_treatment)]

    features = ['Age', 'Gender', 'Country', 'self_employed', 'family_history',
                'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                'benefits', 'care_options', 'wellness_program', 'seek_help',
                'anonymity', 'leave', 'mental_health_consequence',
                'phys_health_consequence', 'coworkers', 'supervisor',
                'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence']

    categorical_columns = [col for col in features if df_processed[col].dtype == 'object' or col == 'Gender']
    for col in categorical_columns:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column].astype(str))
        label_encoders[column] = le

    imputer = KNNImputer(n_neighbors=5)
    df_processed[features] = imputer.fit_transform(df_processed[features])
    scaler = StandardScaler()
    df_processed['Age'] = scaler.fit_transform(df_processed[['Age']])
    df_processed['treatment'] = df_processed['treatment'].astype(str)

    df_majority = df_processed[df_processed['treatment'] == df_processed['treatment'].mode()[0]]
    df_minority = df_processed[df_processed['treatment'] != df_processed['treatment'].mode()[0]]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X = df_balanced[features]
    le_target = LabelEncoder()
    y = le_target.fit_transform(df_balanced['treatment'])
    return X, y, label_encoders, scaler, le_target


# Prepare data
X, y, label_encoders, scaler, le_target = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
all_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}


best_model = None
best_score = 0
best_model_name = ''

print("\n=== Training and Evaluation of All Models ===")
for name, model in all_models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if acc > best_score:
        best_score = acc
        best_model = model
        best_model_name = name

print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")
print("\n=== Final Model Evaluation ===")
y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if hasattr(best_model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:\n", importances)


# Save using pickle
print("\nSaving model and preprocessing objects with pickle...")
with open('mental_health_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
print("Saved successfully!")


# Prediction function
def predict_mental_health(input_data):
    with open('mental_health_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    input_df = pd.DataFrame([input_data])
    categorical_columns = ['Gender', 'Country', 'self_employed', 'family_history',
                           'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                           'benefits', 'care_options', 'wellness_program', 'seek_help',
                           'anonymity', 'leave', 'mental_health_consequence',
                           'phys_health_consequence', 'coworkers', 'supervisor',
                           'mental_health_interview', 'phys_health_interview',
                           'mental_vs_physical', 'obs_consequence']

    for column in categorical_columns:
        known_categories = label_encoders[column].classes_
        input_df[column] = input_df[column].apply(lambda x: x if x in known_categories else known_categories[0])
        input_df[column] = label_encoders[column].transform(input_df[column])

    input_df['Age'] = scaler.transform(input_df[['Age']])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return {'prediction': prediction[0], 'probability': probability[0].max()}


# Example usage
if __name__ == "__main__":
    example_input = {
        'Age': 30,
        'Gender': 'Male',
        'Country': 'United States',
        'self_employed': 'No',
        'family_history': 'Yes',
        'work_interfere': 'Sometimes',
        'no_employees': '26-100',
        'remote_work': 'No',
        'tech_company': 'Yes',
        'benefits': 'Yes',
        'care_options': 'Yes',
        'wellness_program': 'Yes',
        'seek_help': 'Yes',
        'anonymity': 'Yes',
        'leave': 'Somewhat easy',
        'mental_health_consequence': 'No',
        'phys_health_consequence': 'No',
        'coworkers': 'Yes',
        'supervisor': 'Yes',
        'mental_health_interview': 'No',
        'phys_health_interview': 'No',
        'mental_vs_physical': 'Yes',
        'obs_consequence': 'No'
    }

    result = predict_mental_health(example_input)
    print("\nExample Prediction:")
    print(f"Prediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2f}")