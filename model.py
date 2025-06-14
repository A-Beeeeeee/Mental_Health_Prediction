import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
df = pd.read_csv('survey.csv')

# Data Preprocessing
def preprocess_data(df):
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['self_employed'] = df_processed['self_employed'].fillna(df_processed['self_employed'].mode()[0])
    df_processed['work_interfere'] = df_processed['work_interfere'].fillna(df_processed['work_interfere'].mode()[0])
    df_processed['state'] = df_processed['state'].fillna('Unknown')
    df_processed['comments'] = df_processed['comments'].fillna('Nil')
    
    # Select features for model
    features = ['Age', 'Gender', 'Country', 'self_employed', 'family_history',
                'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                'benefits', 'care_options', 'wellness_program', 'seek_help',
                'anonymity', 'leave', 'mental_health_consequence',
                'phys_health_consequence', 'coworkers', 'supervisor',
                'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence']
    
    # Create and fit label encoders for each categorical column
    label_encoders = {}
    categorical_columns = ['Gender', 'Country', 'self_employed', 'family_history',
                         'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                         'benefits', 'care_options', 'wellness_program', 'seek_help',
                         'anonymity', 'leave', 'mental_health_consequence',
                         'phys_health_consequence', 'coworkers', 'supervisor',
                         'mental_health_interview', 'phys_health_interview',
                         'mental_vs_physical', 'obs_consequence']
    
    for column in categorical_columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column].astype(str))
        label_encoders[column] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    df_processed['Age'] = scaler.fit_transform(df_processed[['Age']])
    
    return df_processed[features], df_processed['treatment'], label_encoders, scaler

# Prepare data
X, y, label_encoders, scaler = preprocess_data(df)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Model Evaluation
print("\n=== Model Evaluation ===")
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance)

# Save the model and preprocessing objects
print("\nSaving model and preprocessing objects...")
joblib.dump(rf_model, 'mental_health_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

print("\nModel and preprocessing objects saved successfully!")

# Create a simple prediction function
def predict_mental_health(input_data):
    """
    Make predictions using the trained model
    input_data: Dictionary containing the required features
    """
    # Load the saved model and preprocessing objects
    model = joblib.load('mental_health_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoders = joblib.load('label_encoders.joblib')
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Preprocess the input data
    categorical_columns = ['Gender', 'Country', 'self_employed', 'family_history',
                         'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                         'benefits', 'care_options', 'wellness_program', 'seek_help',
                         'anonymity', 'leave', 'mental_health_consequence',
                         'phys_health_consequence', 'coworkers', 'supervisor',
                         'mental_health_interview', 'phys_health_interview',
                         'mental_vs_physical', 'obs_consequence']
    
    # Handle categorical variables
    for column in categorical_columns:
        # Get unique values from training data
        known_categories = label_encoders[column].classes_
        # Replace unknown categories with the most common category
        input_df[column] = input_df[column].apply(
            lambda x: x if x in known_categories else known_categories[0]
        )
        # Transform using the label encoder
        input_df[column] = label_encoders[column].transform(input_df[column])
    
    # Scale numerical features
    input_df['Age'] = scaler.transform(input_df[['Age']])
    
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    return {
        'prediction': prediction[0],
        'probability': probability[0].max()
    }

# Example usage
if __name__ == "__main__":
    # Example input data
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