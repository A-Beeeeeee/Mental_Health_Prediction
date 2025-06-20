import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('survey.csv')

def preprocess_data(df):
    df_processed = df.copy()

    # Drop irrelevant columns
    df_processed= df_processed.drop(columns=['Timestamp', 'comments'])

    # Handle missing values
    df_processed['state'] = df_processed['state'].fillna('Unknown')
    df_processed['self_employed'] = df_processed['self_employed'].fillna('No')
    df_processed['work_interfere'] = df_processed['work_interfere'].fillna('Never')

    df_processed = df_processed[(df_processed['Age'] >= 0) & (df_processed['Age'] <= 100)]
    df_processed['Gender'] = df_processed['Gender'].str.strip()  # Remove any leading or trailing spaces

    # Standardize entries
    gender_mapping = {
        'M': 'Male',
        'male': 'Male',
        'Male-ish': 'Male',
        'maile': 'Male',
        'Trans-female': 'Trans',
        'Cis Female': 'Female',
        'F': 'Female',
        'something kinda male?': 'Other',
        'Cis Male': 'Male',
        'Woman': 'Female',
        'f': 'Female',
        'Mal': 'Male',
        'Male (CIS)': 'Male',
        'queer/she/they': 'Non-binary',
        'non-binary': 'Non-binary',
        'Femake': 'Female',
        'woman': 'Female',
        'Make': 'Male',
        'Nah': 'Other',
        'All': 'Other',
        'Enby': 'Other',
        'fluid': 'Other',
        'Genderqueer': 'Non-binary',
        'Female ': 'Female',
        'Androgyne': 'Other',
        'Agender': 'Non-binary',
        'cis-female/femme': 'Female',
        'Guy (-ish) ^_^': 'Other',
        'male leaning androgynous': 'Male',
        'Male ': 'Male',
        'Man': 'Male',
        'Trans woman': 'Trans',
        'msle': 'Male',
        'Neuter': 'Other',
        'Female (trans)': 'Trans',
        'queer': 'Non-binary',
        'Female (cis)': 'Female',
        'Mail': 'Male',
        'cis male': 'Male',
        'A little about you': 'Other',
        'Malr': 'Male',
        'p': 'Other',
        'femail': 'Female',
        'Cis Man': 'Male',
        'ostensibly male, unsure what that really means': 'Other',
        'female' : 'Female',
        'm' : 'Male',
    }

    # Apply the mapping
    df_processed['Gender'] = df_processed['Gender'].map(gender_mapping).fillna(df_processed['Gender'])

    df_processed = df_processed[df_processed['Gender'] != 'Other']

    # Standardize categorical variables
    df_processed['Gender'] = df_processed['Gender'].str.lower()
    
    features = ['Age', 'Gender', 'Country', 'self_employed', 'family_history',
                'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                'benefits', 'care_options', 'wellness_program', 'seek_help',
                'anonymity', 'leave', 'mental_health_consequence',
                'phys_health_consequence', 'coworkers', 'supervisor',
                'mental_health_interview', 'phys_health_interview',
                'mental_vs_physical', 'obs_consequence']
    
    categorical_columns = ['Gender', 'Country', 'self_employed', 'family_history',
                         'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                         'benefits', 'care_options', 'wellness_program', 'seek_help',
                         'anonymity', 'leave', 'mental_health_consequence',
                         'phys_health_consequence', 'coworkers', 'supervisor',
                         'mental_health_interview', 'phys_health_interview',
                         'mental_vs_physical', 'obs_consequence']

    # One-hot encode categorical variables
    df_features = pd.get_dummies(df_processed[features], drop_first=True)
    
    # Scale Age
    scaler = StandardScaler()
    df_processed['Age'] = scaler.fit_transform(df_processed[['Age']])
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(df_processed['treatment'])
    
    return df_features, y_encoded, scaler, le_target

# Prepare data
X, y, scaler, le_target = preprocess_data(df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes in training data
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Define models and their hyperparameters for tuning
models = {
    'Logistic Regression': (LogisticRegression(), {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear']
    }),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {
        'metric': ['manhattan', 'euclidean'],
        'n_neighbors': [1, 2, 3, 5, 10, 15]
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'criterion': ['entropy', 'gini'],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10]
    }),
    'Random Forest': (RandomForestClassifier(), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10]
    }),
    'Naive Bayes': (GaussianNB(), {
        #No parameters to tune for GaussianNB 
    }),
    'SVM': (SVC(probability=True), {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }),
    'XGBoost': (XGBClassifier(use_label_encoder=False, eval_metric='logloss'), {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0]
    }),
    'AdaBoost': (AdaBoostClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.5, 1.0, 1.5]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [1, 3, 5, 7],
        'min_samples_split': [2, 5, 10]
    })
}

best_model_name = None
best_accuracy = 0
best_report = None

# Train and tune models
for model_name, (model, params) in models.items():
    print(f"Training {model_name}...")

    if params:
        search = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=20, scoring='accuracy', cv=3, n_jobs=-1, random_state=42)
    else:
        # If no params to search (e.g., Naive Bayes), use the default model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        search = None

    if search:
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        y_pred = best_estimator.predict(X_test)
    else:
        best_estimator = model
        
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"\n=== {model_name} Evaluation ===")
    # print(f"Best Parameters: {grid_search.best_params_}")
    if search:
        print(f"Best Parameters: {search.best_params_}")
    print(f"Accuracy Score: {accuracy:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_report = report
        best_model = best_estimator 

print("\n=== Best Model Summary ===")
print(f"Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print("Detailed Classification Report (dict):")
print(best_report)

# Save best model and preprocessing objects
print("\nSaving best model and preprocessing objects...")
with open(f'{best_model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('le_target.pkl', 'wb') as f:
    pickle.dump(le_target, f)
print("Saved successfully!")

# Example prediction function
def predict_mental_health(input_data):

    with open(f'{best_model_name.lower().replace(" ", "_")}_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_target.pkl', 'rb') as f:
        le_target = pickle.load(f)
    
    input_df = pd.DataFrame([input_data])
    categorical_columns = ['Gender', 'Country', 'self_employed', 'family_history',
                           'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                           'benefits', 'care_options', 'wellness_program', 'seek_help',
                           'anonymity', 'leave', 'mental_health_consequence',
                           'phys_health_consequence', 'coworkers', 'supervisor',
                           'mental_health_interview', 'phys_health_interview',
                           'mental_vs_physical', 'obs_consequence']

    gender_mapping = {
        'M': 'Male', 'male': 'Male', 'Male-ish': 'Male', 'maile': 'Male',
        'Trans-female': 'Trans', 'Cis Female': 'Female', 'F': 'Female',
        'something kinda male?': 'Other', 'Cis Male': 'Male', 'Woman': 'Female',
        'f': 'Female', 'Mal': 'Male', 'Male (CIS)': 'Male', 'queer/she/they': 'Non-binary',
        'non-binary': 'Non-binary', 'Femake': 'Female', 'woman': 'Female',
        'Make': 'Male', 'Nah': 'Other', 'All': 'Other', 'Enby': 'Other', 'fluid': 'Other',
        'Genderqueer': 'Non-binary', 'Female ': 'Female', 'Androgyne': 'Other',
        'Agender': 'Non-binary', 'cis-female/femme': 'Female', 'Guy (-ish) ^_^': 'Other',
        'male leaning androgynous': 'Male', 'Male ': 'Male', 'Man': 'Male',
        'Trans woman': 'Trans', 'msle': 'Male', 'Neuter': 'Other',
        'Female (trans)': 'Trans', 'queer': 'Non-binary', 'Female (cis)': 'Female',
        'Mail': 'Male', 'cis male': 'Male', 'A little about you': 'Other',
        'Malr': 'Male', 'p': 'Other', 'femail': 'Female', 'Cis Man': 'Male',
        'ostensibly male, unsure what that really means': 'Other',
        'female': 'Female', 'm': 'Male'
    }

    input_df['Gender'] = input_df['Gender'].map(gender_mapping).fillna(input_df['Gender'])
    input_df = input_df[input_df['Gender'] != 'Other']
    input_df['Gender'] = input_df['Gender'].str.lower()
    input_df['Gender'] = input_df['Gender'].str.strip()

    input_df = input_df[(input_df['Age'] >= 0) & (input_df['Age'] <= 100)]

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Load training feature columns
    model_columns = model.feature_names_in_

    # Align input to training feature columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Scale Age
    input_encoded['Age'] = scaler.transform(input_encoded[['Age']])
  
    # Predict
    prediction = model.predict(input_encoded)
    probability = model.predict_proba(input_encoded)
    
    # Decode prediction to original label
    prediction_label = le_target.inverse_transform(prediction)
    
    return {
         'prediction': prediction_label[0],
         'probability': probability[0].max()
    }

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