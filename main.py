import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
import joblib

def load_and_preprocess():
    # Load datasets
    train_df = pd.read_csv("Training Data.csv")
    test_df = pd.read_csv("Test Data.csv")
    
    # Preprocessing: mapping categorical features
    mapping_married = {'single': '0', 'married': '1'}
    train_df['married'] = train_df['married'].map(mapping_married)
    test_df['married'] = test_df['married'].map(mapping_married)
    
    mapping_house = {'owned': '2', 'rented': '1', 'norent_noown': '0'}
    train_df['house_ownership'] = train_df['house_ownership'].map(mapping_house)
    test_df['house_ownership'] = test_df['house_ownership'].map(mapping_house)
    
    train_df['car_ownership'] = train_df['car_ownership'].replace({'no': '0', 'yes': '1'})
    test_df['car_ownership'] = test_df['car_ownership'].replace({'no': '0', 'yes': '1'})
    
    # One-hot encoding for 'profession' and 'city' (only keep top 30 values)
    top_30_prof_train = train_df['profession'].value_counts().head(30).index.tolist()
    for label in top_30_prof_train:
        train_df[label] = np.where(train_df['profession'] == label, 1, 0)
    
    top_30_prof_test = test_df['profession'].value_counts().head(30).index.tolist()
    for label in top_30_prof_test:
        test_df[label] = np.where(test_df['profession'] == label, 1, 0)
    
    top_30_city_train = train_df['city'].value_counts().head(30).index.tolist()
    for label in top_30_city_train:
        train_df[label] = np.where(train_df['city'] == label, 1, 0)
    
    top_30_city_test = test_df['city'].value_counts().head(30).index.tolist()
    for label in top_30_city_test:
        test_df[label] = np.where(test_df['city'] == label, 1, 0)
    
    # Convert state to numerical codes
    train_df['state'] = train_df['state'].astype('category').cat.codes
    test_df['state'] = test_df['state'].astype('category').cat.codes
    
    # Drop unnecessary columns
    train = train_df.drop(columns=['Id', 'city', 'profession'])
    test = test_df.drop(columns=['id', 'city', 'profession'])
    
    # Split features and target
    target_var = train['approval']  # 0 = approved, 1 = not approved
    train_var = train.drop(['approval'], axis=1)
    
    return train_var, target_var

def train_evaluate_random_forest(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Trains a cost-sensitive Random Forest model with adjusted parameters
    so that class 1 (not approved) is given higher weight to reduce false negatives.
    """
    # Key parameters:
    # - n_estimators: # of trees
    # - max_depth: limit to avoid overfitting
    # - min_samples_leaf: stable splits
    # - class_weight: penalize missing class 1 more (since 1=not approved)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight={0: 1, 1: 2},  # Heavier penalty for misclassifying class 1
        random_state=0
    )
    rf_model.fit(X_train_scaled, y_train)
    
    preds = rf_model.predict(X_test_scaled)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"Random Forest F1 Score (weighted): {f1:.4f}")
    
    print("Classification Report:")
    print(classification_report(y_test, preds))
    
    return rf_model

# Main workflow
train_var, target_var = load_and_preprocess()

# Resample data using SMOTE to balance classes
smote = SMOTE()
x_resample, y_resample = smote.fit_resample(train_var, target_var.values.ravel())

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(
    x_resample, y_resample, test_size=0.3, random_state=0
)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Number of features expected:", scaler.n_features_in_)

# Train & evaluate the Random Forest model
best_model = train_evaluate_random_forest(X_train_scaled, y_train, X_test_scaled, y_test)

# Ensure the model folder exists
if not os.path.exists('model'):
    os.makedirs('model')

# Save the trained model and scaler
joblib.dump(best_model, 'model/rf_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')

print("Random Forest model (cost-sensitive) and scaler saved successfully in the 'model' folder.")
