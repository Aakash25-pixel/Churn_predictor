import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Load data
print("Loading dataset...")
df = pd.read_csv(r"C:\Users\gurja\OneDrive\Desktop\Ml Start\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Target encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop customer ID
X = df.drop(columns=['customerID', 'Churn'])
y = df['Churn']

# Identify features
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [col for col in X.columns if col not in numerical_features]

# Train Test Split (guarantees zero leakage)
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build preprocessing steps
numeric_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Build complete pipeline with SMOTE
# IMPORTANT: Using imblearn.pipeline.Pipeline so SMOTE only applies to training data
pipeline = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Train model
print("Training model with SMOTE...")
pipeline.fit(X_train, y_train)

# Evaluate
print("Evaluating model...")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
model_path = r"C:\Users\gurja\OneDrive\Desktop\Ml Start\ChurnApp\model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(pipeline, model_path)
print(f"Model saved to {model_path}")
