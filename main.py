import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load dataset
df = pd.read_csv('creditcard.csv')

# Step 2: Data Preprocessing
df = df.drop(['Time'], axis=1)
X = df.drop('Class', axis=1)
y = df['Class']

# Step 3: Handling Class Imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto')
X_res, y_res = smote.fit_resample(X, y)

# Step 4: Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Step 5: Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Step 8: Save the model for later use
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
