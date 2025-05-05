import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the saved model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Step 2: Load the test dataset (e.g., a separate dataset for testing)
df = pd.read_csv('test_data.csv')

# Step 3: Preprocess the test data
df = df.drop(['Time'], axis=1)
X_test = df.drop('Class', axis=1)
y_test = df['Class']

# Step 4: Standardize the test data
X_test = scaler.transform(X_test)

# Step 5: Make predictions with the loaded model
y_pred = model.predict(X_test)

# Step 6: Evaluate the model on the test data
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
