import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

df['diagnosis'] = data.target

selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 
    'mean area', 'mean smoothness'
]
X = df[selected_features]
y = df['diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1-score: {f1_score(y_test, y_pred)}")

joblib.dump({'model': model, 'scaler': scaler}, 'model/breast_cancer_model.pkl')

loaded_data = joblib.load('model/breast_cancer_model.pkl')
reloaded_model = loaded_data['model']
print("Model reloaded and ready for prediction.")