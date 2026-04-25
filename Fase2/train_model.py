import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# 1. Read the dataset (CSV file)
print("Loading data...")
df = pd.read_csv('Fase2/dataset_phase2.csv')

X = df[['Variance_C3', 'Variance_C4']]  # Features (numbers)
y = df['Class']                         # Labels ('Rest' or 'Passthought')

# 2. Scale the data (CRITICAL for SVM optimization)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Train the AI (Support Vector Machine)
print("Training the Artificial Intelligence...")
svm_model = SVC(kernel='linear')
svm_model.fit(X_scaled, y)

# 4. Quick evaluation to check learning accuracy
predictions = svm_model.predict(X_scaled)
accuracy = accuracy_score(y, predictions) * 100
print(f"🎯 Exam passed with: {accuracy:.2f}% accuracy")

# 5. Save both the model and the scaler
model_data = {
    'model': svm_model,
    'scaler': scaler
}

with open('Fase2/passthought_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("✅ Brain successfully saved as 'passthought_model.pkl'!")