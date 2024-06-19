import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = pd.read_csv("D:/diabetes_prediction/diabetes.csv")
X = data.drop("Outcome", axis=1)
y = data["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = []
models.append(("Logistic Regression", LogisticRegression()))
models.append(("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)))
models.append(("Random Forest", RandomForestClassifier(n_estimators=100)))
models.append(("SVM", SVC()))
best_model = None
best_accuracy = 0
for name, model in models:
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
print("\n\nBest Model:")
print(f"Name: {best_model.__class__.__name__}")
print(f"Accuracy: {best_accuracy}")

import pickle
pickle.dump(best_model,open('D:/diabetes_prediction/best_model.pkl','wb'))

pickle.dump(scaler,open('D:/diabetes_prediction/sc.pkl','wb'))