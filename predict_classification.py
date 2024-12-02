from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, classification_report
import joblib

# Charger le modèle et les données
model = joblib.load('model.joblib')
X_val, y_val = joblib.load('validation_data.joblib')

# Prédire avec le modèle et évaluer
y_pred = model.predict(X_val)

precision_weighted = precision_score(y_val, y_pred, average='weighted')
recall_weighted = recall_score(y_val, y_pred, average='weighted')
f1_weighted = f1_score(y_val, y_pred, average='weighted')

print(f"Précision (Pondérée): {precision_weighted}")
print(f"Rappel (Pondéré): {recall_weighted}")
print(f"F1-Score (Pondéré): {f1_weighted}")

# Afficher le rapport de classification
class_report = classification_report(y_val, y_pred)
print("Classification Report:")
print(class_report)