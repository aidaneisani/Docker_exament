from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

# Génération des données avec make_classification
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=3,  # Augmenté à 3
    n_classes=3,
    n_clusters_per_class=2,
    class_sep=2,
    random_state=42
)

# Réduction de la dimension à 2 composantes
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# Division initiale : 70% entraînement et 30% temporaire (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Division secondaire : 20% test et 10% validation
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.333, random_state=42)

# créer le modèle d'arbre de décision
model = DecisionTreeClassifier(random_state=42)

# entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

# faire des prédictions sur les données de test
y_test_pred = model.predict(X_test)

# Faire des prédictions sur les données de test
y_test_pred = model.predict(X_test)

# Calculer la précision
accuracy = accuracy_score(y_test, y_test_pred)
print(f"Accuracy: {accuracy}")

# Calculer le F1-score (utile si vous avez des classes déséquilibrées)
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f"F1-score (weighted): {f1}")

# Afficher le rapport de classification
class_report = classification_report(y_test, y_test_pred)
print("Classification Report:")
print(class_report)

joblib.dump(model, 'model.joblib')

# Exporter le modèle
joblib.dump(model, 'model.joblib')

# Exporter les données de validation
joblib.dump((X_val, y_val), 'validation_data.joblib')