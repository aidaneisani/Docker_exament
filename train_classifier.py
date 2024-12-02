from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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


# Afficher les dimensions de X et y
print(X.shape, y.shape)