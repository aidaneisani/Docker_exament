# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY train_classifier.py /app/train_classifier.py
COPY predict_classification.py /app/predict_classification.py

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Lancer les deux scripts dans l'ordre (train_classifier puis predict_classification)
CMD ["sh", "-c", "python train_classifier.py && python predict_classification.py"]