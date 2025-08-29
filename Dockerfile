# Utiliser une image Python légère
FROM python:3.13-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements
COPY requirements.txt .
COPY app.py .
COPY models.py .
COPY sentiment_analyzer.py .
COPY static ./static


# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu


ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface && chmod -R 777 /app/.cache/huggingface

# Création du répertoire de cache
RUN mkdir -p /app/cache

# Exposer le port souhaité
EXPOSE 8000

# Commande pour démarrer l'API avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
