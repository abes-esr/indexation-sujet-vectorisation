import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("INDEXATION_SUJET_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("INDEXATION_SUJET_QDRANT_PORT", "6333"))
QDRANT_TEST_LIMIT = int(os.getenv("INDEXATION_SUJET_QDRANT_TEST_LIMIT", "6"))
QDRANT_MIN_VECTORS = int(os.getenv("INDEXATION_SUJET_QDRANT_MIN_VECTORS", "1000"))

# Chemins standardisés dans le conteneur (ne dépendent plus de variables d'environnement)
APP_DATA_DIR = "/app/data"
CSV_DIR = "/app/data/csv"
PKL_DIR = "/app/data/pkl"

DOCKER_VOLUME_BIND = os.getenv("INDEXATION_SUJET_DOCKER_VOLUME_BIND", "./volumes")
DOCKER_NETWORK = os.getenv("INDEXATION_SUJET_DOCKER_NETWORK", "mon_reseau")
DOCKER_IMAGE_BATCH = os.getenv("INDEXATION_SUJET_DOCKER_IMAGE_BATCH", "rameau_vectorize_batch:latest")
DOCKER_SOCK = os.getenv("INDEXATION_SUJET_DOCKER_SOCK", "unix:///var/run/docker.sock")
DOCKER_BATCH_USER = os.getenv("INDEXATION_SUJET_DOCKER_BATCH_USER", "root")

# Configuration des fichiers CSV d'ingestion RAMEAU
CSV_INIT_FILENAME = os.getenv("INDEXATION_SUJET_CSV_INIT_FILENAME", "export_rameau.csv")
CSV_UPDATE_FILENAME = os.getenv("INDEXATION_SUJET_CSV_UPDATE_FILENAME", "export_rameau_update.csv")

# Paramètres par défaut pour les pipelines de vectorisation (/init et /update)
VECTORIZE_CONCEPTS_OR_CHAINS = os.getenv("INDEXATION_SUJET_VECTORIZE_CONCEPTS_OR_CHAINS", "concepts")
VECTORIZE_AVEC_THESE = os.getenv("INDEXATION_SUJET_VECTORIZE_AVEC_THESE", "only_mono")
MODELS = ["allMin", "distiluse", "e5-large"]

# Accélération GPU pour les conteneurs batch (désactivée par défaut pour compatibilité locale)
ENABLE_GPU = os.getenv("ENABLE_GPU", "false").lower() in ("true", "1", "yes")

API_HOST = os.getenv("INDEXATION_SUJET_VECTORISATION_HOST", "0.0.0.0")
API_PORT = int(os.getenv("INDEXATION_SUJET_VECTORISATION_PORT", "6381"))
