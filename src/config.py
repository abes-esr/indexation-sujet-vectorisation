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

# Configuration des fichiers CSV d'ingestion RAMEAU
CSV_INIT_FILENAME = os.getenv("INDEXATION_SUJET_CSV_INIT_FILENAME", "export_rameau.csv")
CSV_UPDATE_FILENAME = os.getenv("INDEXATION_SUJET_CSV_UPDATE_FILENAME", "export_rameau_update.csv")

# Paramètres par défaut pour les pipelines de vectorisation (/init et /update)
VECTORIZE_CONCEPTS_OR_CHAINS = [
    v.strip() for v in (os.getenv("INDEXATION_SUJET_VECTORIZE_CONCEPTS_OR_CHAINS") or "concepts,chains").split(",") if v.strip()
]
VECTORIZE_LANGS = [
    v.strip() for v in (os.getenv("INDEXATION_SUJET_VECTORIZE_LANGS") or "fr,all").split(",") if v.strip()
]
VECTORIZE_AVEC_THESE = os.getenv("INDEXATION_SUJET_VECTORIZE_AVEC_THESE", "only_mono")
MODELS = ["allMin", "distiluse", "e5-large"]

# Exécution des sous-processus de vectorisation (la mémoire et les CPU globaux sont bornés par le conteneur)
# Nombre maximal de sous-processus exécutés simultanément (les autres attendent dans une file)
VECTORIZE_MAX_PARALLEL = max(1, int(os.getenv("INDEXATION_SUJET_VECTORIZE_MAX_PARALLEL") or "2"))
# Threads de calcul (torch/OMP/MKL) par sous-processus. 0 = CPU du conteneur / VECTORIZE_MAX_PARALLEL
VECTORIZE_THREADS_PER_JOB = max(0, int(os.getenv("INDEXATION_SUJET_VECTORIZE_THREADS_PER_JOB") or "0"))

API_HOST = os.getenv("INDEXATION_SUJET_VECTORISATION_HOST", "0.0.0.0")
API_PORT = int(os.getenv("INDEXATION_SUJET_VECTORISATION_PORT", "6381"))
