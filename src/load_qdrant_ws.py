import os
import shutil
import socket
from fastapi import FastAPI, UploadFile, File
import docker
import sys
from pathlib import Path

# Assure la résolution directe de config quel que soit le contexte d'exécution (racine ou src)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

app = FastAPI()

@app.get("/health")
async def health():
    """
    Sonde de disponibilité applicative (Healthcheck).
    Retourne le statut de fonctionnement du service.

    :return: Dictionnaire indiquant le statut opérationnel ("ok").
    """
    return {"status": "ok"}

def run_vectorization_pipeline(action: str) -> dict:
    """
    Lance le pipeline de vectorisation (init ou update) dans des conteneurs Docker batch distincts
    via le socket Docker de l'hôte (DooD - Docker-out-of-Docker), un conteneur par modèle configuré
    (allMin, distiluse, e5-large).
    Les paramètres (typologie, concepts/chaînes, CSV) sont extraits directement
    de la configuration applicative afin de prévenir toute injection de commande.

    :param action: Action à exécuter ('init' ou 'update').
    :return: Dictionnaire contenant le statut et les détails d'exécution de chaque conteneur.
    """
    concepts_or_chains = config.VECTORIZE_CONCEPTS_OR_CHAINS
    avec_these = config.VECTORIZE_AVEC_THESE
    csv_filename = config.CSV_INIT_FILENAME if action == "init" else config.CSV_UPDATE_FILENAME
    models_to_run = getattr(config, "MODELS", ["allMin", "distiluse", "e5-large"])

    print(f"Lancement de la vectorisation conteneurisée (action={action}) pour les modèles: {models_to_run}")

    try:
        client = docker.DockerClient(base_url=config.DOCKER_SOCK)

        # Alignement strict sur la réservation GPU d'indexation-sujet-docker (driver: nvidia, count: all, capabilities: [gpu])
        device_requests = [
            docker.types.DeviceRequest(
                driver="nvidia",
                count=-1,
                capabilities=[["gpu"]],
            )
        ] if config.ENABLE_GPU else None

        # Détection automatique du chemin de volume hôte monté sur /app/data
        volume_bind = None
        if os.path.exists("/.dockerenv"):
            try:
                hostname = socket.gethostname()
                try:
                    current_container = client.containers.get(hostname)
                except Exception:
                    current_container = client.containers.get("indexation-sujet-vectorisation")

                for mount in current_container.attrs.get("Mounts", []):
                    if mount.get("Destination") == "/app/data":
                        volume_bind = mount.get("Source")
                        print(f"Volume hôte détecté depuis le conteneur parent : {volume_bind} -> /app/data")
                        break
            except Exception as vol_err:
                print(f"Impossible de détecter le volume hôte automatiquement : {vol_err}")

        if not volume_bind:
            volume_bind = os.path.abspath(config.DOCKER_VOLUME_BIND)

        # Transmission des variables d'environnement au conteneur batch
        batch_env = {
            "INDEXATION_SUJET_QDRANT_HOST": config.QDRANT_HOST,
            "INDEXATION_SUJET_QDRANT_PORT": str(config.QDRANT_PORT),
            "INDEXATION_SUJET_QDRANT_TEST_LIMIT": str(getattr(config, "QDRANT_TEST_LIMIT", 6)),
            "INDEXATION_SUJET_QDRANT_MIN_VECTORS": str(getattr(config, "QDRANT_MIN_VECTORS", 1000)),
            "INDEXATION_SUJET_CSV_INIT_FILENAME": config.CSV_INIT_FILENAME,
            "INDEXATION_SUJET_CSV_UPDATE_FILENAME": config.CSV_UPDATE_FILENAME,
            "INDEXATION_SUJET_VECTORIZE_CONCEPTS_OR_CHAINS": config.VECTORIZE_CONCEPTS_OR_CHAINS,
            "INDEXATION_SUJET_VECTORIZE_AVEC_THESE": config.VECTORIZE_AVEC_THESE,
        }

        launched_containers = []
        for model in models_to_run:
            command_args = [
                "python", "rameau_vectorize.py",
                "--action", action,
                "--conceptsORchains", concepts_or_chains,
                "--alias_model", model,
                "--avec_these", avec_these,
                "--csv_filename", csv_filename
            ]

            run_kwargs = {
                "network": config.DOCKER_NETWORK,
                "environment": batch_env,
                "command": command_args,
                "volumes": {
                    volume_bind: {
                        "bind": "/app/data",
                        "mode": "rw",
                    }
                },
                "detach": True,
                "auto_remove": True, #Suppression automatique à la fin du conteneur
            }
            batch_user = getattr(config, "DOCKER_BATCH_USER", "root")
            if batch_user:
                run_kwargs["user"] = batch_user

            if device_requests:
                run_kwargs["device_requests"] = device_requests

            try:
                container = client.containers.run(config.DOCKER_IMAGE_BATCH, **run_kwargs)
            except Exception as launch_err:
                # En cas d'échec lié au runtime GPU (ex: WSL sans adaptateur GPU NVIDIA), repli automatique en mode CPU
                err_str = str(launch_err).lower()
                if device_requests and any(k in err_str for k in ("gpu", "nvidia", "adapters were found", "device_requests")):
                    print(f"Échec d'allocation GPU pour {model} ({launch_err}). Repli automatique sur l'exécution en mode CPU...")
                    run_kwargs.pop("device_requests", None)
                    container = client.containers.run(config.DOCKER_IMAGE_BATCH, **run_kwargs)
                else:
                    raise launch_err

            print(f"Conteneur batch lancé pour le modèle '{model}' : ID={container.short_id}")
            launched_containers.append({
                "model": model,
                "container_id": container.short_id,
                "command": command_args
            })

        return {
            "status": "success",
            "message": f"{len(launched_containers)} conteneurs batch lancés avec succès ({action})",
            "containers": launched_containers
        }
    except Exception as e:
        print(f"Erreur lors du lancement Docker: {e}")
        return {
            "status": "error",
            "message": f"Impossible de contacter le démon Docker: {str(e)}"
        }


@app.post("/init")
@app.get("/init")
async def init_vectorization():
    """
    Route dédiée pour lancer une initialisation complète du corpus vectoriel RAMEAU.
    Les paramètres sont issus de la configuration (config.py / .env).
    """
    return run_vectorization_pipeline("init")


@app.post("/update")
@app.get("/update")
async def update_vectorization():
    """
    Route dédiée pour lancer une mise à jour différentielle du corpus vectoriel RAMEAU.
    Les paramètres sont issus de la configuration (config.py / .env).
    """
    return run_vectorization_pipeline("update")


def save_uploaded_csv(file: UploadFile, target_filename: str) -> dict:
    """
    Enregistre un fichier CSV uploadé en flux continu dans le répertoire cible (/app/data/csv).
    Le fichier est obligatoirement conformé sous le nom target_filename et écrase l'éventuel fichier précédent.

    :param file: Fichier UploadFile transmis via FastAPI.
    :param target_filename: Nom de fichier cible imposé (ex: config.CSV_INIT_FILENAME ou config.CSV_UPDATE_FILENAME).
    :return: Dictionnaire contenant les métadonnées de l'opération ou le message d'erreur.
    """
    try:
        csv_dir = getattr(config, "CSV_DIR", "/app/data/csv")
        try:
            os.makedirs(csv_dir, exist_ok=True)
        except OSError:
            # Repli pour exécution locale hors conteneur sans privilèges sur /app
            csv_dir = os.path.join(".", "data", "csv")
            os.makedirs(csv_dir, exist_ok=True)

        # Conformation du nom de fichier cible
        clean_filename = target_filename if target_filename.endswith(".csv") else f"{target_filename}.csv"
        destination_path = os.path.join(csv_dir, clean_filename)

        # Écriture par flux (streaming) pour supporter les fichiers volumineux sans saturer la mémoire RAM
        # Le mode "wb" écrase automatiquement le fichier existant
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "filename": clean_filename,
            "original_filename": file.filename,
            "message": f"Fichier enregistré sous '{clean_filename}'",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors de l'enregistrement du fichier CSV: {str(e)}",
        }


@app.post("/init/upload")
async def upload_init_file(file: UploadFile = File(...)):
    """
    Upload du fichier CSV pour l'initialisation RAMEAU (/init) et lancement automatique
    du pipeline de vectorisation d'initialisation complet (un conteneur batch par modèle).
    Conforme le nom du fichier vers config.CSV_INIT_FILENAME (par défaut 'export_rameau.csv'),
    écrase le fichier précédent dans /app/data/csv, puis déclenche immédiatement l'initialisation.

    Exemple d'utilisation :
    curl -X POST -F "file=@mon_export.csv" http://localhost:8100/init/upload

    :param file: Fichier CSV transmis en multipart/form-data.
    :return: Dictionnaire confirmant la sauvegarde et le résultat du lancement du pipeline.
    """
    upload_result = save_uploaded_csv(file, config.CSV_INIT_FILENAME)
    if upload_result.get("status") != "success":
        return upload_result

    pipeline_result = run_vectorization_pipeline("init")
    return {
        "status": pipeline_result.get("status", "success"),
        "upload": upload_result,
        "pipeline": pipeline_result,
    }


@app.post("/update/upload")
async def upload_update_file(file: UploadFile = File(...)):
    """
    Upload du fichier CSV pour la mise à jour différentielle RAMEAU (/update) et lancement
    automatique du pipeline de vectorisation incrémentale (un conteneur batch par modèle).
    Conforme le nom du fichier vers config.CSV_UPDATE_FILENAME (par défaut 'export_rameau_update.csv'),
    écrase le fichier précédent dans /app/data/csv, puis déclenche immédiatement la mise à jour.

    Exemple d'utilisation :
    curl -X POST -F "file=@mon_delta.csv" http://localhost:8100/update/upload

    :param file: Fichier CSV transmis en multipart/form-data.
    :return: Dictionnaire confirmant la sauvegarde et le résultat du lancement du pipeline.
    """
    upload_result = save_uploaded_csv(file, config.CSV_UPDATE_FILENAME)
    if upload_result.get("status") != "success":
        return upload_result

    pipeline_result = run_vectorization_pipeline("update")
    return {
        "status": pipeline_result.get("status", "success"),
        "upload": upload_result,
        "pipeline": pipeline_result,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("load_qdrant_ws:app", host=config.API_HOST, port=config.API_PORT, workers=1)