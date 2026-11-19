import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, UploadFile, File

# Assure la résolution directe de config quel que soit le contexte d'exécution (racine ou src)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# Script batch exécuté dans un sous-processus par combinaison modèle × typologie × langue
VECTORIZE_SCRIPT = Path(__file__).resolve().parent / "rameau_vectorize.py"
APP_ROOT = VECTORIZE_SCRIPT.parent.parent

app = FastAPI()

# Verrou empêchant le lancement de deux pipelines simultanés (init/update)
_pipeline_lock = threading.Lock()
# État courant (ou dernier) du pipeline, exposé via /status
_pipeline_state: dict = {"running": False, "action": None, "jobs": []}


@app.get("/health")
async def health():
    """
    Sonde de disponibilité applicative (Healthcheck).
    Retourne le statut de fonctionnement du service.

    :return: Dictionnaire indiquant le statut opérationnel ("ok").
    """
    return {"status": "ok"}


def _available_cpus() -> float:
    """
    Nombre de CPU réellement alloués au conteneur : lit le quota cgroup (cpus: du docker-compose)
    car os.cpu_count() renvoie le nombre de cœurs de l'hôte.
    """
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    return float(os.cpu_count() or 1)


def _threads_per_job() -> int:
    """
    Nombre de threads de calcul (torch/OMP/MKL) par sous-processus : valeur configurée, sinon
    répartition des CPU du conteneur entre les sous-processus simultanés (évite la sursouscription).
    """
    if config.VECTORIZE_THREADS_PER_JOB > 0:
        return config.VECTORIZE_THREADS_PER_JOB
    return max(1, int(_available_cpus() // config.VECTORIZE_MAX_PARALLEL))


def _run_job(job: dict, env: dict):
    """
    Exécute un sous-processus de vectorisation et attend sa fin (exécuté dans un thread du pool).
    La sortie est relayée dans les logs du service, préfixée par la combinaison traitée.
    """
    label = f"{job['model']}/{job['concepts_or_chains']}/{job['lang']}"
    try:
        job["status"] = "running"
        process = subprocess.Popen(
            job["command"],
            cwd=APP_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        job["pid"] = process.pid
        print(f"[{label}] sous-processus lancé : PID={process.pid}", flush=True)
        for line in process.stdout:
            print(f"[{label}] {line}", end="", flush=True)
        exit_code = process.wait()
        job["exit_code"] = exit_code
        job["status"] = "success" if exit_code == 0 else "failed"
        print(f"[{label}] terminé : code={exit_code}", flush=True)
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        print(f"[{label}] erreur : {e}", flush=True)


def _run_queue(jobs: list, env: dict):
    """
    Exécute la file des sous-processus avec au plus config.VECTORIZE_MAX_PARALLEL simultanés,
    puis libère le verrou du pipeline.
    """
    try:
        with ThreadPoolExecutor(max_workers=config.VECTORIZE_MAX_PARALLEL) as pool:
            for job in jobs:
                pool.submit(_run_job, job, env)
        failed = [j for j in jobs if j["status"] != "success"]
        print(f"Pipeline {_pipeline_state['action']} terminé : {len(jobs) - len(failed)} succès, "
              f"{len(failed)} échec(s)", flush=True)
    finally:
        _pipeline_state["running"] = False
        _pipeline_lock.release()


def run_vectorization_pipeline(action: str) -> dict:
    """
    Lance le pipeline de vectorisation (init ou update) : un sous-processus rameau_vectorize.py par
    combinaison modèle × typologie × langue, exécutés en arrière-plan dans une file limitée à
    config.VECTORIZE_MAX_PARALLEL sous-processus simultanés. Les ressources globales sont bornées par
    les limites (mem_limit / cpus) du conteneur. Suivi de l'avancement via la route /status.
    Les paramètres (typologie, concepts/chaînes, CSV) sont extraits directement
    de la configuration applicative afin de prévenir toute injection de commande.

    :param action: Action à exécuter ('init' ou 'update').
    :return: Dictionnaire contenant le statut et la liste des traitements mis en file.
    """
    csv_filename = config.CSV_INIT_FILENAME if action == "init" else config.CSV_UPDATE_FILENAME

    # Validation stricte des valeurs (elles finissent dans la ligne de commande des sous-processus)
    invalid = [v for v in config.VECTORIZE_CONCEPTS_OR_CHAINS if v not in ("concepts", "chains")]
    invalid += [v for v in config.VECTORIZE_LANGS if v not in ("fr", "all")]
    if invalid or not config.VECTORIZE_CONCEPTS_OR_CHAINS or not config.VECTORIZE_LANGS:
        return {
            "status": "error",
            "message": f"Configuration invalide (concepts/chains ou langs) : {invalid or 'liste vide'}",
        }

    if not _pipeline_lock.acquire(blocking=False):
        return {
            "status": "error",
            "message": f"Un pipeline ({_pipeline_state['action']}) est déjà en cours, réessayez plus tard (suivi : /status)",
        }

    threads = str(_threads_per_job())
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
    }

    jobs = [
        {
            "model": model,
            "concepts_or_chains": concepts_or_chains,
            "lang": lang,
            "status": "queued",
            "pid": None,
            "command": [
                sys.executable, str(VECTORIZE_SCRIPT),
                "--action", action,
                "--conceptsORchains", concepts_or_chains,
                "--alias_model", model,
                "--lang", lang,
                "--avec_these", config.VECTORIZE_AVEC_THESE,
                "--csv_filename", csv_filename,
            ],
        }
        for concepts_or_chains in config.VECTORIZE_CONCEPTS_OR_CHAINS
        for lang in config.VECTORIZE_LANGS
        for model in config.MODELS
    ]

    print(f"Lancement de la vectorisation (action={action}) : {len(jobs)} sous-processus, "
          f"{config.VECTORIZE_MAX_PARALLEL} en parallèle max, {threads} threads chacun "
          f"(modèles={config.MODELS}, typologies={config.VECTORIZE_CONCEPTS_OR_CHAINS}, "
          f"langues={config.VECTORIZE_LANGS})", flush=True)

    _pipeline_state.update({"running": True, "action": action, "jobs": jobs})
    try:
        # Exécution en arrière-plan : la route répond immédiatement, la file est traitée par le thread
        threading.Thread(target=_run_queue, args=(jobs, env), daemon=True).start()
    except Exception:
        _pipeline_state["running"] = False
        _pipeline_lock.release()
        raise

    return {
        "status": "success",
        "message": f"{len(jobs)} traitements mis en file ({action}), "
                   f"{config.VECTORIZE_MAX_PARALLEL} exécutés en parallèle max (suivi : /status)",
        "jobs": jobs,
    }


@app.get("/status")
async def pipeline_status():
    """
    État du pipeline de vectorisation en cours (ou du dernier exécuté) :
    statut de chaque traitement (queued, running, success, failed).
    """
    return _pipeline_state


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
    du pipeline de vectorisation d'initialisation complet.
    Conforme le nom du fichier vers config.CSV_INIT_FILENAME (par défaut 'export_rameau.csv'),
    écrase le fichier précédent dans /app/data/csv, puis déclenche immédiatement l'initialisation.

    Exemple d'utilisation :
    curl -X POST -F "file=@mon_export.csv" http://localhost:8100/init/upload

    :param file: Fichier CSV transmis en multipart/form-data.
    :return: Dictionnaire confirmant la sauvegarde et le résultat du lancement du pipeline.
    """
    # Refus avant écrasement du CSV si un pipeline est en cours (il est peut-être en train de le lire)
    if _pipeline_state["running"]:
        return {"status": "error", "message": "Un pipeline est déjà en cours, réessayez plus tard (suivi : /status)"}
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
    automatique du pipeline de vectorisation incrémentale.
    Conforme le nom du fichier vers config.CSV_UPDATE_FILENAME (par défaut 'export_rameau_update.csv'),
    écrase le fichier précédent dans /app/data/csv, puis déclenche immédiatement la mise à jour.

    Exemple d'utilisation :
    curl -X POST -F "file=@mon_delta.csv" http://localhost:8100/update/upload

    :param file: Fichier CSV transmis en multipart/form-data.
    :return: Dictionnaire confirmant la sauvegarde et le résultat du lancement du pipeline.
    """
    # Refus avant écrasement du CSV si un pipeline est en cours (il est peut-être en train de le lire)
    if _pipeline_state["running"]:
        return {"status": "error", "message": "Un pipeline est déjà en cours, réessayez plus tard (suivi : /status)"}
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
