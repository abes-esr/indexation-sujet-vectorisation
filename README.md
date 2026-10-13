# ⚙️ indexation-sujet-vectorisation : Pipeline de Vectorisation & Ingestion Qdrant RAMEAU

[![Docker Pulls](https://img.shields.io/docker/pulls/abesesr/indexation-sujet.svg)](https://hub.docker.com/r/abesesr/indexation-sujet/)
[![Buildx Publish](https://github.com/abes-esr/indexation-sujet-vectorisation/actions/workflows/buildx-pubtodockerhub.yml/badge.svg)](https://github.com/abes-esr/indexation-sujet-vectorisation/actions/workflows/buildx-pubtodockerhub.yml)

> 📌 **Rôle central du dépôt `indexation-sujet-docker`** : Le dépôt [**indexation-sujet-docker**](https://github.com/abes-esr/indexation-sujet-docker) centralise l'état et le stockage de tous les services de la plateforme. En cas d'incident majeur ou de réinstallation complète d'un serveur, la restauration est opérée depuis ce dossier. Pour la restauration de l'application, veuillez vous référer à sa documentation : [indexation-sujet-docker - Procédure de restauration globale](https://github.com/abes-esr/indexation-sujet-docker#8-procédure-de-restauration-globale).

---

## 📌 Sommaire

- [1. Description brève du projet](#1-description-brève-du-projet)
- [2. Liens vers les autres parties du projet](#2-liens-vers-les-autres-parties-du-projet)
- [3. Flux entrants / sortants](#3-flux-entrants--sortants)
- [4. Scripts & Points d'entrée (+ arguments CLI)](#4-scripts--points-dentrée--arguments-cli)
- [5. Explications des variables d'environnement](#5-explications-des-variables-denvironnement)
- [6. Diagramme d'architecture](#6-diagramme-darchitecture)
- [7. Procédure de déploiement / d'installation](#7-procédure-de-déploiement--dinstallation)
- [8. Procédure de supervision](#8-procédure-de-supervision)
- [9. Procédure de testing](#9-procédure-de-testing)

---

## 1. Description brève du projet

**`indexation-sujet-vectorisation`** est le composant de traitement par lots (_batch_) et d'ingestion vectorielle du projet **INDEXATION_SUJET** (**I**ndexation **A**utomatique **R**AMEAU), conçu et maintenu par l'**ABES** (Agence Bibliographique de l'Enseignement Supérieur).

Ce module a pour mission de transformer les données bibliographiques et d'autorités RAMEAU issues du catalogue Sudoc en représentations vectorielles denses (_embeddings_) et d'alimenter la base de données vectorielle **Qdrant** :

- **Extraction & encodage dense multi-modèles** : Calcul des vecteurs sémantiques via la bibliothèque [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`, `distiluse-base-multilingual-cased-v2`, `multilingual-e5-large`).
- **Explosion & agrégation par barycentre** : Décomposition des chaînes d'autorités en vedettes/concepts élémentaires (Unimarc `606`), regroupement de tous les documents associés à un même concept et calcul de la moyenne vectorielle afin d'établir l'empreinte sémantique unique de chaque vedette.
- **Peuplement et optimisation Qdrant** : Création et alimentation des collections vectorielles avec quantification scalaire (`INT8`) maintenue en mémoire RAM pour garantir des temps de recherche sub-milliseconde lors de l'inférence par le composant [indexation-sujet-api](https://github.com/abes-esr/indexation-sujet-api).
- **Double mode d'exécution** : Un orchestrateur Web Service REST FastAPI ([`src/load_qdrant_ws.py`](./src/load_qdrant_ws.py)) permettant de piloter les traitements via Docker-out-of-Docker (DooD), et un script CLI autonome ([`src/rameau_vectorize.py`](./src/rameau_vectorize.py)) pour les exécutions directes en environnement de calcul haute performance (GPU CUDA).

---

## 2. Liens vers les autres parties du projet

Le projet INDEXATION_SUJET est composé de plusieurs modules complémentaires hébergés sur l'organisation [abes-esr](https://github.com/abes-esr) :

| Dépôt GitHub                                                                                     | Rôle & Description                                                                                                                    |
| :----------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------ |
| [**indexation-sujet-docker**](https://github.com/abes-esr/indexation-sujet-docker)               | Configuration Docker Compose pour le déploiement global de la plateforme (API, Qdrant, Ollama/vLLM).                                  |
| [**indexation-sujet-api**](https://github.com/abes-esr/indexation-sujet-api)                     | Service web backend d'inférence (FastAPI) exposant le service de suggestion d'indexation sujet RAMEAU.                                |
| [**indexation-sujet-vectorisation**](https://github.com/abes-esr/indexation-sujet-vectorisation) | **Ce dépôt** : Pipeline de vectorisation batch des notices RAMEAU et de peuplement des collections Qdrant.                            |
| [**indexation-sujet-batch-docker**](https://github.com/abes-esr/indexation-sujet-batch-docker)   | Déploiement Docker pour les traitements par lots et les interfaces de mise à jour des données.                                        |
| [**indexation-sujet-batch-dump**](https://github.com/abes-esr/indexation-sujet-batch-dump)       | Extraction, transformation et génération des dumps de données d'autorités RAMEAU et notices Sudoc.                                    |
| [**indexation-sujet-script-winibw**](https://github.com/abes-esr/indexation-sujet-script-winibw) | Script client (VBScript) s'intégrant au client lourd de catalogage **WinIBW** pour interroger l'API depuis le poste des catalogueurs. |

---

## 3. Flux entrants / sortants

### Flux entrants :

- **Fichiers d'export RAMEAU (Dumps CSV)** :
  - Générés par le composant [indexation-sujet-batch-dump](https://github.com/abes-esr/indexation-sujet-batch-dump) (ou issus de la procédure Oracle de l'ABES `QE_EXPORT_RAMEAU`).
  - Déposés dans le répertoire partagé [`volumes/csv/`](./volumes/csv/) ou téléversés via l'API :
    - `export_rameau.csv` : Fichier de référence pour l'initialisation complète (`/init`).
    - `export_rameau_update.csv` : Fichier différentiel pour les mises à jour incrémentales (`/update`).
- **Poids des modèles de langage pré-entraînés (HuggingFace)** :
  - Téléchargés automatiquement lors de la première instanciation via `sentence-transformers` :
    - `all-MiniLM-L6-v2` (alias `allMin`, dimension 384)
    - `sentence-transformers/distiluse-base-multilingual-cased-v2` (alias `distiluse`, dimension 512)
    - `intfloat/multilingual-e5-large` (alias `e5-large`, dimension 1024)
- **Requêtes HTTP d'orchestration (REST)** :
  - Déclenchement d'initialisation, de mise à jour ou téléversement de fichiers multipart vers le web service [`src/load_qdrant_ws.py`](./src/load_qdrant_ws.py).
- **Socket Docker de l'hôte** (`/var/run/docker.sock`) :
  - Ordres de création et d'exécution de conteneurs de calcul éphémères en mode Docker-out-of-Docker (DooD).

### Flux sortants :

- **Points vectoriels et métadonnées vers Qdrant** (port `6333`) :
  - Ingestion gRPC / REST par lots dans les collections vectorielles cibles (ex. `concepts_allMin_only_mono`).
  - Chaque point contient : un identifiant numérique, un vecteur dense moyenné normalisé, et un payload associant le libellé de la vedette-matière RAMEAU.
- **Fichiers de persistance et archives vectorielles** :
  - Fichiers sérialisés NumPy / Pandas [`volumes/pkl/archive_*.pkl`](./volumes/pkl/) servant de point de restauration rapide ou de base de calcul incrémental.
  - Fichiers d'horodatage et de configuration de suivi (`*_init.ini`, `*_update.ini`).
- **Snapshots binaires Qdrant** :
  - Fichiers binaires de sauvegarde de collections générés dans `volumes/qdrant/snapshots/`.
- **Flux de logs (stdout / stderr)** :
  - Sortie standard capturée par Docker, visualisable en temps réel sur Dozzle et indexée dans le puits de logs Kibana de l'ABES.

---

## 4. Scripts & Points d'entrée (+ arguments CLI)

Le projet s'articule autour de deux composants logiciels situés dans le répertoire [`src/`](./src/) :

```text
src/
├── config.py             # Centralisation des variables d'environnement et chemins
├── load_qdrant_ws.py     # Web Service REST FastAPI d'orchestration Docker batch
└── rameau_vectorize.py   # Moteur CLI de vectorisation batch et d'ingestion Qdrant
```

---

### A. Web Service d'Orchestration REST (`load_qdrant_ws.py`)

Le Web Service FastAPI écoute par défaut sur le port configuré par `INDEXATION_SUJET_VECTORISATION_PORT` (défaut `6381`, ou `8100` selon déploiement) et pilote l'exécution conteneurisée des traitements lourds sans bloquer le serveur web.

#### Endpoints disponibles :

| Méthode        | Route            | Description                                                                                                                                                                                              |
| :------------- | :--------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`          | `/health`        | Sonde de santé applicative (Healthcheck). Renvoie `{"status": "ok"}`.                                                                                                                                    |
| `POST` / `GET` | `/init`          | Déclenche l'initialisation complète du corpus vectoriel. Lance en parallèle un conteneur éphémère par modèle (`allMin`, `distiluse`, `e5-large`).                                                        |
| `POST` / `GET` | `/update`        | Déclenche la mise à jour incrémentale différentielle du corpus vectoriel pour les 3 modèles.                                                                                                             |
| `POST`         | `/init/upload`   | Téléverse un fichier CSV d'initialisation, le conforme sous `INDEXATION_SUJET_CSV_INIT_FILENAME` (`export_rameau.csv`), puis **déclenche automatiquement le pipeline d'initialisation** (`init`).        |
| `POST`         | `/update/upload` | Téléverse un fichier CSV de mise à jour, le conforme sous `INDEXATION_SUJET_CSV_UPDATE_FILENAME` (`export_rameau_update.csv`), puis **déclenche automatiquement le pipeline de mise à jour** (`update`). |

#### Exemples d'appels `curl` :

```bash
# Vérifier la disponibilité du service
curl -s http://localhost:6381/health

# Téléverser un nouvel export initial et lancer la vectorisation complète dans la foulée
curl -X POST -F "file=@mon_export_rameau.csv" http://localhost:6381/init/upload

# Téléverser un fichier différentiel et lancer la mise à jour vectorielle dans la foulée
curl -X POST -F "file=@mon_delta_rameau.csv" http://localhost:6381/update/upload

# Déclencher manuellement une initialisation (sans ré-uploader)
curl -X POST http://localhost:6381/init

# Déclencher manuellement une mise à jour différentielle (sans ré-uploader)
curl -X POST http://localhost:6381/update
```

---

### B. Pipeline CLI de Vectorisation Batch (`rameau_vectorize.py`)

Le script [`src/rameau_vectorize.py`](./src/rameau_vectorize.py) constitue le moteur de calcul haute performance. Il peut être lancé directement en ligne de commande (sur l'hôte ou dans un conteneur éphémère) :

```bash
python src/rameau_vectorize.py --action init --conceptsORchains concepts --alias_model allMin --avec_these only_mono
```

#### Options en ligne de commande :

| Option longue        | Option courte | Valeurs acceptées                         |   Défaut    | Description                                                                                                                                                                                                                                                                                                                                                                                    |
| :------------------- | :-----------: | :---------------------------------------- | :---------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--action`           |     `-a`      | `init`, `update`, `auto`, `restore`       |  `update`   | **Mode d'exécution** :<br>• `init` : Recalcul complet et écrasement de la collection Qdrant.<br>• `update` : Mise à jour différentielle à partir de l'archive existante.<br>• `auto` : Détection automatique (`init` ou `update`) basée sur les dates de modification des CSV.<br>• `restore` : Ré-ingestion directe dans Qdrant depuis l'archive `archive_*.pkl` sans ré-encoder les modèles. |
| `--conceptsORchains` |     `-c`      | `concepts`, `chains`                      | `concepts`  | **Typologie de découpage RAMEAU** :<br>• `concepts` : Vedettes élémentaires individuelles (ex. `Algèbre linéaire`).<br>• `chains` : Chaînes d'indexation complètes avec subdivisions (ex. `France -- Histoire -- 1789-1799`).                                                                                                                                                                  |
| `--alias_model`      |     `-m`      | `allMin`, `distiluse`, `e5-large`         |  `allMin`   | **Modèle d'embedding** :<br>• `allMin` : `all-MiniLM-L6-v2` (384 dim)<br>• `distiluse` : `distiluse-base-multilingual-cased-v2` (512 dim)<br>• `e5-large` : `intfloat/multilingual-e5-large` (1024 dim)                                                                                                                                                                                        |
| `--avec_these`       |     `-t`      | `only_mono`, `only_theses`, `with_theses` | `only_mono` | **Périmètre documentaire** :<br>• `only_mono` : Monographies imprimées/électroniques uniquement.<br>• `only_theses` : Thèses de doctorat uniquement.<br>• `with_theses` : Monographies et thèses confondues.                                                                                                                                                                                   |
| `--csv_filename`     |       -       | _Nom de fichier_                          |    `""`     | Nom ou chemin d'un CSV spécifique à traiter (ex. `test100.csv` pour un jeu de test). Par défaut, utilise les fichiers configurés dans `.env`.                                                                                                                                                                                                                                                  |

#### Règles de gestion appliquées par le script :

1. **Nommage standardisé de la collection Qdrant** :  
   La collection créée dans Qdrant adopte la nomenclature stricte `{conceptsORchains}_{alias_model}_{avec_these}` (exemple : `concepts_allMin_only_mono`).
2. **Garde-fou anti-écrasement (`INDEXATION_SUJET_QDRANT_MIN_VECTORS`)** :  
   Si le nombre de vecteurs générés est inférieur au seuil minimal (configuré à 1 000 en production, ajustable pour les tests), l'ingestion est immédiatement interrompue afin de ne pas vider une collection opérationnelle suite à un export CSV corrompu.
3. **Quantification scalaire INT8** :  
   La collection est créée avec `type: INT8` et `always_ram: True`, réduisant l'empreinte mémoire d'un facteur 4 tout en préservant la précision cosinus.
4. **Vérification k-PPV post-ingestion** :  
   Dès la fin de l'upload des points, une recherche k-PPV de contrôle est exécutée sur la requête `"la radio"` pour valider le bon fonctionnement de l'index vectoriel.

---

## 5. Explications des variables d'environnement

Les variables d'environnement sont centralisées dans [`src/config.py`](./src/config.py) et chargées à partir du fichier `.env` (dérivé du modèle versionné [`.env-dist`](./.env-dist)).

| Variable                                        | Description                                                                                          | Exemple / Défaut                   | Contexte          |
| :---------------------------------------------- | :--------------------------------------------------------------------------------------------------- | :--------------------------------- | :---------------- |
| `INDEXATION_SUJET_QDRANT_HOST`                  | Hôte ou nom de service réseau de l'instance Qdrant.                                                  | `localhost` (ou `qdrant`)          | Connexion Qdrant  |
| `INDEXATION_SUJET_QDRANT_PORT`                  | Port de communication REST/gRPC de Qdrant.                                                           | `6333`                             | Connexion Qdrant  |
| `INDEXATION_SUJET_QDRANT_TEST_LIMIT`            | Nombre de résultats retournés lors de la requête de test de conformité post-ingestion.               | `6`                                | Test post-vol     |
| `INDEXATION_SUJET_QDRANT_MIN_VECTORS`           | Seuil minimal de vecteurs requis pour autoriser l'écrasement d'une collection Qdrant.                | `1000` (prod) / `10` (test)        | Sécurité données  |
| `INDEXATION_SUJET_VECTORISATION_HOST`           | Adresse IP d'écoute du serveur FastAPI / Uvicorn.                                                    | `0.0.0.0`                          | Web Service       |
| `INDEXATION_SUJET_VECTORISATION_PORT`           | Port d'écoute du serveur Web Service d'orchestration.                                                | `6381` (ou `8100`)                 | Web Service       |
| `INDEXATION_SUJET_DOCKER_VOLUME_BIND`           | Chemin du répertoire hôte contenant les volumes à monter dans les conteneurs batch (`/app/data`).    | `./volumes`                        | Docker batch      |
| `INDEXATION_SUJET_DOCKER_NETWORK`               | Nom du réseau Docker partagé entre Qdrant et les conteneurs de vectorisation.                        | `mon_reseau`                       | Docker batch      |
| `INDEXATION_SUJET_DOCKER_IMAGE_BATCH`           | Image Docker exécutée pour les tâches de calcul batch éphémères.                                     | `rameau_vectorize_batch:latest`    | Docker batch      |
| `INDEXATION_SUJET_DOCKER_SOCK`                  | URI du socket du démon Docker utilisé pour l'orchestration DooD.                                     | `unix:///var/run/docker.sock`      | Docker batch      |
| `INDEXATION_SUJET_DOCKER_BATCH_USER`            | Utilisateur système exécutant le processus dans le conteneur batch.                                  | `root`                             | Sécurité / Droits |
| `INDEXATION_SUJET_CSV_INIT_FILENAME`            | Nom du fichier CSV d'initialisation attendu dans `./volumes/csv/`.                                   | `export_rameau.csv`                | Données sources   |
| `INDEXATION_SUJET_CSV_UPDATE_FILENAME`          | Nom du fichier CSV de mise à jour différentielle attendu dans `./volumes/csv/`.                      | `export_rameau_update.csv`         | Données sources   |
| `INDEXATION_SUJET_VECTORIZE_CONCEPTS_OR_CHAINS` | Typologie de vedettes à vectoriser par défaut lors d'un appel à `/init` ou `/update`.                | `concepts`                         | Paramétrage batch |
| `INDEXATION_SUJET_VECTORIZE_AVEC_THESE`         | Périmètre documentaire traité par défaut lors d'un appel à `/init` ou `/update`.                     | `only_mono`                        | Paramétrage batch |
| `ENABLE_GPU`                                    | Active l'allocation matérielle GPU NVIDIA (`--gpus all` / device requests) pour l'inférence PyTorch. | `false` (local) / `true` (serveur) | Performance       |

> [!CAUTION]
> **Consigne de sécurité ABES** : Ne versionnez jamais le fichier `.env` contenant les configurations locales ou spécifiques aux serveurs. Seul [`.env-dist`](./.env-dist) doit être commité.

---

## 6. Diagramme d'architecture

Le diagramme suivant détaille le cycle de transformation de la donnée au sein d'`indexation-sujet-vectorisation` et ses interactions avec l'environnement Docker et Qdrant :

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SOURCES DE DONNÉES                                     │
│  • Dumps CSV RAMEAU (indexation-sujet-batch-dump / Oracle) dans ./volumes/csv/                      │
│  • Modèles HuggingFace : all-MiniLM-L6-v2 (384d), distiluse (512d), e5-large (1024d)   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Dépôt CSV ou upload HTTP multipart
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATEUR FASTAPI (src/load_qdrant_ws.py)                       │
│                        Écoute : Port 6381 (ou 8100 dans Docker)                        │
│  • Endpoints : /health, /init, /update, /init/upload, /update/upload                   │
│  • Pilotage Docker-out-of-Docker (DooD) via /var/run/docker.sock                       │
│  • Instanciation en parallèle de 3 conteneurs batch distincts (1 par modèle)           │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Déclenchement conteneurs éphémères (--rm)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             PIPELINE DE VECTORISATION BATCH (src/rameau_vectorize.py)                  │
│                        Image : rameau_vectorize_batch:latest                           │
│                                                                                        │
│  ┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────────┐  │
│  │     1. Prétraitement    │   │   2. Encodage Dense      │   │  3. Agrégation      │  │
│  │ • Nettoyage CSV         │──>| • SentenceTransformers   │──>│ • Groupby vedette   │  │
│  │ • Découpage par lots    │   │ • Accélération GPU CUDA  │   │ • Moyenne vecteurs  │  │
│  │ • Explosion concepts    │   │ • Mini-batchs(5k notices)│   │ • Sauvegarde .pkl   │  │
│  └─────────────────────────┘   └──────────────────────────┘   └─────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Ingestion gRPC / REST (Points, Vecteurs, Payloads)
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     BASE VECTORIELLE QDRANT (Instance Port 6333)                       │
│                                                                                        │
│  ┌─────────────────────────────────┐       ┌────────────────────────────────────────┐  │
│  │       Collections RAMEAU        │       │             Optimisations              │  │
│  │ • concepts_allMin_only_mono     │       │ • Quantification INT8 (always_ram)     │  │
│  │ • concepts_distiluse_only_mono  │       │ • Stockage vecteurs sur disque         │  │
│  │ • concepts_e5-large_only_mono   │       │ • Snapshots binaires de sauvegarde     │  │
│  └─────────────────────────────────┘       └────────────────────────────────────────┘  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Métriques, Logs & Dashboard
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              SUPERVISION & OBSERVABILITÉ                               │
│  • Dashboard Qdrant (Port 6333) : Métriques de collections, comptage de vecteurs       │
│  • Dozzle (Port 29999) : Consultation temps réel des flux de logs conteneurs           │
│  • Grafana / Kibana ABES (Port 3000) : Puits de logs centralisé                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Procédure de déploiement / d'installation

> ⚠️
> **Architecture de déploiement ABES** :  
> Sur les serveurs de test et de production de l'ABES (`donut-test` et `donut-prod`), **seul le dépôt [`indexation-sujet-docker`](https://github.com/abes-esr/indexation-sujet-docker)** est déployé sur les machines hôtes (dans `/opt/pod/indexation-sujet-docker/`).  
> Le composant `indexation-sujet-vectorisation` est intégré sous forme d'image Docker précompilée (`abesesr/indexation-sujet:*-vectorisation`).  
> Les procédures détaillées ci-après décrivent l'installation pour les développeurs, le build local et l'exécution conteneurisée autonome.

---

### A. Déploiement Conteneurisé avec Docker (Recommandé)

Cette méthode isole entièrement les dépendances PyTorch, gère l'accélération GPU NVIDIA et assure la communication réseau avec Qdrant.

#### 1. Préparation du réseau et de l'instance Qdrant

```bash
# 1. Création du réseau Docker partagé
docker network create mon_reseau

# 2. Démarrage de Qdrant sur le réseau partagé
docker run -d --name qdrant \
  --network mon_reseau \
  -p 6333:6333 \
  -v "$(pwd)/volumes/qdrant/storage":/qdrant/storage \
  -v "$(pwd)/volumes/qdrant/snapshots":/qdrant/snapshots \
  qdrant/qdrant
```

#### 2. Construction de l'image Docker

```bash
# 1. Cloner le dépôt et se positionner à la racine
git clone https://github.com/abes-esr/indexation-sujet-vectorisation.git
cd indexation-sujet-vectorisation

# 2. Copier le fichier d'environnement
cp .env-dist .env

# 3. Construire l'image applicative
docker build -t indexation-sujet-vectorisation .

# 4. Appliquer le tag requis pour les conteneurs éphémères déclenchés par le Web Service
docker tag indexation-sujet-vectorisation rameau_vectorize_batch:latest
```

#### 3. Démarrage du Web Service d'orchestration (`load_qdrant_ws.py`)

Le conteneur monte le socket Docker hôte afin de pouvoir instancier les sous-conteneurs de vectorisation (mode DooD) :

```bash
# Sur Linux / macOS :
docker run -d \
  --name indexation-sujet-vectorisation \
  --user root \
  --network mon_reseau \
  -p 6381:8100 \
  --env-file .env \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(pwd)/volumes":/app/data \
  indexation-sujet-vectorisation

# Sur Windows PowerShell :
docker run -d `
  --name indexation-sujet-vectorisation `
  --user root `
  --network mon_reseau `
  -p 6381:8100 `
  --env-file .env `
  -v //var/run/docker.sock:/var/run/docker.sock `
  -v ${PWD}/volumes:/app/data `
  indexation-sujet-vectorisation
```

#### 4. Exécution ponctuelle directe du batch en conteneur éphémère (`--rm`)

Pour lancer un traitement complet en ligne de commande sans démarrer le serveur web :

- **Avec accélération GPU NVIDIA** (strictement équivalent à la réservation `count: all` / `capabilities: [gpu]` définie dans le `docker-compose.yml` d'[indexation-sujet-docker](https://github.com/abes-esr/indexation-sujet-docker)) :

  ```bash
  docker run --rm -it \
    --network mon_reseau \
    --gpus all \
    -v "$(pwd)/volumes":/app/data \
    indexation-sujet-vectorisation \
    python rameau_vectorize.py --action init --conceptsORchains concepts --alias_model allMin --avec_these only_mono
  ```

- **Sans GPU (exécution CPU)** :
  ```bash
  docker run --rm -it \
    --network mon_reseau \
    -v "$(pwd)/volumes":/app/data \
    indexation-sujet-vectorisation \
    python rameau_vectorize.py --action init --conceptsORchains concepts --alias_model allMin --avec_these only_mono
  ```

---

### B. Installation Locale pour le Développement (Python 3.10+)

Pour développer et déboguer directement sans conteneur Docker :

#### 1. Prérequis système

- Python 3.10 ou supérieur
- Carte graphique NVIDIA avec pilotes CUDA récents (recommandé pour la vitesse d'inférence)

#### 2. Environnement virtuel & Dépendances

```bash
# 1. Création de l'environnement virtuel
python -m venv venv

# 2. Activation de l'environnement virtuel
# Linux / macOS :
source venv/bin/activate
# Windows (PowerShell) :
.\venv\Scripts\Activate.ps1
# Windows (Git Bash) :
source venv/Scripts/activate
# Windows (CMD) :
venv\Scripts\activate.bat

# 3. Mise à niveau de pip et installation de PyTorch avec support CUDA si disponible
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 4. Installation des dépendances du projet
pip install -r requirements.txt

# 5. Configuration de l'environnement
cp .env-dist .env
# Adapter .env (régler INDEXATION_SUJET_QDRANT_HOST=localhost, ENABLE_GPU=true/false)
```

#### 3. Lancement des composants en local

- **Lancement du Web Service (mode rechargement à chaud)** :

  ```bash
  python -m uvicorn src.load_qdrant_ws:app --host 0.0.0.0 --port 6381 --reload
  ```

- **Exécution directe du pipeline batch** :
  ```bash
  python src/rameau_vectorize.py --action init --conceptsORchains concepts --alias_model allMin --avec_these only_mono
  ```

---

## 8. Procédure de supervision

La surveillance de la vectorisation et de l'ingestion s'appuie sur plusieurs sondes et interfaces :

### 1. Tableaux de bord Qdrant (Dashboard)

Permet de vérifier en temps réel le nombre de points insérés, l'état de la quantification `INT8`, l'indexation HNSW et la taille mémoire des segments :

- **Local** : [`http://localhost:6333/dashboard#/collections`](http://localhost:6333/dashboard#/collections)
- **Serveur de Test ABES** : [`http://donut-test.abes.fr:6333/dashboard#/collections`](http://donut-test.abes.fr:6333/dashboard#/collections)
- **Serveur de Production ABES** : [`http://donut-prod.abes.fr:6333/dashboard#/collections`](http://donut-prod.abes.fr:6333/dashboard#/collections)

### 2. Consultation des logs applicatifs en temps réel (Dozzle)

Les logs d'avancement des mini-lots de vectorisation et de création des fichiers `.pkl` sont streamés en continu :

- **Serveur de Test ABES** : [`http://donut-test.abes.fr:29999/`](http://donut-test.abes.fr:29999/)
- **Serveur de Production ABES** : [`http://donut-prod.abes.fr:29999/`](http://donut-prod.abes.fr:29999/)
- **En ligne de commande Docker** :

  ```bash
  # Suivre les logs du web service
  docker logs -f indexation-sujet-vectorisation

  # Suivre les conteneurs batchs éphémères lancés
  docker ps -a --filter "ancestor=rameau_vectorize_batch:latest"
  ```

### 3. Sonde de santé applicative (Healthcheck)

Le web service expose une route standard de vérification de disponibilité :

```bash
curl -f http://localhost:6381/health
# Réponse attendue : {"status":"ok"}
```

### 4. Supervision globale et observabilité ABES

Les métriques serveurs et conteneurs sont agrégées sur les instances Grafana de l'infrastructure ABES :

- **Développement** : `http://diplotaxis7-dev.v212.abes.fr:3000`
- **Test** : `http://diplotaxis7-test.v202.abes.fr:3000`
- **Production** : `http://diplotaxis7-prod.v102.abes.fr:3000`

---

## 9. Procédure de testing

La validation technique et sémantique s'articule autour des scripts du répertoire [`test/`](./test/) :

```text
test/
├── data/
│   └── test_data.csv          # Référence des PPN exclus pour prévenir le data leakage
└── test_preflight_checks.py   # Script de contrôle pré-vol de la RAM et des paramètres
```

### 1. Script de Vérification Pré-Vol (`test_preflight_checks.py`)

Ce script s'assure que la machine dispose des prérequis matériels nécessaires (mémoire RAM suffisante pour charger les modèles volumineux comme `e5-large`) et valide la conformité des arguments passés en ligne de commande. Il prévient les arrêts brutaux dus aux dépassements de mémoire (_OOM Kill_) :

```bash
python test/test_preflight_checks.py \
  --action update \
  --conceptsORchains concepts \
  --alias_model allMin \
  --avec_these only_mono
```

---

### 2. Données d'Évaluation & Prévention du _Data Leakage_ (`test/data/test_data.csv`)

Le fichier [`test/data/test_data.csv`](./test/data/test_data.csv) recense la liste stricte des PPN de notices bibliographiques réservés aux tests d'évaluation d'exactitude (_Top-1, Top-3, Top-5 accuracy_) exécutés dans [indexation-sujet-api](https://github.com/abes-esr/indexation-sujet-api).  
Le pipeline de vectorisation garantit l'isolation de ces notices afin de préserver l'intégrité scientifique des benchmarks algorithmiques.

---

### 3. Tests d'Intégration & Cohérence Sémantique

#### Validation automatique en fin de pipeline :

À l'issue de chaque exécution d'ingestion par [`src/rameau_vectorize.py`](./src/rameau_vectorize.py), une requête de test sur l'expression `"la radio"` est immédiatement soumise à la collection Qdrant créée. Le script valide que :

- Des points sont renvoyés avec un score de similarité cosinus pertinent.
- Les payloads associés contiennent des vedettes cohérentes (ex. `Radio`, `Radiodiffusion`).

#### Validation des dimensions vectorielles :

Les dimensions des collections Qdrant créées doivent strictement correspondre aux spécifications des modèles :

- `concepts_allMin_only_mono` : **384** dimensions.
- `concepts_distiluse_only_mono` : **512** dimensions.
- `concepts_e5-large_only_mono` : **1024** dimensions.

#### Procédure de test rapide sur échantillon réduit :

Pour valider le bon fonctionnement de la chaîne complète sans traiter les 40 000 vedettes de la base RAMEAU :

1. Déposer un fichier d'échantillon (ex. `test100.csv`) dans `./volumes/csv/`.
2. Définir `INDEXATION_SUJET_QDRANT_MIN_VECTORS=10` dans votre fichier `.env`.
3. Exécuter le batch avec l'option `--csv_filename` :
   ```bash
   python src/rameau_vectorize.py --action init --conceptsORchains concepts --alias_model allMin --avec_these only_mono --csv_filename test100.csv
   ```
4. Consulter la collection créée sur le dashboard Qdrant ([`http://localhost:6333/dashboard#/collections`](http://localhost:6333/dashboard#/collections)).
