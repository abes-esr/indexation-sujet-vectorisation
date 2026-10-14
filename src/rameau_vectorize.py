#!/usr/bin/env python
# coding: utf-8

"""
Module de vectorisation batch et d'ingestion des autorités RAMEAU dans Qdrant.
Calcule les représentations vectorielles (SentenceTransformers), agrège les barycentres
par vedette-matière et insère les points dans Qdrant.
"""

# la procedure oracle de traitement sur la base xml est QE_EXPORT_RAMEAU
# pour initier les serveurs ; 
# cd /opt ; source jupyter/bin/activate ; cd /home/ubuntu/cbd
# lancer le service de chargement et de lancement de la vectorisation il ecoute sur le port 8100:
# python load_qdrant_ws.py 
# url de test 
# http://152.228.161.151:8100/lanceVectorisation/?action=update1&conceptsORchains=concepts&alias_model=allMin&avec_these=only_mono

# lancer le service wided de test il ecoute sur le port 8068:
# cd /opt ; source jupyter/bin/activate ; cd /home/ubuntu/yann/webservice
# python wided-test_cbd3.py
#pour tester http://152.228.161.151:8068/subject_indexation/?docId=NULL&Title=la%20radio&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=intersection2models,llm&subjects&MaxCount=6&Agent=RCR&vocabulary=rameau&Format=text

import re
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import pickle
from datetime import datetime 
from pathlib import Path
import os
import configparser
import optparse
import sys
# Assure la résolution directe de config quel que soit le contexte d'exécution (racine ou src)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# In[5]:

def resolve_csv_path(filename: str) -> str:
    """
    Localise un fichier CSV selon l'ordre de priorité suivant:
    1. Dans /app/data/csv
    2. Dans /app/data
    3. Répertoires de secours ou relatifs
    4. Le chemin direct fourni

    :param filename: Nom du fichier ou chemin (avec ou sans extension .csv)
    :return: Chemin résolu existant ou première cible par défaut
    """
    if not filename.endswith('.csv'):
        filename_with_ext = f"{filename}.csv"
    else:
        filename_with_ext = filename

    candidate_paths = [
        os.path.join('/app/data/csv', filename_with_ext),
        os.path.join('/app/data', filename_with_ext),
        os.path.join('/app', filename_with_ext),
        os.path.join('.', filename_with_ext),
        os.path.join('./data/csv', filename_with_ext),
        os.path.join('./data', filename_with_ext),
        filename_with_ext,
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            print(f"Fichier CSV trouvé : {path}")
            return path

    print(f"ATTENTION : Fichier CSV '{filename_with_ext}' introuvable dans les répertoires candidats : {candidate_paths}")
    return candidate_paths[0]


def launch(action_in, conceptsORchains, alias_model, avec_these, csv_filename_in=None):
    """
    Orchestration principale du pipeline de vectorisation batch.
    Calcule les représentations vectorielles, agrège les moyennes et alimente Qdrant.

    :param action_in: Mode d'exécution ('init', 'update', 'auto', 'restore').
    :param conceptsORchains: Typologie ('concepts' ou 'chains').
    :param alias_model: Identifiant court du modèle d'embedding ('allMin', 'distiluse', 'e5-large').
    :param avec_these: Périmètre documentaire ('only_mono', 'only_theses', 'with_theses').
    :param csv_filename_in: Nom ou chemin optionnel du fichier CSV d'entrée.
    """
    cwd = str(Path.cwd())
    print(cwd)

    # Répertoire de stockage des fichiers .pkl et .ini (standardisé à /app/data/pkl)
    pkl_dir = '/app/data/pkl'
    try:
        os.makedirs(pkl_dir, exist_ok=True)
    except OSError:
        # Fallback pour exécution locale hors conteneur sans droits sur /app
        pkl_dir = './data/pkl'
        os.makedirs(pkl_dir, exist_ok=True)
    root = os.path.join(pkl_dir, '')
    root1 = pkl_dir
    print(f"Répertoire de stockage des fichiers PKL : {pkl_dir}")
    adress_qdrant=config.QDRANT_HOST
    port_qdrant=config.QDRANT_PORT
    action=action_in
    #laisser a true pour que les updates puissent passer
    nettoie_file_trv=True

    init_csv_name = getattr(config, 'CSV_INIT_FILENAME', 'export_rameau.csv')
    update_csv_name = getattr(config, 'CSV_UPDATE_FILENAME', 'export_rameau_update.csv')
    if csv_filename_in:
        if action == "update":
            update_csv_name = csv_filename_in
        else:
            init_csv_name = csv_filename_in

    # Résolution des chemins de fichiers CSV pour vérifier les dates
    path_init_file = resolve_csv_path(init_csv_name)
    path_update_file = resolve_csv_path(update_csv_name)

    train_filename = "export_rameau"
    if csv_filename_in:
        train_filename = os.path.splitext(os.path.basename(csv_filename_in))[0]
    elif action == "update":
        train_filename = os.path.splitext(os.path.basename(update_csv_name))[0]
    else:
        train_filename = os.path.splitext(os.path.basename(init_csv_name))[0]

    #si 100 alors on ne decoupe pas, si 10 alors on decoupe par (nb fichier:82) % 10 soit 8 fichiers de moyennes qui seront ensuite
    # re-moyennises en 1 seul
    #pour plus de prcision et si assez de memoire alors mettre 100
    nb_lots_de_moyenne=100
    
    bibprepared_filename = train_filename+"_"+conceptsORchains+'_'+alias_model+'_'+avec_these
    
    RameauVectors = conceptsORchains+'_'+alias_model+'_'+avec_these
    try:
        date_init_file = str(datetime.fromtimestamp(os.path.getmtime(path_init_file)))
    except:
        date_init_file = f"pas de fichier {path_init_file}"
    print(f"date file init ({path_init_file}): {date_init_file}")
    try:
        date_update_file = str(datetime.fromtimestamp(os.path.getmtime(path_update_file)))
    except:
        date_update_file = f"pas de fichier {path_update_file}"
    print(f"date file update ({path_update_file}): {date_update_file}")
    config_init = configparser.ConfigParser()
    try:
        config_init.read(root+bibprepared_filename+'_init.ini')
        date_init_file_last=config_init['DEFAULT']['date_mod']
    except:
        date_init_file_last="null"
        config_init['DEFAULT'] = {'date_mod': 'null'}
        with open(root+bibprepared_filename+'_init.ini', 'w') as configfile:
            config_init.write(configfile)
    try:
        config_init.read('global_init.ini')
        if 'adress_qdrant' in config_init['DEFAULT']:
            adress_qdrant=config_init['DEFAULT']['adress_qdrant']
    except:
        pass
    
    print("adress_qdrant:"+adress_qdrant)
    print("port_qdrant:"+str(port_qdrant))
    
    print("last date file init:"+date_init_file_last)
    config_update = configparser.ConfigParser()
    try:
        config_update.read(root+bibprepared_filename+'_update.ini')
        date_update_file_last=config_update['DEFAULT']['date_mod']
    except:
        date_update_file_last="null"
        config_update['DEFAULT'] = {'date_mod': 'null'}
        with open(root+bibprepared_filename+'_update.ini', 'w') as configfile:
            config_update.write(configfile)
    
    print("last date file update:"+date_update_file_last)
    update=True
    #ICI update ou pas
    if (action=="init"):
        print("mode init")
        update=False
    if (action=="update"):
        print("mode update")
        update=True
    if (action=="auto"):
        if date_init_file != date_init_file_last:
            action="init"
            print("mode init")
        else:
            if date_update_file != date_update_file_last:
                print("mode update")
                action="update"
                update=True
            else:
                print("rien a traiter on quitte")
                del update
                #exit()
    size_csv=0
    size_csv_update=0
    size_origine=0
    size_origine_new=0
    # In[6]:
    
    if action =="init" or action =="update" or action =="restore"  :
        # Instancier le modèle de l'encodeur de phrases,
        
        #emb_model = 'intfloat/multilingual-e5-small'   # pas mal du tout, à réévaluer
        #les fichiers update doivent finir par _update et porter le meme nom initial que le fichier initial
        if csv_filename_in:
            csv_name = csv_filename_in
            train_filename = os.path.splitext(os.path.basename(csv_filename_in))[0]
        elif update==True:
            csv_name = getattr(config, 'CSV_UPDATE_FILENAME', 'export_rameau_update.csv')
            train_filename = os.path.splitext(os.path.basename(csv_name))[0]
        else:
            csv_name = getattr(config, 'CSV_INIT_FILENAME', 'export_rameau.csv')
            train_filename = os.path.splitext(os.path.basename(csv_name))[0]
            
        if (alias_model=='allMin'):
            emb_model = 'all-MiniLM-L6-v2'
        if (alias_model=='distiluse'):
            emb_model = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
        if (alias_model=='e5-large'):
            emb_model = 'intfloat/multilingual-e5-large'
        
        #encoder = SentenceTransformer(emb_model, device='cuda')  # pas mal du tout, à réévaluer, cbd ajout cuda
        encoder = SentenceTransformer(emb_model)  # pas mal du tout, à réévaluer, cbd ajout cuda
    
    
    # In[7]:
    
    if action =="init" or action =="update"  :
        
        
        def encode_concepts(text):
            res = text.replace(' -- ',';') 
            return res
        
        columns = ['PPN','THESE','TITRE','RESUME','RAMEAU','lang']
        #chargement fichier tabule
        resolved_csv_path = resolve_csv_path(csv_name)
        print("")
        print("")
        print("")
        print("")
        print(f"chargement fichier tabulé: {resolved_csv_path} et preparation")
        training_data=pd.read_csv(resolved_csv_path,sep='\t',header=None, names=columns)
        #supprime mono
        if (avec_these =='only_theses'):
            training_data=training_data[training_data.THESE.notnull()]
        #supprime theses
        if (avec_these =='only_mono'):
            training_data=training_data[training_data.THESE.isnull()]
        #on prepare pour pouvoir spliter 
        if (conceptsORchains == 'concepts'):
            training_data['RAMEAU'] = training_data['RAMEAU'].apply(encode_concepts)
        training_data['TITRE'] = training_data['TITRE'].apply(lambda x: x.replace("\x98","").replace("\x9c", ""))
        def encode_special(text):
            res = text.replace('_','#u#').replace('"','#d#').replace('\'','#c#').replace(' ','_').replace(',','_').replace(';',',') 
            return res
        training_data['RAMEAU'] = training_data['RAMEAU'].apply(encode_special)
        training_data["DESCR"] = training_data['TITRE'].astype(str) +", "+ training_data["RESUME"]
        print("")
        print("")
        print("")
        print("")
        print("sauvegarde dans fichier: "+bibprepared_filename+'.pkl')
        training_data.to_pickle(root+bibprepared_filename+'.pkl')
        size_csv=len(training_data)
        
        
        training_data.info()
        print(training_data[training_data['PPN']== '257658238'])
        #del training_data
        
        
        # In[8]:
        
        print("")
        print("")
        print("")
        print("")
        print("chargement fichier prepare:"+bibprepared_filename+'.pkl')
        training_data = pd.read_pickle(root+bibprepared_filename+'.pkl')
        training_data.info()
        training_data = training_data[['PPN','DESCR', 'RAMEAU']]
        
        if len(training_data.index) <2:
            print("les données a charger sont vides on quit")
            quit()
        # ## Calcul des embeddings
        
        # In[9]:
        
        
        start_time = datetime.now() 
        
        # Fonction pour encoder les données en paquets
        batch_size = 5000
        
        # Fonction pour encoder les données en paquets
        def encode_batches(data, encoder, batch_size):
            print(len(data))
            # Initialiser une liste pour stocker les batches encodés
            encoded_batches = []
            # Calculer le nombre total de batches nécessaires pour traiter toutes les données
            num_batches = int(np.ceil(len(data) / batch_size))
            print(num_batches)
        
            # Parcourir chaque batch
            for i in range(num_batches):
                # Sélectionner les données pour le batch actuel
                #print("encode",i)
                batch_data = data.iloc[i * batch_size: (i + 1) * batch_size]
                # Encoder le texte dans le batch actuel, en montrant une barre de progression
                #print("encode2",i)
                encoded_batch = encoder.encode(batch_data['DESCR'].tolist(), show_progress_bar=False)
                # Ajouter le batch encodé à la liste des batches encodés
                encoded_batches.append(encoded_batch)
            # Concaténer tous les batches encodés en un seul tableau numpy
            return np.concatenate(encoded_batches)
        
        
        
        
        
        # In[10]:
        print("")
        print("")
        print("")
        print("")
        
        print("calcul embedding...calcul par lots de 5000")
        start_time = datetime.now() 
        #nettoyage
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        
        bt_size=5000 # meilleur rendement
        nb_tours=len(training_data)//bt_size
        
        print("nb tours:",nb_tours)
        df4 = training_data.iloc[0:bt_size].reset_index()
        #premier lot
        encoded_articles = encode_batches(df4, encoder, bt_size)
        df4['content_vector'] = pd.Series(encoded_articles.tolist())
        df4 = df4.drop(columns=['DESCR'])
        pos_file=1
        with open(root+'df_all_'+RameauVectors+'_'+str(pos_file)+'.pkl','wb') as f:
         pickle.dump(df4,f)
        
        
        end_time = datetime.now() 
        time_difference = (end_time - start_time).total_seconds() * 10**3
        print("Temps d execution de l embedding pour le premier lot de 5000: ", time_difference, "ms") 
        
        # lots suivants
        for i in range(1,nb_tours+1):
            print((i*bt_size)+1,(i*bt_size)+bt_size)
            df4 = training_data.iloc[(i*bt_size)+1:(i*bt_size)+bt_size].reset_index()
            #on relance l encoder car au bout d un moment il ralenti
            if i==50 or i==100  or i==150 :
                #encoder = SentenceTransformer(emb_model, device='cuda')
                encoder = SentenceTransformer(emb_model)
            encoded_articles = encode_batches(df4, encoder, bt_size)
            # Ajouter les vecteurs encodés en tant que nouvelle colonne 'content_vector' dans le DataFrame
            df4['content_vector'] = pd.Series(encoded_articles.tolist())
            
            df4 = df4.drop(columns=['DESCR'])
            pos_file=pos_file+1
            #sauvegarde des 5000 vectorise
            with open(root+'df_all_'+RameauVectors+'_'+str(pos_file)+'.pkl','wb') as f:
                pickle.dump(df4,f)
                    
            end_time = datetime.now() 
            time_difference = (end_time - start_time).total_seconds() * 10**3
            print("                             cumul Temps d execution de l embedding apres le lot suivant: ", time_difference, "ms") 
        
        end_time = datetime.now() 
        time_difference = (end_time - start_time).total_seconds() * 10**3
        print("Temps d execution total de l embedding : ", time_difference, "ms") 
        
        #os.remove(df_file_path)
        
        
        # In[11]:
    
    #si on est en init (on recharge tout)
    if action =="init"   :
        print("")
        print("")
        print("")
        print("")
        print("en mode init : fusion et sauvegarde des fichiers vectorisés en : "+RameauVectors+".pkl")
        
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        pos=0
        #concatene en u seul fichier
        for file in list_of_files:
                    #print(file)
                    df4 = pd.read_pickle(file)
                    if pos==0:
                        result = df4
                    else:
                        result = pd.concat([result, df4], ignore_index=True)
                    pos=pos+1
        size_csv_update=len(result)
        
        size_origine_new=len(result)
        if size_csv_update*2 < size_csv:
            print("en mode init : le resultat des embedding est incoherent avec la taille du fichier d origine :"+str(size_csv)+" et le fichier de embeddings : "+str(size_csv_update)+" on quit")
            quit()
        with open(root+RameauVectors+'.pkl','wb') as f:
                pickle.dump(result,f) 
    
    
    # In[12]:
    
    
    #del result
    # si on est en update
    if action =="update"  :
        print("")
        print("")
        print("")
        print("")
        print("en mode update" )
        if os.path.exists(root+RameauVectors+'.pkl')==False:
            print("le fichier global: "+RameauVectors+".pkl n existe pas, il a ete detruit ou un init n'as pas ete fait au part avant,  on  quit")
            quit()
        #recup tous les vecteur par ppn
        print("")
        print("en mode update : chargement du fichier global init vectorisé : "+root+RameauVectors+".pkl")
        df4 = pd.read_pickle(root+RameauVectors+'.pkl')
        #df4 = pickle.load(root+RameauVectors+'.pkl')
        #with open(root+RameauVectors+'.pkl', 'rb') as pickle_file:
        #    df4 = pickle.load(pickle_file)
        
        size_origine=len(df4) 
        
        if size_origine <10:
            print("les vecteurs sauvegardes dans le fichier origine :"+RameauVectors+".pkl sont inferieur a 10 il y a probablement un pb on  quit")
            quit()
        #recup des fichiers vectors d update
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        pos=0
        print("")
        print("en mode update : chargement des fichiers de mise a jours vectorises : "+'df_all_'+RameauVectors+'*.pkl')
        #concat des nouveau vecteurs
        for file in list_of_files:
                    print("               chargement fichier de mise a jour :")
                    print(file)
                    df = pd.read_pickle(file)
                    if pos==0:
                        result = df
                    else:
                        result = pd.concat([result, df], ignore_index=True)
                    pos=pos+1
        #suppression ajout 
        print("en mode update : suppression des ppn du fichier origine se trouvant dans le fichier d update")
        df4 = df4[~df4['PPN'].isin(result['PPN'])]
        #ajout
        print("en mode update : ajout des ppn du fichier d update dans le fichier origine nettoye")
        df4=pd.concat([df4, result], ignore_index=True)
        size_origine_new=len(df4)
        
        if size_origine_new*2 < size_origine:
            print("le nombre total de vecteurs fusionnés "+str(size_origine_new)+" est tres inferieur au fichier origine :"+str(size_origine)+" , il y a probablement un pb on  quit pour eviter de veroler le fichier origine :"+RameauVectors+".pkl")
            quit()
        #sauvegarde tous les nouveau vecteurs
        print("mise a jour du fichier global init :"+RameauVectors+".pkl")
        #df4.to_pickle(root+RameauVectors+'.pkl')
        with open(root+RameauVectors+'.pkl','wb') as f:
                pickle.dump(df4,f)    
        
        
        #nettoyage
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        #tronconne
        print("")
        print("en mode update : generation des fichiers de vecteurs : df_all_ ,par lot de 5000")
        bt_size=5000 # meilleur rendement
        nb_tours=len(df4)//bt_size
    
        for i in range(0,nb_tours+1):
            #print((i*bt_size)+1,(i*bt_size)+bt_size)
            resu = df4.iloc[(i*bt_size)+1:(i*bt_size)+bt_size].reset_index(drop=True)
            with open(root+'df_all_'+RameauVectors+'_'+str(i)+'.pkl','wb') as f:
                pickle.dump(resu,f)
    
    
    # In[13]:
    if action =="init" or action =="update"  :
        print("")
        print("")
        print("")
        print("")
        print("fusion des fichiers de travail df_all_ en lots plus grands (de 1 fichier si possible, cela depend de la variable nb_lots_de_moyenne) et sauvegarde dans les fichiers de travail: df_resu_")
        
        
        list_of_files = Path(root1).glob('df_resu_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        pos=0
        
        for file in list_of_files:
                    #print(file)
                    df4 = pd.read_pickle(file)
                    if pos==0:
                        result = df4
                    else:
                        result = pd.concat([result, df4], ignore_index=True)
                    if (pos  % nb_lots_de_moyenne == 0) and pos>0 :
                        #print('df_resu'+str(pos)+'.pkl')
                        with open(root+'df_resu_'+RameauVectors+'_'+str(pos)+'.pkl','wb') as f:
                            pickle.dump(result,f)
                        result = result.head(0)
        
                    pos=pos+1
        with open(root+'df_resu_'+RameauVectors+'_'+str(pos)+'.pkl','wb') as f:
                pickle.dump(result,f)            
        print("")
        print('echantillon:')
        result.info()
         
        
        
        # In[14]:
        
        del result
        #regroupement pour chaque fichier
        print("")
        print("")
        print("")
        print("")
        print("Pour chaque fichiers de travail df_resu_ : explosion par labels, puis regroupement par moyennes des vecteurs puis ecriture des resultats dans des  fichiers de travail : df_group_")
       
        list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        
        list_of_files = Path(root1).glob('df_group_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        pattern = r','    
        list_of_files = Path(root1).glob('df_resu_'+RameauVectors+'*.pkl')
        pos=0
        
        for file in list_of_files:
            print("")
            print("                       Fichier traité:")
            print(file)
            result = pd.read_pickle(file)
            result.dropna(subset = ['content_vector'], inplace=True)
            #df4 = df4.rename(columns={'RAMEAU_chains2': 'RAMEAU_list'})
            result = result.rename(columns={'RAMEAU': 'RAMEAU_list'})
            result['RAMEAU_list'] = result['RAMEAU_list'].apply(lambda x: re.split(pattern, x))  
            result = result.explode('RAMEAU_list')
            
            
            label_vectors = result.groupby('RAMEAU_list').agg(mean=('content_vector', lambda x: np.vstack(x).mean(axis=0).tolist()))
            label_vectors['target'] = label_vectors.index
            label_vectors.columns = ['content_vector', 'label']
            with open(root+'df_group_'+RameauVectors+'_'+str(pos)+'.pkl','wb') as f:
                pickle.dump(label_vectors,f)
            pos=pos+1
            #del label_vectors
        
        
        
        # In[15]:
        
        
        #concatenation des fichiers regroupes si on a plusieurs fichiers de travail
        if pos>1:
            print("")
            print("")
            print("")
            print("")
            print("si il y a plusieurs fichiers de travail df_group_ alors on les regroupe dans un dataframe label_vectors:")
           
            #if update==False:
            list_of_files = Path(root1).glob('df_group_'+RameauVectors+'*.pkl')
            #else:
            #    list_of_files = Path(root1).glob('df_group_'+RameauVectors+'*.pkl') # pour prendre aussi les fichiers du chargement initial
            pos=0
            
            for file in list_of_files:
                print(file)
                if pos==0:
                    label_vectors=pd.read_pickle(file)
                else:
                    label_vectors = pd.concat([label_vectors,pd.read_pickle(file)], ignore_index=True)
                pos=pos+1
            label_vectors.info()
        
        
        
        #moyenne sur les fichiers regroupes si plusieurs fichiers de groupement
        if pos>1:
            print("")
            print("")
            print("")
            print("")
            print("si il y a eu plusieurs fichiers de travail df_group_ alors on re-calcul les moyennes des vecteurs du dataframe label_vectors:")
           
            label_vectors = label_vectors.groupby('label').agg(mean=('content_vector', lambda x: np.vstack(x).mean(axis=0).tolist()))
            label_vectors['target'] = label_vectors.index
            label_vectors.columns = ['content_vector', 'label']
            #del result
            label_vectors.info()
        
        
        # In[19]:
        
        print("")
        print("")
        print("")
        print("")
        print("a partir du dataframe label_vectors, generations des fichiers pour qdrant: df_emb_ , par lots de 10000 :")
       
        list_of_files = Path(root1).glob('df_emb_'+RameauVectors+'*.pkl')
        for file in list_of_files:
            os.remove(file)
        
        pattern = r';\s*(?![^()]*\))|--'
        #pattern = r';'
        chunks = [label_vectors.iloc[i:i+10000] for i in range(0, len(label_vectors), 10000)]
        pos=0
        for chunk in chunks:
            print("")
            print("                        LOT info:")
            chunk.info()
            pos=pos+1
            label_vectors_trv=chunk.copy()
            
            label_vectors_trv.to_pickle(root+'df_emb_'+RameauVectors+'_'+str(pos)+'.pkl')
    
        list_of_files = Path(root1).glob('df_emb_'+RameauVectors+'*.pkl')
        pos=0
        for file in list_of_files:
            #print(file)
            df4 = pd.read_pickle(file)
            if pos==0:
                result = df4
            else:
                result = pd.concat([result, df4], ignore_index=True)
            pos=pos+1
        print("")
        print("")
        print("Fichier vecteur resultat pour qdrant :")
        result.info()
        if len(result.index)*20 <size_origine_new:
                print("le nombre de  vecteurs :"+str(len(result.index))+ "*10 est inferieur au vecteurs origine:  "+ str(size_origine_new)+" ce qui est trop peut pour rameau (environ 40000) il y a probablement un pb on  quit le chargement dans qdrant pour ne pas detruire la version existante")
                quit() 
        #creation fichier de sauvegarde
        print("Sauvegarde du Fichier vecteur resultat pour qdrant dans :archive_"+RameauVectors+'.pkl')
        result.to_pickle(root+'archive_'+RameauVectors+'.pkl')

    # si une des actions
    if action =="init" or action =="update" or action =="restore" :
        print("")
        print("")
        print("")
        print("")
        print("pour qdrant recuperation du dernier fichier archive généré:")
        print("archive_"+RameauVectors+'.pkl')
        if os.path.exists(root+'archive_'+RameauVectors+'.pkl')==False:
            print("le fichier des vecteurs : 'archive_'+"+RameauVectors+".pkl n existe pas on  quit")
            quit()
        print(root+'archive_'+RameauVectors+'.pkl')
        result = pd.read_pickle(root+'archive_'+RameauVectors+'.pkl')
        
        
        
        min_vectors = int(getattr(config, 'QDRANT_MIN_VECTORS', 1000))
        if len(result.index) < min_vectors:
                print(f"le nombre de vecteurs ({len(result.index)}) est inferieur au seuil minimal configuré ({min_vectors}) ce qui est trop peu pour rameau (environ 40000). Abandon du chargement dans qdrant pour ne pas detruire la version existante.")
                quit() 
        
        # ## Initialisation d'une instance Qdrant et création d'une collection pour stocker les données
        client = QdrantClient(host=adress_qdrant, port=port_qdrant)
        # In[ ]:
        
        print("")
        print("")
        print("")
        collection_name = conceptsORchains+'_'+alias_model+'_'+avec_these
        print("")
        print("qdrant delete collection..")
        print(collection_name)
        client = QdrantClient(host=adress_qdrant, port=port_qdrant)

        # Tentative de suppression de la collection avec repli sur la purge des points en cas de verrou système (Windows/NTFS)
        if client.collection_exists(collection_name):
            try:
                client.delete_collection(collection_name=collection_name)
            except Exception as del_err:
                print(f"delete_collection a échoué ({del_err}). Repli sur le vidage des points existants...")
                try:
                    client.delete(
                        collection_name=collection_name,
                        points_selector=models.FilterSelector(filter=models.Filter())
                    )
                    print(f"Collection '{collection_name}' vidée avec succès.")
                except Exception as clear_err:
                    print(f"Avertissement lors du vidage des points: {clear_err}")

        from qdrant_client.models import VectorParams, Distance
        embedding_dim = encoder.get_embedding_dimension() if hasattr(encoder, "get_embedding_dimension") else encoder.get_sentence_embedding_dimension()
        if not client.collection_exists(collection_name):
            print("qdrant create collection..")
            print(collection_name)
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE, on_disk=True),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8,
                            always_ram=True,
                        ),
                    ),
                )
            except Exception as create_err:
                if "already exists" in str(create_err).lower():
                    print(f"La collection existe déjà sur le disque ({create_err}), continuation...")
                else:
                    raise create_err
        
        
        # In[ ]:
        
        
        print("")
        print("qdrant chargement du dataframe result")
        points = [
            models.PointStruct(
                id=idx,
                vector=list(doc[0]),
                payload=dict.fromkeys("aaa", doc[1])
            ) for idx, doc in enumerate(result.values)
        ]

        if hasattr(client, "upload_points"):
            client.upload_points(
                collection_name=conceptsORchains+'_'+alias_model+'_'+avec_these,
                points=points
            )
        else:
            client.upload_records(
                collection_name=conceptsORchains+'_'+alias_model+'_'+avec_these,
                records=points
            )
    
        
        
        # In[ ]:
        if action =="init" or action =="update":
            print("")
            print("")
            print("")
            print("")
            print("nettoie fichiers de travail..")
            if nettoie_file_trv==True:
                os.remove(root+bibprepared_filename+'.pkl')
                list_of_files = Path(root1).glob('df_all_'+RameauVectors+'*.pkl')
                for file in list_of_files:
                    os.remove(file)
                list_of_files = Path(root1).glob('df_resu_'+RameauVectors+'*.pkl')
                for file in list_of_files:
                    os.remove(file)
                list_of_files = Path(root1).glob('df_group_'+RameauVectors+'*.pkl')
                for file in list_of_files:
                    os.remove(file)
                list_of_files = Path(root1).glob('df_emb_'+RameauVectors+'*.pkl')
                for file in list_of_files:
                    os.remove(file)
                
            if update==True:
                config_update['DEFAULT']['date_mod']=date_update_file
                with open(root+bibprepared_filename+'_update.ini', 'w') as configfile:
                    config_update.write(configfile)
            if update==False:
                config_init['DEFAULT']['date_mod']=date_init_file
                with open(root+bibprepared_filename+'_init.ini', 'w') as configfile:
                    config_init.write(configfile)
                
        
        
        # ## Prédiction des labels
        
        # In[ ]:
        print("")
        print("")
        print("")
        print("")
        test_limit = int(getattr(config, 'QDRANT_TEST_LIMIT', 6))
        print(f"test requete sur : la radio (limit={test_limit})")
        client = QdrantClient(host=adress_qdrant, port=port_qdrant)
        collection = conceptsORchains+'_'+alias_model+'_'+avec_these
        query_vector = encoder.encode("la radio").tolist()

        if hasattr(client, "query_points"):
            response = client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=test_limit
            )
            hits = response.points
        elif hasattr(client, "search"):
            hits = client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=test_limit
            )
        else:
            hits = []

        load_items = []
        for hit in hits:
            load_items.append({'score': hit.score, 'label': hit.payload})
        
        print(load_items)


# In[ ]:


def main():
    """
    Point d'entrée CLI du script de vectorisation batch.
    Gère la validation des options, la redirection des logs et le lancement du pipeline.
    """
    print ('MAIN')
    try:
        p = optparse.OptionParser()
        p.add_option('--action', '-a', default="update")
        p.add_option('--conceptsORchains', '-c', default="concepts")
        p.add_option('--alias_model', '-m', default="allMin")
        p.add_option('--avec_these', '-t', default="only_mono")
        p.add_option('--csv_filename', default="")
        options, arguments = p.parse_args()
        ok =True
        if options.action not in ['init','update','auto','restore']:
            print ('action doit avoir :init, update ou auto')
            ok =False
            quit()
        if options.conceptsORchains not in ['concepts','chains']:
            print ('conceptsORchains doit avoir :concepts ou chains')
            ok =False
            quit() 
        if options.alias_model not in ['allMin','distiluse','e5-large']:
            print ('alias_model doit avoir :allMin, distiluse ou e5-large')
            ok =False
            quit()
        if options.avec_these not in ['only_mono','only_theses','with_theses']:
            print ('avec_these doit avoir :only_mono, only_theses ou with_theses')
            ok =False
            quit()

        if ok==True:
            launch(options.action, options.conceptsORchains, options.alias_model, options.avec_these, options.csv_filename)
    except BaseException as e:
        print('Failed to do something: ' + str(e))
        raise

if __name__ == '__main__':
  main()

