"""
Script de validation d'environnement ("Pre-flight Check") pour le pipeline de vectorisation.

Ce script vérifie la conformité des arguments passés en ligne de commande 
et s'assure que la mémoire RAM libre sur la machine est suffisante
avant de charger les gros fichiers de modèles.
"""
#!/usr/bin/env python
# coding: utf-8



# la procedure oracle de traitement sur la base xml est QE_EXPORT_RAMEAU
# pour initier les serveurs ; 
# cd /opt ; source jupyter/bin/activate ; cd /home/ubuntu/cbd
# lancer le service de chargement et de lancement de la vectorisation il ecoute sur le port 8100:
# python load_qdrant_ws.py 
# url de test 
# http://SERVEUR_DE_TEST_IP:8100/lanceVectorisation/?action=update1&conceptsORchains=concepts&alias_model=allMin&avec_these=only_mono

# lancer le service wided de test il ecoute sur le port 8068:
# cd /opt ; source jupyter/bin/activate ; cd /home/ubuntu/yann/webservice
# python wided-test_cbd3.py
#pour tester http://SERVEUR_DE_TEST_IP:8068/subject_indexation/?docId=NULL&Title=la%20radio&Summary=&models=victor1_concept,victor2,victor3_chain&aggregationType=intersection2models,llm&subjects&MaxCount=6&Agent=RCR&vocabulary=rameau&Format=text

import optparse
import pandas as pd



# In[5]:



def main():
    print ('MAIN sur non docker')
    p = optparse.OptionParser()
    p.add_option('--action', '-a', default="update")
    p.add_option('--conceptsORchains', '-c', default="concepts")
    p.add_option('--alias_model', '-m', default="allMin")
    p.add_option('--avec_these', '-t', default="only_mono")
    options, arguments = p.parse_args()
    ok =True
    if options.action not in ['init','update','auto','restore']:
        print ('action doit avoir :init, update ou auto')
        ok =False
        
    if options.conceptsORchains not in ['concepts','chains']:
        print ('conceptsORchains doit avoir :concepts ou chains')
        ok =False
         
    if options.alias_model not in ['allMin','distiluse','e5-large']:
        print ('alias_model doit avoir :allMin, distiluse ou e5-large')
        ok =False
        
    if options.avec_these not in ['only_mono','only_theses','with_theses']:
        print ('avec_these doit avoir :only_mono, only_theses ou with_theses')
        ok =False
    print ('df_all_concepts_allMin_only_mono_1.pkl')
    training_data = pd.read_pickle('df_all_concepts_allMin_only_mono_1.pkl')
    '''total_memory, used_memory, free_memory = map(
	int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print("ram dispo:")
    print(free_memory)
    if (options.alias_model=='e5-large' and free_memory<10866 and options.action!='restore'):
        print("Pas assez de ram pour ce model, on quit")
        quit()
    if (free_memory<7866):
        print("Pas assez de ram pour les petits model, on quit")
        quit()'''
    
    if ok==True:
        print(options.action, options.conceptsORchains, options.alias_model,options.avec_these)

if __name__ == '__main__':
  main()

