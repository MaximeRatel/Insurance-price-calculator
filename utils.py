import numpy as np
import pandas as pd

def OneHotEncoding(df, encoder):
  """ Encode un dataframe et trie les données par index croissant (par commodité)"""
  
  list_num = ['PERMIS','ACV','AGECOND','RM','CLA','VIT','GARAGE']
  list_cat = ['SEX','STATUT','CSP','USAGE','K8000','CAR','ALI','ENE','SEGM']
  
  "On isole les variables numériques"
  df_num = df[list_num].sort_index()

  "On encode les variables categorielles"
  df_cat = df[list_cat]
  df_cat = encoder.transform(df_cat)
  df_cat = pd.DataFrame(df_cat, columns=encoder.get_feature_names())
  df_cat = df_cat.set_index(df_num.index)

  "On retourne le jeu de données ré-assemblé"
  return pd.concat([df_num, df_cat],axis=1)
