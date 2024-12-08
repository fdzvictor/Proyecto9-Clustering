#IMPORTS
#--------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import math

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler

#--------------------------------------------------------------------------------------------------------

def separar_categorias(dataframe):
   return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = "O")


#--------------------------------------------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np



# Comenzamos con las núméricas:
def escalado_robusto(dataframe, columnas):
    
    df_robust = pd.DataFrame()

    #Robustas
    escalador_robust = RobustScaler()


    for columna in columnas:
        datos_transf_robust = escalador_robust.fit_transform(dataframe[columna])  
        df_robust[f"{columna}_robust"] = datos_transf_robust  
    
    return df_robust

def escalado_minmax(dataframe, columnas):
    
    df_minmax = pd.DataFrame()

    #Minmax
    escalador_min_max = MinMaxScaler()

    # Iteramos sobre las columnas para aplicar el escalado
    for columna in columnas:
        datos_transf_min_max = escalador_min_max.fit_transform(dataframe[[columna]])  
        df_minmax[f"{columna}_minmax"] = datos_transf_min_max
    
    return df_minmax

def escalado_standard(dataframe):
    df_num = dataframe.select_dtypes(include = np.number).columns
    df_standard = pd.DataFrame()

    #Standard
    escalador_stand = StandardScaler()

    for columna in df_num:
        datos_transf_stand = escalador_stand.fit_transform(df_num[[columna]])  
        df_num[f"{columna}_stand"] = datos_transf_stand

    return df_standard


#--------------------------------------------------------------------------------------------------------

def escalado_normalizer(dataframe,columnas):

    df_vacio = pd.DataFrame()

    escalador_normal = Normalizer()


    for columna in columnas:
        datos_transf_normal = escalador_normal.fit_transform(dataframe[[columna]])  
        df_vacio[f"{columna}_normal"] = datos_transf_normal  

#--------------------------------------------------------------------------------------------------------

def identificar_outliers_iqr (dataframe, columnas_numericas, k = 1.5):
    diccionario_outliers = {}
    for columna in columnas_numericas:
        Q1, Q3 = np.nanpercentile(dataframe[columna], (25,75))
        iqr = Q3-Q1
        limite_superior = Q3 + (iqr*k)
        limite_inferior = Q1 - (iqr*k)

        condicion_superior = dataframe[columna] > limite_superior
        condicion_inferior = dataframe[columna] < limite_inferior

        df_outliers = dataframe[condicion_inferior | condicion_superior]
        print(f"la columna {columna} tiene {df_outliers.shape[0]} outliers")


        if not df_outliers.empty:
            diccionario_outliers[columna] = df_outliers
            

    return diccionario_outliers
    

#--------------------------------------------------------------------------------------------------------

def outliers_zscore (dataframe, lista_columnas):
    diccionario_outliers = {}
    for columna in lista_columnas:
        condicion_zscore = abs(zscore(dataframe[columna])) >= 3
        df_outliers = dataframe[condicion_zscore]

        if not df_outliers.empty:
            diccionario_outliers[columna] = df_outliers

    return df_outliers
