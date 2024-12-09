import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
# import plotly_express as px


# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
# from pyod.models.mad import MAD # para calcula la desviación estandar absoluta
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon
from sklearn.cluster import DBSCAN # para usar DBSCAN

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product

# Gestionar warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')


def exploracion_dataframe(dataframe, columna_control):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ..................... \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ..................... \n")
    
    
    # generamos un DataFrame para los valores nulos
    print("Los nulos que tenemos en el conjunto de datos son:")
    df_nulos = pd.DataFrame(dataframe.isnull().sum() / dataframe.shape[0] * 100, columns = ["%_nulos"])
    display(df_nulos[df_nulos["%_nulos"] > 0])
    
    print("\n ..................... \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    
    
    print("\n ..................... \n")
    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = dataframe.select_dtypes(include = "O")
    
    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        display(pd.DataFrame(dataframe[col].value_counts()).head())    
    
    # como estamos en un problema de A/B testing y lo que realmente nos importa es comparar entre el grupo de control y el de test, los principales estadísticos los vamos a sacar de cada una de las categorías
    
    for categoria in dataframe[columna_control].unique():
        dataframe_filtrado = dataframe[dataframe[columna_control] == categoria]

        if type(categoria) == str:
    
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas categóricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe(include = "O").T)
            
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para el {categoria.upper()} son: ")
            display(dataframe_filtrado.describe().T)
        
        else:
            
            print("\n ..................... \n")
            print(f"Los principales estadísticos de las columnas numéricas para {categoria} son: ")
            display(dataframe_filtrado.describe().T)

class Visualizador:

    """
    Clase para visualizar la distribución de variables numéricas y categóricas de un DataFrame.

    Attributes:
    - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.

    Methods:
    - __init__: Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.
    - separar_dataframes: Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.
    - plot_numericas: Grafica la distribución de las variables numéricas del DataFrame.
    - plot_categoricas: Grafica la distribución de las variables categóricas del DataFrame.
    - plot_relacion2: Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.
    """

    def __init__(self, dataframe):
        """
        Inicializa el VisualizadorDistribucion con un DataFrame y un color opcional para las gráficas.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe

    def separar_dataframes(self):
        """
        Separa el DataFrame en dos subconjuntos, uno para variables numéricas y otro para variables categóricas.

        Returns:
        - pandas.DataFrame: DataFrame con variables numéricas.
        - pandas.DataFrame: DataFrame con variables categóricas.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")
    
    def plot_numericas(self, palette="grey", tamano_grafica=(15, 5)):
        """
        Grafica la distribución de las variables numéricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
        lista_num = self.separar_dataframes()[0].columns
        _, axes = plt.subplots(nrows = 2, ncols = math.ceil(len(lista_num)/2), figsize=tamano_grafica, sharey=True)
        axes = axes.flat
        for indice, columna in enumerate(lista_num):
            sns.histplot(x=columna, data=self.dataframe, ax=axes[indice], palette = palette, bins=40)
        plt.suptitle("Distribución de variables numéricas")
        plt.tight_layout()

    def plot_categoricas(self, palette="Set1", tamano_grafica=(40, 10)):
        """
        Grafica la distribución de las variables categóricas del DataFrame.

        Parameters:
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (15, 5).
        """
    
        dataframe_cat = self.separar_dataframes()[1]
        _, axes = plt.subplots(2, math.ceil(len(dataframe_cat.columns) / 2), figsize=tamano_grafica)
        axes = axes.flat
        for indice, columna in enumerate(dataframe_cat.columns):
            sns.countplot(x=columna, data=self.dataframe, order=self.dataframe[columna].value_counts().index,
                          ax=axes[indice], palette=palette)

            axes[indice].tick_params(rotation=90)
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.suptitle("Distribución de variables categóricas")

    def plot_relacion(self, vr, tamano_grafica=(40, 12), color="grey"):
        """
        Visualiza la relación entre una variable y todas las demás, incluyendo variables numéricas y categóricas.

        Parameters:
            - vr (str): El nombre de la variable en el eje y.
            - tamaño_grafica (tuple, opcional): El tamaño de la figura de la gráfica. Por defecto es (40, 12).
            - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        Returns:
            No devuelve nada    
        """
        df_numericas = self.separar_dataframes()[0].columns
        meses_ordenados = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        fig, axes = plt.subplots(3, int(len(self.dataframe.columns) / 3), figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.columns):
            if columna == vr:
                fig.delaxes(axes[indice])
            elif columna in df_numericas:
                sns.scatterplot(x=vr, 
                                y=columna, 
                                data=self.dataframe, 
                                color=color, 
                                ax=axes[indice])
                axes[indice].set_title(columna)
                axes[indice].set(xlabel=None)
            else:
                if columna == "Month":
                    sns.barplot(x=columna, y=vr, data=self.dataframe, order=meses_ordenados, ax=axes[indice],
                                color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)
                else:
                    sns.barplot(x=columna, y=vr, data=self.dataframe, ax=axes[indice], color=color)
                    axes[indice].tick_params(rotation=90)
                    axes[indice].set_title(columna)
                    axes[indice].set(xlabel=None)

        plt.tight_layout()
    
    def analisis_temporal(self, var_respuesta, var_temporal, color = "black", order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):


        """
        Realiza un análisis temporal mensual de una variable de respuesta en relación con una variable temporal. Visualiza un gráfico de líneas que muestra la relación entre la variable de respuesta y la variable temporal (mes), con la línea de la media de la variable de respuesta.


        Params:
        -----------
        dataframe : pandas DataFrame. El DataFrame que contiene los datos.
        var_respuesta : str. El nombre de la columna que contiene la variable de respuesta.
        var_temporal : str. El nombre de la columna que contiene la variable temporal (normalmente el mes).
        order : list, opcional.  El orden de los meses para representar gráficamente. Por defecto, se utiliza el orden estándar de los meses.

        Returns:
        --------
        None

 
        """


        plt.figure(figsize = (15, 5))

        # Convierte la columna "Month" en un tipo de datos categórico con el orden especificado
        self.dataframe[var_temporal] = pd.Categorical(self.dataframe[var_temporal], categories=order, ordered=True)

        # Trama el gráfico
        sns.lineplot(x=var_temporal, 
                     y=var_respuesta, 
                     data=self.dataframe, 
                     color = color)

        # Calcula la media de PageValues
        mean_page_values = self.dataframe[var_respuesta].mean()

        # Agrega la línea de la media
        plt.axhline(mean_page_values, 
                    color='green', 
                    linestyle='--', 
                    label='Media de PageValues')


        # quita los ejes de arriba y de la derecha
        sns.despine()

        # Rotula el eje x
        plt.xlabel("Month");


    def deteccion_outliers(self, color = "grey"):

        """
        Detecta y visualiza valores atípicos en un DataFrame.

        Params:
            - dataframe (pandas.DataFrame):  El DataFrame que se va a usar

        Returns:
            No devuelve nada

        Esta función selecciona las columnas numéricas del DataFrame dado y crea un diagrama de caja para cada una de ellas para visualizar los valores atípicos.
        """

        lista_num = self.separar_dataframes()[0].columns

        fig, axes = plt.subplots(2, ncols = math.ceil(len(lista_num)/2), figsize=(15,5))
        axes = axes.flat

        for indice, columna in enumerate(lista_num):
            sns.boxplot(x=columna, data=self.dataframe, 
                        ax=axes[indice], 
                        color=color, 
                        flierprops={'markersize': 4, 'markerfacecolor': 'orange'})

        if len(lista_num) % 2 != 0:
            fig.delaxes(axes[-1])

        
        plt.tight_layout()

    def dataframe_correlación(self):
        def color_neg_pos(val):
            if val < -0.5 or val > 0.5:
                color = "yellow"
            elif val < 0:
                color = "red"
            else:
                color = "green"
            return f'color: {color}'
        
        df_num, df_cat = self.separar_dataframes()

        df_correlaciones = df_num.corr().T
        df_styled = df_correlaciones.style.applymap(color_neg_pos)
        
        return df_styled

    def heatmap_correlacion(self, tamano_grafica = (7, 5)):

        """
        Visualiza la matriz de correlación de un DataFrame utilizando un mapa de calor.

        Params:
            - dataframe : pandas DataFrame. El DataFrame que contiene los datos para calcular la correlación.

        Returns:
        No devuelve nada

        Muestra un mapa de calor de la matriz de correlación.

        - Utiliza la función `heatmap` de Seaborn para visualizar la matriz de correlación.
        - La matriz de correlación se calcula solo para las variables numéricas del DataFrame.
        - La mitad inferior del mapa de calor está oculta para una mejor visualización.
        - Permite guardar la imagen del mapa de calor como un archivo .png si se solicita.

        """

        plt.figure(figsize = tamano_grafica )

        mask = np.triu(np.ones_like(self.dataframe.corr(numeric_only=True), dtype = np.bool_))

        sns.heatmap(data = self.dataframe.corr(numeric_only = True), 
                    annot = True, 
                    vmin=-1,
                    vmax=1,
                    cmap="viridis",
                    linecolor="black", 
                    fmt='.1g', 
                    mask = mask)
    

class GestionOutliersUnivariados:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def _separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")



    def visualizar_outliers_univariados(self, color="blue", whis=1.5, tamano_grafica=(20, 15)):
        """
        Visualiza los outliers univariados mediante boxplots o histogramas.

        Parámetros:
        -----------
        color (str): Color de los gráficos.
        whis (float): Valor para definir el límite de los bigotes en los boxplots.
        tamano_grafica (tuple): Tamaño de la figura.
        """
        tipo_grafica = input("Qué gráfica quieres usar, Histograma (H) o Boxplot(B): ").upper()
        
        num_cols = len(self._separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        _, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.dataframe.select_dtypes(include=np.number).columns):
            if tipo_grafica == "B":
                sns.boxplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], whis=whis,
                            flierprops={'markersize': 4, 'markerfacecolor': 'orange'})
            elif tipo_grafica == "H":
                sns.histplot(x=columna, data=self.dataframe, color=color, ax=axes[indice], bins=50)
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None)

        plt.tight_layout()
        plt.show()

    def detectar_outliers_z_score(self):
        """
        Detecta outliers utilizando z-score.
        """
        diccionario_resultados_z = {}

        for columna in self._separar_variables_tipo()[0].columns:
            z_scores = abs(zscore(self.dataframe[columna]))
            diccionario_resultados_z[columna] = self.dataframe[z_scores > 3]
            print(f"La cantidad de outliers que tenemos para la columna {columna.upper()} es ", 
                  f"{diccionario_resultados_z[columna].shape[0]}")
        return diccionario_resultados_z

    def detectar_outliers_iqr(self, limite_outliers=1.5):
        """
        Detecta outliers utilizando el rango intercuartil (IQR).
        """
        diccionario_iqr = {}
        for columna in self._separar_variables_tipo()[0].columns:
            q1, q3 = np.nanpercentile(self.dataframe[columna], (25, 75))
            iqr = q3 - q1
            limite_inferior = q1 - limite_outliers * iqr
            limite_superior = q3 + limite_outliers * iqr
            df_outliers = self.dataframe[(self.dataframe[columna] < limite_inferior) | (self.dataframe[columna] > limite_superior)]
            if not df_outliers.empty:
                diccionario_iqr[columna] = self.dataframe[self.dataframe.index.isin(df_outliers.index.tolist())]
                print(f"La cantidad de outliers que tenemos para la columna {columna.upper()} es "
                      f"{diccionario_iqr[columna].shape[0]}")
        return diccionario_iqr

    def detectar_outliers(self, limite_outliers = 1.5, metodo="iqr" ):
        """
        Detecta outliers utilizando el método especificado.

        Parámetros:
        -----------
        metodo (str): Método para detectar outliers: "z_score", "z_score_modificado" o "iqr".
        kwargs: Argumentos adicionales para los métodos.

        Returns:
        --------
        dict: Diccionario de columnas con listas de índices de outliers.
        """
        if metodo == "z_score":
            return self.detectar_outliers_z_score()
        elif metodo == "iqr":
            return self.detectar_outliers_iqr(limite_outliers)
        else:
            raise ValueError("Método no válido. Los métodos disponibles son 'z_score', 'z_score_modificado' e 'iqr'.")


class GestionOutliersMultivariados:

    def __init__(self, dataframe, contaminacion = [0.01, 0.05, 0.1, 0.15]):
        self.dataframe = dataframe
        self.contaminacion = contaminacion

    def separar_variables_tipo(self):
        """
        Divide el DataFrame en columnas numéricas y de tipo objeto.
        """
        return self.dataframe.select_dtypes(include=np.number), self.dataframe.select_dtypes(include="O")


    def visualizar_outliers_bivariados(self, vr, tamano_grafica = (20, 15)):

        num_cols = len(self.separar_variables_tipo()[0].columns)
        num_filas = math.ceil(num_cols / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
        axes = axes.flat

        for indice, columna in enumerate(self.separar_variables_tipo()[0].columns):
            if columna == vr:
                fig.delaxes(axes[indice])
        
            else:
                sns.scatterplot(x = vr, 
                                y = columna, 
                                data = self.dataframe,
                                ax = axes[indice])
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None, ylabel = None)

        plt.tight_layout()


    
    def explorar_outliers_lof(self, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], vecinos=[20, 30], colores={-1: "red", 1: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Local Outlier Factor (LOF) y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo LOF. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - vecinos : list of int, opcional. Lista de números de vecinos a usar en el algoritmo LOF. Por defecto es [600, 1200].
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo LOF.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).

        Returns:
        
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada combinación de vecinos y nivel de contaminación especificado.
        """

        # Hacemos una copia del dataframe original para no hacer modificaciones sobre el original
        df_lof = self.dataframe.copy()
        
        # Extraemos las columnas numéricas 
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Generamos todas las posibles combinaciones entre los vecinos y el nivel de contaminación
        combinaciones = list(product(vecinos, indice_contaminacion))

        # Iteramos por cada posible combinación
        for combinacion in combinaciones:
            # Aplicar LOF con un número de vecinos y varias tasas de contaminación
            clf = LocalOutlierFactor(n_neighbors=combinacion[0], contamination=combinacion[1])
            y_pred = clf.fit_predict(self.dataframe[col_numericas])

            # Agregar la predicción de outliers al DataFrame
            df_lof["outlier"] = y_pred

            num_filas = math.ceil(len(col_numericas) / 2)

            fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
            axes = axes.flat

            # Asegurar que la variable dependiente no está en las columnas numéricas
            if var_dependiente in col_numericas:
                col_numericas.remove(var_dependiente)

            for indice, columna in enumerate(col_numericas):
                # Visualizar los outliers en un gráfico
                sns.scatterplot(x=var_dependiente, 
                                y=columna, 
                                data=df_lof,
                                hue="outlier", 
                                palette=colores, 
                                style="outlier", 
                                size=2,
                                ax=axes[indice])
                
                axes[indice].set_title(f"Contaminación = {combinacion[1]} y vecinos {combinacion[0]} y columna {columna.upper()}")
            
            plt.tight_layout()

            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])

            plt.show()

  


    def explorar_outliers_if(self, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], estimadores=1000, colores={-1: "red", 1: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo Isolation Forest. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - estimadores : int, opcional. Número de estimadores (árboles) a utilizar en el algoritmo Isolation Forest. Por defecto es 1000.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo Isolation Forest.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).
        
        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada valor de contaminación especificado.
        """
    
        df_if = self.dataframe.copy()

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()
         
        num_filas = math.ceil(len(col_numericas) / 2)

        for contaminacion in indice_contaminacion: 
            
            ifo = IsolationForest(random_state=42, 
                                n_estimators=estimadores, 
                                contamination=contaminacion,
                                max_samples="auto",  
                                n_jobs=-1)
            ifo.fit(self.dataframe[col_numericas])
            prediccion_ifo = ifo.predict(self.dataframe[col_numericas])
            df_if["outlier"] = prediccion_ifo

            fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
            axes = axes.flat
            for indice, columna in enumerate(col_numericas):
                if columna == var_dependiente:
                    fig.delaxes(axes[indice])

                else:
                    # Visualizar los outliers en un gráfico
                    sns.scatterplot(x=var_dependiente, 
                                    y=columna, 
                                    data=df_if,
                                    hue="outlier", 
                                    palette=colores, 
                                    style="outlier", 
                                    size=2,
                                    ax=axes[indice])
                    
                    axes[indice].set_title(f"Contaminación = {contaminacion} y columna {columna.upper()}")
                    plt.tight_layout()
                
                        
            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])

    def calcular_epsilon_dbscan(self):
        """
        Calcula el valor óptimo de epsilon para el algoritmo DBSCAN utilizando el método del gráfico K-distance.

        Este método separa las variables numéricas del DataFrame, calcula las distancias a los vecinos más cercanos,
        y genera un gráfico de línea que muestra la distancia al segundo vecino más cercano para cada punto.
        El punto donde la curva tiene el mayor cambio de pendiente puede ser un buen valor para epsilon.

        Params:
            No devuelve ningún parámetro

        Retorna:
            Esta función no retorna ningún valor, pero muestra un gráfico de línea interactivo utilizando Plotly.
        """
        df_num = self.separar_variables_tipo()[0]

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(df_num)
        distancias, _ = nbrs.kneighbors(df_num)

        df_distancias = pd.DataFrame(np.sort(distancias, axis=0)[:,1], columns=["epsilon"])
        _ = px.line(df_distancias, x=df_distancias.index, y="epsilon", title='Gráfico K-distance')


    def explorar_outliers_dbscan(self, epsilon, min_muestras, var_dependiente, colores={-1: "red", 0: "grey"}):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo DBSCAN y visualiza los resultados.

        Params:
            - epsilon : float. El valor de epsilon (radio máximo de la vecindad) para el algoritmo DBSCAN.
        
            - min_muestras : int. El número mínimo de muestras en una vecindad para que un punto sea considerado como un núcleo en DBSCAN.
        
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo DBSCAN.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 0: "grey"}).

        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados utilizando el algoritmo DBSCAN.
        """
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()
        num_filas = math.ceil(len(col_numericas) / 2)

        df_dbscan = self.dataframe.copy()

        model = DBSCAN(eps=epsilon, min_samples=min_muestras).fit(self.dataframe[col_numericas])
        outliers = model.labels_

        df_dbscan["outlier"] = outliers

        fig, axes = plt.subplots(num_filas, 2, figsize=(20, 15))
        axes = axes.flat

        for indice, columna in enumerate(col_numericas):
            if columna == var_dependiente:
                fig.delaxes(axes[indice])
            else:
                # Visualizar los outliers en un gráfico
                sns.scatterplot(x=var_dependiente, 
                                y=columna, 
                                data=df_dbscan,
                                hue="outlier", 
                                palette=colores, 
                                style="outlier", 
                                size=2,
                                ax=axes[indice])
                
                axes[indice].set_title(f"Columna {columna.upper()}")
                plt.tight_layout()
        
        if len(col_numericas) % 2 != 0:
            fig.delaxes(axes[-1])


    def detectar_outliers_lof(self, n_neighbors = [10,20,5], contaminacion = [0.01, 0.05,0.1]):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Local Outlier Factor (LOF).
        """
        df_lof = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        combinaciones_lof = list(product(n_neighbors, contaminacion))

        for vecino, contam in combinaciones_lof:

            lof = LocalOutlierFactor(
            n_neighbors=vecino,
            algorithm='auto',
            metric='minkowski',
            contamination=contam,
            n_jobs=-1
            )

            df_lof[f"outliers_lof_{contam}_{vecino}"]=(lof.fit_predict(df_lof[col_numericas]))


        return df_lof

    def detectar_outliers_if(self, contaminacion, n_estimators=[1000,2000]):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest.
        """
        df_if = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        combinaciones = list(product(n_estimators, contaminacion))

        for esti, contam in combinaciones:

            ifo = IsolationForest(random_state=42, 
                              n_estimators=esti, 
                              contamination=contam, 
                              max_samples="auto", 
                              n_jobs=-1)
        
            df_if[f"outliers_ifo_{contam}_{esti}"]=(ifo.fit_predict(self.dataframe[col_numericas]))

        return df_if

    def detectar_outliers_dbscan(self, epsilon, min_samples):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo DBSCAN.
        """
        df_dbscan = self.dataframe.copy()
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        model = DBSCAN(eps=epsilon, min_samples=min_samples).fit(self.dataframe[col_numericas])
        outliers = model.labels_
        df_dbscan["outlier"] = outliers

        return df_dbscan

    
    def imputar_outliers(self, data, metodo='media'):
        """
        Imputa los valores outliers en las columnas numéricas según el método especificado.
        
        Params:
            - data: DataFrame con los datos incluyendo la columna 'outlier'.
            - metodo: str, método de imputación ('media', 'mediana', 'moda').
        
        Returns:
            - DataFrame con los valores outliers imputados.
        """

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Diccionario de métodos de imputación
        metodos_imputacion = {
            'media': lambda x: x.mean(),
            'mediana': lambda x: x.median(),
            'moda': lambda x: x.mode()[0]
        }

        if metodo not in metodos_imputacion:
            raise ValueError("Método de imputación no reconocido. Utilice 'media', 'mediana' o 'moda'.")

        for col in col_numericas:
            valor_imputacion = metodos_imputacion[metodo](data.loc[data['outlier'] != -1, col])
            data.loc[data['outlier'] == -1, col] = valor_imputacion
        
        return data.drop("outlier", axis = 1)

    def capar_outliers(self, data,  lower_percentile=0.01, upper_percentile=0.99):
        """
        Capa los valores outliers en las columnas numéricas según los percentiles especificados.
        
        Params:
            - lower_percentile: float, percentil inferior para capar los valores (por defecto 0.01).
            - upper_percentile: float, percentil superior para capar los valores (por defecto 0.99).
        
        Returns:
            - DataFrame con los valores outliers capados.
        """
        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        for col in col_numericas:
            lower_bound = data[col].quantile(lower_percentile)
            upper_bound = data[col].quantile(upper_percentile)
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
        
        return data.drop("outlier", axis = 1)

    def transformar_outliers(self, data, metodo='log'):
        """
        Transforma los valores outliers en las columnas numéricas según el método especificado.
        
        Params:
            - metodo: str, método de transformación ('log', 'sqrt', 'inv').
        
        Returns:
            - DataFrame con los valores outliers transformados.
        """

        col_numericas = self.separar_variables_tipo()[0].columns.to_list()

        # Diccionario de métodos de transformación
        metodos_transformacion = {
            'log': np.log1p,  # log(1 + x) para evitar problemas con log(0)
            'sqrt': np.sqrt,
            'inv': lambda x: 1 / (x + np.finfo(float).eps)  # añadir epsilon para evitar división por cero
        }

        if metodo not in metodos_transformacion:
            raise ValueError("Método de transformación no reconocido. Utilice 'log', 'sqrt' o 'inv'.")

        for col in col_numericas:
            transform_func = metodos_transformacion[metodo]
            outlier_indices = data['outlier'] == -1
            data.loc[outlier_indices, col] = transform_func(data.loc[outlier_indices, col])
        
        return data.drop("outlier", axis = 1)

    
    def eliminar_outliers_abs(self,df_out):

            indices_a_eliminar = {}

            for index, row in df_out.iterrows():
                outliers = row.filter(like="outliers")
            
                if (outliers == -1).all():
                    indices_a_eliminar[index] = row
            
            df_outliers = pd.DataFrame(index = indices_a_eliminar.keys(),data=indices_a_eliminar.values())

            # Eliminar las filas identificadas
                
            df_sin_outliers = self.dataframe.drop(index=indices_a_eliminar.keys())
            print(f"se han detectado {len(indices_a_eliminar)} columnas con outliers absolutos")

            return df_outliers,df_sin_outliers
        
    def eliminar_outliers_porcj(self,df_out,limite = 0.7):

        indices_a_eliminar = {}

        for index, row in df_out.iterrows():
            outliers = row.filter(like="outliers")
        
            if (outliers == -1).sum() >= len(outliers) * limite:
                indices_a_eliminar[index] = row

        
        df_outliers = pd.DataFrame(index = indices_a_eliminar.keys(),data=indices_a_eliminar.values())
        # Eliminar las filas identificadas
        df_sin_outliers = self.dataframe.drop(index=indices_a_eliminar)
        print(f"se han eliminado {len(indices_a_eliminar)} columnas")

        return df_outliers,df_sin_outliers




        

class ImputarNulos:

    def __init__(self, dataframe):
        """
        Inicializa el ImputarNulos con un DataFrame.

        Parameters:
        - dataframe (pandas.DataFrame): El DataFrame que contiene las variables a visualizar.
        - color (str, opcional): El color a utilizar en las gráficas. Por defecto es "grey".
        """
        self.dataframe = dataframe
        
    def imputar_knn(self, lista_columnas, vecinos = 5):

        knn = KNNImputer(n_neighbors = vecinos)  
        df_knn = knn.fit_transform(self.dataframe[lista_columnas])

        nuevas_columnas_knn = [col + "_knn" for col in lista_columnas]

        df_num_knn = pd.DataFrame(data = df_knn, columns= nuevas_columnas_knn)

        return df_num_knn
    
    def imputar_iterative(self, lista_columnas, max_iter = 20):

        self.dataframe = self.dataframe.reset_index()

        imputer_iterative = IterativeImputer(max_iter=max_iter, random_state=42)

        iterative_imputado = imputer_iterative.fit_transform(self.dataframe[lista_columnas])

        nuevas_columnas_iter = [col + "_iterative" for col in lista_columnas]

        df_num_iter = pd.DataFrame(data = iterative_imputado, columns= nuevas_columnas_iter)

        return df_num_iter



