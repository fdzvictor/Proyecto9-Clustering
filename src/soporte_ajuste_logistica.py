

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

import time
import psutil

import pickle

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix
)
# import shap

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import roc_curve, auc

import statsmodels.api as sm
import shap



class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(),
            "tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier()
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}


    def ajustar_modelo(self, modelo_nombre, param_grid=None, cross_validation = 5, nombre_modelo = "mejor_modelo"):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        parametros_default = {
            "logistic_regression": [
                {'penalty': ['l1'], 'solver': ['saga'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['l2'], 'solver': ['liblinear'], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['none'], 'solver': ['lbfgs'], 'max_iter': [10000]}
                ]              
            ,
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})
        
        else:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Ajuste del modelo
        grid_search = GridSearchCV(estimator=modelo, 
                                   param_grid=param_grid, 
                                   cv=cross_validation, 
                                   scoring='accuracy')
        
        grid_search.fit(self.X_train, self.y_train)
        self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_
        self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
        self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)
        display (grid_search.best_estimator_)

        # Guardar el modelo
        with open(f"{nombre_modelo + '_' + modelo_nombre}.pkl", 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

    def calcular_metricas(self, modelo_nombre):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC, Kappa,
        tiempo de computación y núcleos utilizados.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        # Registrar tiempo de ejecución
        start_time = time.time()
        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None
        elapsed_time = time.time() - start_time

        # Registrar núcleos utilizados
        num_nucleos = getattr(modelo, "n_jobs", psutil.cpu_count(logical=True))

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None,
            "time_seconds": elapsed_time,
            "n_jobs": num_nucleos
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None,
            "tiempo_computacion(segundos)": elapsed_time,
            "nucleos_usados": num_nucleos
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test}).T

    def plot_matriz_confusion(self, modelo_nombre):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()

    def plot_shap_summary(self, modelo_nombre):
        """
        Genera un SHAP summary plot para el modelo seleccionado.
        Maneja correctamente modelos de clasificación con múltiples clases.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        modelo = self.resultados[modelo_nombre]["mejor_modelo"]

        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de generar el SHAP plot.")

        # Usar TreeExplainer para modelos basados en árboles
        if modelo_nombre in ["tree", "random_forest", "gradient_boosting", "xgboost"]:
            explainer = shap.TreeExplainer(modelo)
            shap_values = explainer.shap_values(self.X_test)

            # Verificar si los SHAP values tienen múltiples clases (dimensión 3)
            if isinstance(shap_values, list):
                # Para modelos binarios, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[1]
            elif len(shap_values.shape) == 3:
                # Para Decision Trees, seleccionar SHAP values de la clase positiva
                shap_values = shap_values[:, :, 1]
        else:
            # Usar el explicador genérico para otros modelos
            explainer = shap.Explainer(modelo, self.X_test, check_additivity=False)
            shap_values = explainer(self.X_test).values

        # Generar el summary plot estándar
        shap.summary_plot(shap_values, self.X_test, feature_names=self.X.columns)

    
    def importancia_predictores(self, modelo_nombre):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        display(importancia_df)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()


    def calcular_graficar_roc(self, modelo_nombre, conjunto='test', mostrar_auc=True):
        """
        Calcula y grafica la curva ROC para un modelo ajustado.
        
        Parámetros:
        - modelo_nombre: str, nombre del modelo ajustado.
        - conjunto: str, 'train' o 'test', indica el conjunto de datos para calcular la curva ROC.
        - mostrar_auc: bool, si True, muestra el AUC en la gráfica.
        
        Devuelve:
        - fpr: False Positive Rate
        - tpr: True Positive Rate
        - thresholds: Umbrales usados para calcular la curva ROC
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la curva ROC.")
        
        # Seleccionar conjunto de datos
        if conjunto == 'train':
            X, y = self.X_train, self.y_train
        elif conjunto == 'test':
            X, y = self.X_test, self.y_test
        else:
            raise ValueError("El parámetro 'conjunto' debe ser 'train' o 'test'.")

        # Calcular probabilidades
        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X)[:, 1]
        else:
            raise ValueError(f"El modelo '{modelo_nombre}' no soporta cálculo de probabilidades para la curva ROC.")

        # Calcular puntos ROC
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        
        # Calcular AUC
        auc = roc_auc_score(y, y_prob) if mostrar_auc else None
        
        # Graficar la curva ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}' if mostrar_auc else 'ROC Curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Línea aleatoria')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Curva ROC ({modelo_nombre} - {conjunto})')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        
        return fpr, tpr, thresholds
    

        # Función para asignar colores
    def color_filas_por_modelo(row):
        if row["modelo"] == "decision ¡tree":
            return ["background-color: #e6b3e0; color: black"] * len(row)  
        
        elif row["modelo"] == "random_forest":
            return ["background-color: #c2f0c2; color: black"] * len(row) 

        elif row["modelo"] == "gradient_boosting":
            return ["background-color: #ffd9b3; color: black"] * len(row)  

        elif row["modelo"] == "xgboost":
            return ["background-color: #f7b3c2; color: black"] * len(row)  

        elif row["modelo"] == "regresion lineal":
            return ["background-color: #b3d1ff; color: black"] * len(row)  
        
        return ["color: black"] * len(row)






    