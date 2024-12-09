# README - Proyecto de Clustering y Regresión de Ventas

## Objetivo del Proyecto

Este proyecto tiene como objetivo realizar un análisis de segmentación (clustering) y regresión sobre los datos de ventas de una empresa para obtener insights valiosos que permitan tomar decisiones estratégicas informadas. Las principales preguntas que busca responder este proyecto son:

1. **¿Cómo podemos agrupar a los clientes o productos de manera significativa?**
   - Si queremos ver qué efecto tenemos en las ventas, Agruparlos por producto comprado (Maximizando beneficio y teiendo muy en cuenta el coste de envío).
   
2. **¿Qué factores son más relevantes para predecir el beneficio o las ventas dentro de cada grupo?**
   - El segmento de los suministros de oficna es muy lucrativo. Es importante tener en cuenta el gasto de envío.
   
3. **¿Cómo podemos utilizar estos insights para tomar decisiones estratégicas?**
   - Resulta importante maximizar aquellos productos que nos dan mayor beneficio y minimizar los de beneficio negativo. Estrategias como la venta conjunta o el uso de ofertas pueden ser adecuados.

### Para responder estas preguntas, el objetivo del proyecto es realizar:

1. **Clustering:**
   - Realizar un análisis de segmentación para agrupar clientes o productos según características clave, las cuales serán elegidas y justificadas a lo largo del proyecto.
   
2. **Regresión por Segmentos:**
   - Diseñar modelos de predicción para cada segmento, explicando las relaciones entre las variables e intentando predecir el total de ventas en cada uno de los segmentos.

## Estructura del Proyecto

### 1. Preparación de los Datos:
- Carpetas de preprocesado. EL que mejor resultado nos ha dado es el 4

### 2. Clustering:

- **Elección de variables:** Selección de atributos clave para agrupar las ventas, tales como: 
  - Sales, Profit, Discount, o Shipping Cost como principales.
  
- **Escalado:** Normalización robusta de las variables para evitar que algunas dominen sobre otras debido a diferencias en las escalas.
  
- **Método de clustering:** Aplicación de algoritmos como K-means, clustering jerárquico o DBSCAN, destacando el primero.

- **Evaluación:** Las mejores métricas las hemos obtenido en un método K-means con un coeficiente de silueta de 0,53 y y D_B de 0,67

### 3. Modelos de Regresión por Clusters:

- **Variable objetivo:** Sales.

---

## Detalles del Modelo

En este proyecto, hemos utilizado diferentes métodos de clustering para agrupar según la variable **"Ventas"**. Las columnas empleadas como regresores de **"Sales"** incluyen:

- Ship_Mode  
- Segment  
- Market  
- Region  
- Category  
- Quantity  
- Discount  
- Profit  
- Shipping_Cost  
- Order_Priority  

El mejor modelo que obtuvimos fue utilizando **K-means** con las siguientes métricas:

- **Silhouette Score:** 0.53373  
- **Davies-Bouldin Index:** 0.657544  

A continuación, realizamos una serie de regresiones para explicar el modelo. Aunque nos encontramos con algunos problemas de **overfitting**, identificamos que las variables más influyentes en la segmentación fueron **"Profit"** y **"Discount"**. Ambas variables resultaron ser significativas y aportaron al modelo, obteniendo las siguientes métricas con un **Gradient Boosting**:

En la carpeta Modelado2:

| Métrica  | Cluster 1 (Train) | Cluster 1 (Test) | Cluster 2 (Train) | Cluster 2 (Test) |
|----------|--------------------|------------------|--------------------|------------------|
| R²       | 0.905117           | 0.880280         | 0.966892           | 0.882812         |
| MAE      | 0.276518           | 0.288890         | 0.177652           | 0.226560         |
| MSE      | 0.405934           | 0.502789         | 0.122447           | 0.228010         |
| RMSE     | 0.637130           | 0.709076         | 0.349925           | 0.477504         |


---

## Recomendaciones

A partir de los resultados de este análisis, se proponen las siguientes recomendaciones:

1. **Las medidas de clusterización parecen adecuadas:** Unas mayores ventas vienen determinadas por aquellos productos que dan más beneficio, aquellos que se venden en más cantidad y un menor cste de envío

2. **Ajustar políticas de descuento:** Los descuentos parecen funcdamentales, aquellos productos con descuentos muy altos (De hasta el 80%) se venden significativamente más.
3. **Invertir en márketing para el material de oficina:** El material de oficina es significativamente más rentable que los productos tecnológicos. Es más barato y se le puede sacar mucho más beneficio por unidad frente a la tecnología o los muebles


## Instalación y Requisitos

Este proyecto requiere las siguientes bibliotecas de Python:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
