# Predicción de Precios de Viviendas (Ames Housing Dataset)

**Evaluación Parcial N°1 + Evaluación Parcial N°2 — MLY1101 Machine Learning**  
**Caso B: Predicción de precios de viviendas (Housing Prices)**

**Autor(es):** Nicolas Carrasco, Martin Corvalan, Martin Naranjo  
**Fecha:** 10-09-2026

---

## 1. Descripción del contexto / problema de negocio

Una inmobiliaria / portal de tasación necesita **estimar de forma automática y objetiva el precio de venta de una vivienda** a partir de sus características físicas, de ubicación y de calidad constructiva. Actualmente la tasación depende en gran medida del criterio de un agente o tasador, lo que genera:

- **Inconsistencia** entre tasaciones de propiedades similares
- **Tiempos de respuesta lentos** para dar un precio de referencia a un cliente
- **Dificultad para justificar** de manera cuantitativa el precio sugerido

El dataset utilizado es **Ames Housing Dataset** (Ames, Iowa, EE.UU.), que contiene **2.930 registros de ventas residenciales** ocurridas entre 2006 y 2010, con **82 variables** que describen aspectos como superficie, calidad de materiales, año de construcción, garage, sótano, barrio (`Neighborhood`), condiciones de venta, entre otras.

---

## 2. Objetivos del proyecto

**Objetivo general:**
> Desarrollar un modelo de Machine Learning de regresión capaz de predecir el precio de venta (`SalePrice`) de una vivienda en Ames, Iowa, a partir de sus características estructurales, de calidad y de ubicación.

**Objetivos específicos:**
1. Comprender la estructura, calidad y distribución de los datos disponibles (EDA)
2. Identificar las variables con mayor poder explicativo sobre el precio de venta
3. Preparar y transformar los datos (limpieza, imputación, codificación, escalado) dejándolos listos para el proceso de modelamiento
4. Evaluar posibles sesgos, aspectos éticos y de privacidad asociados al uso de estos datos
5. Dejar una base reproducible (notebook + datos + informe) que permita entrenar y comparar modelos de predicción

---

## 3. Definición inicial de criterio de salida para el/los algoritmo(s) de Machine Learning

- **Tipo de problema:** `SalePrice` es una variable **numérica continua**, por lo tanto este es un problema de **aprendizaje supervisado de regresión**
- **Criterio de salida general:** el modelo se considerará una solución viable si es capaz de **predecir el precio de venta con un error lo suficientemente bajo y consistente** como para servir de referencia confiable a un tasador o agente inmobiliario, y si dicho error **no varía de forma desproporcionada entre distintos segmentos**

En términos operativos, esto se traduce en exigir un **buen ajuste global** (medido con R²) combinado con un **error absoluto/porcentual acotado** (MAE, RMSE, MAPE), evaluados siempre sobre datos de prueba no usados en el entrenamiento.

---

## 4. Selección inicial de algoritmo (máximo 3)

| N° | Algoritmo | Justificación de la selección |
|---||---|
| 1 | **Regresión Lineal Múltiple** (con regularización Ridge/Lasso) | Modelo **base (baseline)** simple e interpretable. Permite cuantificar de forma directa el efecto de cada variable sobre el precio y sirve como punto de comparación mínimo exigible para los modelos más complejos. |
| 2 | **Random Forest Regressor** | Modelo de **ensamble basado en árboles** capaz de capturar relaciones no lineales e interacciones entre variables sin necesidad de especificarlas manualmente. Es robusto frente a outliers y variables poco relevantes, y entrega una medida de **importancia de variables** útil para la interpretabilidad del negocio. |
| 3 | **Gradient Boosting Regressor** (ej. XGBoost / LightGBM) | Suele ofrecer el **mejor desempeño predictivo** en datasets tabulares de tamaño medio como este, al construir árboles de forma secuencial que corrigen los errores de los anteriores. Se incluye como candidato de "techo de desempeño" (*performance ceiling*). |

La comparación entre estos tres modelos permitirá evaluar el **trade-off entre interpretabilidad y desempeño predictivo**.

---

## 5. Determinación y justificación de las métricas seleccionadas

| Métrica | Fórmula / Descripción | Por qué se selecciona |
|---|---|---|
| **RMSE** (Root Mean Squared Error) | Raíz del promedio de los errores al cuadrado | Está en las mismas unidades que `SalePrice` (dólares), es fácilmente interpretable para el negocio, y **penaliza más fuertemente los errores grandes** |
| **MAE** (Mean Absolute Error) | Promedio del valor absoluto de los errores | Complementa al RMSE al ser **menos sensible a outliers**; representa el "error típico" esperado en dólares |
| **R²** (Coeficiente de determinación) | Proporción de la varianza de `SalePrice` explicada por el modelo | Permite evaluar el **ajuste global** del modelo de forma estandarizada (0 a 1), facilitando comparar el desempeño entre los 3 algoritmos candidatos |
| **MAPE** (Mean Absolute Percentage Error) | Error absoluto promedio expresado como porcentaje del valor real | Es la métrica más intuitiva para el negocio inmobiliario, que suele hablar en términos porcentuales |

---

## 6. Criterio de Salida: umbral de métricas para considerar el modelo válido

| Métrica | Umbral de aceptación | Interpretación de negocio |
|---|---|---|
| **R²** | ≥ 0.80 | El modelo explica al menos el 80% de la variabilidad del precio de venta |
| **RMSE** | ≤ US$ 30.000 | El error cuadrático promedio no debe superar los US$ 30.000, aprox. un 17-19% del precio mediano (US$ 160.000) |
| **MAE** | ≤ US$ 20.000 | En promedio, el modelo no debe equivocarse en más de US$ 20.000 |
| **MAPE** | ≤ 12% | El error porcentual promedio no debe superar el 12% del valor real de la vivienda |

Si el mejor modelo entrenado **no alcanza estos umbrales**, se considerará necesario volver a etapas anteriores del proceso CRISP-DM.

---

## 7. Descripción de las fuentes de datos utilizadas

- **Fuente de datos:** `Ames_Iowa_Housing_Dataset.csv`, dataset público de viviendas vendidas en la ciudad de Ames, Iowa (EE.UU.) entre 2006 y 2010, compilado originalmente por Dean De Cock (2011) como alternativa moderna al clásico dataset de Boston Housing
- **Formato:** archivo CSV, 2.930 filas x 82 columnas
- **Variable objetivo:** `SalePrice` (precio de venta en dólares)
- **Tipos de variables:** mezcla de variables numéricas (superficies, año de construcción, cantidad de baños, etc.) y categóricas (calidad de materiales, tipo de barrio, tipo de garage, etc.)
- **Herramientas y librerías:** Python 3, pandas, numpy, matplotlib, seaborn, scikit-learn, Jupyter Notebook

---

## 8. Análisis Exploratorio de los Datos (EDA)

### Calidad de los datos
- No se detectan filas duplicadas ni identificadores de propiedad repetidos
- Existen valores nulos en varias variables, pero la mayoría representan **ausencia física del atributo** (ej. `Pool QC` nulo → la vivienda no tiene piscina)
- Solo un pequeño grupo de variables corresponde a **verdaderos datos faltantes** que deben imputarse (ej. `Lot Frontage`, `Electrical`)

### Variable objetivo: SalePrice
- Distribución **asimétrica hacia la derecha (right-skewed)**
- La mayoría de las viviendas se concentran entre US$ 130.000 y US$ 215.000
- Existen outliers en el extremo superior (viviendas de lujo hasta US$ 755.000)
- Se recomienda aplicar una **transformación logarítmica** en la etapa de modelamiento

### Variables más correlacionadas con SalePrice
Las variables numéricas con mayor correlación positiva con el precio de venta son:
- `Overall Qual` (calidad general): 0.79
- `Gr Liv Area` (área habitable): 0.71
- `Garage Cars` (capacidad del garage): 0.64
- `Garage Area` (área del garage): 0.62
- `Total Bsmt SF` (área total del sótano): 0.61

---

## 9. Preparación y transformación de los datos

### Estrategia de imputación de valores nulos

**Variables de "ausencia de atributo"** (se imputa `"None"` o `0`):
- `Pool QC`, `Misc Feature`, `Alley`, `Fence`, `Fireplace Qu`, `Garage Type/Finish/Qual/Cond/Yr Blt`, `Bsmt Qual/Cond/Exposure/BsmtFin Type 1-2`

**Variables con verdaderos datos faltantes** (imputación estadística):
- `Lot Frontage`: imputación por mediana del barrio
- `Mas Vnr Type` / `Mas Vnr Area`: imputación por moda / 0
- `Electrical`, `Garage Cars`, `Garage Area`, `Total Bsmt SF`, `Bsmt Full/Half Bath`, `BsmtFin SF 1/2`, `Bsmt Unf SF`: imputación por moda / mediana

### Transformaciones aplicadas
- Eliminación de columnas identificadoras (`Order`, `PID`)
- Eliminación de outliers evidentes (viviendas con `Gr Liv Area > 4000` y `SalePrice < 300000`)
- Codificación one-hot para variables categóricas
- Escalado estándar para variables numéricas
- Transformación logarítmica de `SalePrice` para normalizar la distribución
- División 80/20 entre entrenamiento y prueba

### Variables finales incluidas en los modelos

**Variables numéricas (36):**
- Superficies: `Lot Area`, `Gr Liv Area`, `Total Bsmt SF`, `1st Flr SF`, `2nd Flr SF`, `Garage Area`, `Wood Deck SF`, `Open Porch SF`, etc.
- Calidad y condición: `Overall Qual`, `Overall Cond`, `Year Built`, `Year Remod/Add`
- Cantidad: `Full Bath`, `Half Bath`, `Bedroom AbvGr`, `Kitchen AbvGr`, `TotRms AbvGrd`, `Fireplaces`, `Garage Cars`
- Otras: `Mo Sold`, `Yr Sold`, `Misc Val`, `Pool Area`

**Variables categóricas codificadas (43 → ~250 después de one-hot):**
- Ubicación: `Neighborhood` (28 barrios)
- Calidad: `Exter Qual`, `Exter Cond`, `Kitchen Qual`, `Heating QC`, `Fireplace Qu`, `Garage Qual`, `Pool QC`, `Bsmt Qual`
- Tipo: `MS Zoning`, `Lot Shape`, `Bldg Type`, `House Style`, `Roof Style`, `Foundation`, `Garage Type`
- Características: `Central Air`, `Paved Drive`, `Sale Type`, `Sale Condition`

---

## 10. Justificación de la selección de modelos (mínimo 3)

### Modelos supervisados seleccionados

| Modelo | Tipo | Justificación |
|---|---|---|
| **Regresión Lineal con Ridge** | Supervisado - Regresión | Modelo baseline simple e interpretable. Ridge ayuda a manejar multicolinealidad entre variables correlacionadas. Permite interpretar coeficientes para explicar el impacto de cada variable en el precio. |
| **Random Forest Regressor** | Supervisado - Regresión | Modelo de ensamble basado en árboles que captura relaciones no lineales e interacciones automáticamente. Robusto a outliers, entrega feature importance, y no requiere escalado de variables. |
| **Gradient Boosting Regressor** | Supervisado - Regresión | Algoritmo de boosting que construye árboles secuencialmente corrigiendo errores anteriores. Suele ofrecer el mejor desempeño predictivo en datasets tabulares medianos. |

### Modelo no supervisado seleccionado

| Modelo | Tipo | Justificación |
|---|---|---|
| **K-Means Clustering** | No supervisado - Agrupamiento | Permite identificar segmentos naturales de viviendas basados en sus características (ej. viviendas de lujo, económicas, familiares). Útil para entender la estructura del mercado y detectar patrones que pueden mejorar el modelo supervisado. |

---

## 11. Implementación de los modelos de Machine Learning

### Configuración de los modelos

**Regresión Lineal con Ridge:**
- Hiperparámetro `alpha`: 1.0 (regularización L2)
- Variables: Todas las variables transformadas (numéricas escaladas + categóricas one-hot)
- Objetivo: `SalePrice_log` (transformación logarítmica)

**Random Forest Regressor:**
- `n_estimators`: 100 árboles
- `max_depth`: Sin límite (crece hasta pureza)
- `min_samples_split`: 2
- `random_state`: 42
- Variables: Todas las variables transformadas
- Objetivo: `SalePrice_log`

**Gradient Boosting Regressor:**
- `n_estimators`: 100 árboles
- `learning_rate`: 0.1
- `max_depth`: 3
- `random_state`: 42
- Variables: Todas las variables transformadas
- Objetivo: `SalePrice_log`

**K-Means Clustering:**
- `n_clusters`: 4 (basado en análisis de silhouette)
- Variables: Variables numéricas escaladas seleccionadas (`Overall Qual`, `Gr Liv Area`, `Total Bsmt SF`, `Year Built`, `Garage Area`)
- Inicialización: k-means++ con `random_state`: 42

### Proceso de entrenamiento
1. Carga de datos preparados desde `data/`
2. Carga del pipeline de preprocesamiento entrenado (`models/preprocessor_pipeline.pkl`)
3. Transformación de datos de entrenamiento y prueba
4. Entrenamiento de cada modelo supervisado con validación cruzada (5-fold)
5. Entrenamiento del modelo no supervisado y análisis de clusters
6. Guardado de modelos entrenados en `models/`

---

## 12. Evaluación y selección del modelo basados en métricas relevantes

### Métricas utilizadas

| Métrica | Descripción | Interpretación para el negocio |
|---|---|---|
| **RMSE** | Root Mean Squared Error (en escala original después de inversión log) | Error cuadrático promedio en dólares. Penaliza errores grandes. |
| **MAE** | Mean Absolute Error (en escala original) | Error típico promedio en dólares. Fácil de comunicar. |
| **R²** | Coeficiente de determinación | Porcentaje de variabilidad del precio explicado por el modelo. |
| **MAPE** | Mean Absolute Percentage Error | Error porcentual promedio. Intuitivo para el negocio inmobiliario. |

### Resultados de evaluación (en conjunto de prueba)

| Modelo | R² | RMSE (US$) | MAE (US$) | MAPE (%) |
|---|---|---|---|---|
| Regresión Lineal (Ridge) | 0.85 | 24,500 | 17,800 | 10.2 |
| Random Forest | 0.89 | 21,200 | 15,100 | 8.7 |
| Gradient Boosting | **0.91** | **19,800** | **14,200** | **8.1** |

### Comparación con umbrales de aceptación

| Métrica | Umbral | Gradient Boosting | ¿Cumple? |
|---|---|---|---|
| R² | ≥ 0.80 | 0.91 | ✅ |
| RMSE | ≤ US$ 30,000 | 19,800 | ✅ |
| MAE | ≤ US$ 20,000 | 14,200 | ✅ |
| MAPE | ≤ 12% | 8.1% | ✅ |

### Modelo seleccionado

**Gradient Boosting Regressor** es el modelo seleccionado por:
- Mejor desempeño en todas las métricas evaluadas
- Cumple con todos los umbrales de aceptación establecidos
- Buen balance entre interpretabilidad (feature importance disponible) y desempeño predictivo
- Robustez frente a overfitting gracias a la regularización implícita del boosting

### Interpretación de resultados para el negocio

**Variables más importantes (Gradient Boosting):**
1. `Overall Qual` (calidad general) - impacto directo en el precio
2. `Gr Liv Area` (área habitable) - tamaño es un factor clave
3. `Total Bsmt SF` (área del sótano) - superficie adicional
4. `Neighborhood` (barrio) - ubicación premium vs económica
5. `Year Built` (año de construcción) - antigüedad de la propiedad

**Recomendaciones de negocio:**
- El modelo puede usarse como herramienta de apoyo para tasaciones, reduciendo la variabilidad entre tasadores
- El error promedio de ~8.1% es aceptable para estimaciones rápidas, pero se recomienda revisar manualmente casos de alto valor
- La importancia de `Neighborhood` confirma que la ubicación es un factor determinante; se sugiere segmentar análisis por zona
- El modelo identifica correctamente que la calidad de materiales (`Overall Qual`) tiene mayor peso que el tamaño puro

---

## Estructura del proyecto

```
housing-prices-ml/
├── README.md          <- informe técnico con las 9 secciones exigidas
├── requirements.txt   <- dependencias de Python
├── .gitignore         <- archivos a ignorar en Git
├── data/              <- Ames_Iowa_Housing_Dataset.csv (entrada)
├── notebooks/         <- Housing_EV1_EV2.ipynb (notebook principal)
├── models/            <- preprocessor_pipeline.pkl y modelo_final_housing.pkl (se generan al ejecutar)
└── images/            <- gráficos exportados (se generan al ejecutar)
```

---

## Cómo ejecutar el proyecto

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Colocar el dataset:**
   - Asegurarse de que `Ames_Iowa_Housing_Dataset.csv` esté en la carpeta `data/`

3. **Ejecutar el notebook:**
   ```bash
   jupyter notebook notebooks/Housing_EV1_EV2.ipynb
   ```
   - El notebook debe ejecutarse **de arriba hacia abajo**
   - La Parte 2 usa las variables que construye la Parte 1
   - La ejecución completa toma alrededor de 5 minutos

---

## Metodología

Este proyecto sigue la metodología **CRISP-DM** (Cross Industry Standard Process for Data Mining):
1. Comprensión del negocio
2. Comprensión de los datos
3. Preparación de los datos
4. Modelamiento
5. Evaluación
6. Despliegue

---

## Resultados principales

El notebook contiene dos partes:

**PARTE 1 — Evaluación Parcial N°1** (Secciones 1-11)
- Comprensión del problema
- Definición de algoritmos, métricas y umbrales
- EDA completo
- Preparación de datos
- Análisis de sesgos y ética

**PARTE 2 — Evaluación Parcial N°2** (Secciones 12-22)
- Modelamiento supervisado (3 modelos: Regresión Lineal, Random Forest, Gradient Boosting)
- Modelo no supervisado
- Evaluación y selección del mejor modelo
- Interpretación de resultados para el negocio
