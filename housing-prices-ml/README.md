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


### 7.1 Especificación detallada de las variables

**Variables excluidas del modelamiento:**

| Variable | Tipo de dato | Motivo de exclusión |
|---|---|---|
| `Order` | int64 (discreta) | Número correlativo de observación, sin relación con el precio |
| `PID` | int64 (nominal) | Identificador catastral (Parcel ID) de la propiedad, sin valor predictivo |

**Variable objetivo:**

| Variable | Tipo de dato | Tipo de variable | Descripción |
|---|---|---|---|
| `SalePrice` | int64 (continua) | Numérica continua | Precio de venta de la vivienda en dólares (variable a predecir) |
| `SalePrice_log` | float64 (continua) | Numérica continua, **derivada** | `log1p(SalePrice)`. No es una variable original del dataset; se crea en la sección 9.4 para corregir la asimetría del precio y usarla como objetivo real del entrenamiento |

**Variables predictoras numéricas (36):**

| Variable | Tipo de dato | Tipo de variable | Descripción |
|---|---|---|---|
| `MS SubClass` | int64 | Nominal (codificada como número) | Código que identifica el tipo de vivienda (ej. 1 piso, 2 pisos, duplex, con/sin ático) |
| `Lot Frontage` | float64 | Continua | Pies lineales de calle conectados a la propiedad |
| `Lot Area` | int64 | Continua | Superficie del lote en pies cuadrados |
| `Overall Qual` | int64 | Ordinal (escala 1–10) | Calidad general de materiales y terminaciones de la vivienda |
| `Overall Cond` | int64 | Ordinal (escala 1–10) | Condición general de la vivienda |
| `Year Built` | int64 | Discreta | Año de construcción original |
| `Year Remod/Add` | int64 | Discreta | Año de la última remodelación (igual a Year Built si nunca se remodeló) |
| `Mas Vnr Area` | float64 | Continua | Superficie de revestimiento de mampostería (m² en pies²) |
| `BsmtFin SF 1` | float64 | Continua | Pies cuadrados terminados del sótano, tipo 1 |
| `BsmtFin SF 2` | float64 | Continua | Pies cuadrados terminados del sótano, tipo 2 (si existe un segundo tipo) |
| `Bsmt Unf SF` | float64 | Continua | Pies cuadrados sin terminar del sótano |
| `Total Bsmt SF` | float64 | Continua | Superficie total del sótano |
| `1st Flr SF` | int64 | Continua | Superficie del primer piso |
| `2nd Flr SF` | int64 | Continua | Superficie del segundo piso |
| `Low Qual Fin SF` | int64 | Continua | Superficie terminada de baja calidad (todos los pisos) |
| `Gr Liv Area` | int64 | Continua | Superficie habitable sobre el nivel del suelo |
| `Bsmt Full Bath` | float64 | Discreta | Baños completos en el sótano |
| `Bsmt Half Bath` | float64 | Discreta | Medios baños en el sótano |
| `Full Bath` | int64 | Discreta | Baños completos sobre el nivel del suelo |
| `Half Bath` | int64 | Discreta | Medios baños sobre el nivel del suelo |
| `Bedroom AbvGr` | int64 | Discreta | Dormitorios sobre el nivel del suelo |
| `Kitchen AbvGr` | int64 | Discreta | Cocinas sobre el nivel del suelo |
| `TotRms AbvGrd` | int64 | Discreta | Total de habitaciones sobre el nivel del suelo (no incluye baños) |
| `Fireplaces` | int64 | Discreta | Cantidad de chimeneas |
| `Garage Yr Blt` | float64 | Discreta | Año de construcción del garage |
| `Garage Cars` | float64 | Discreta | Capacidad del garage en cantidad de autos |
| `Garage Area` | float64 | Continua | Superficie del garage en pies cuadrados |
| `Wood Deck SF` | int64 | Continua | Superficie de terraza de madera |
| `Open Porch SF` | int64 | Continua | Superficie de porche abierto |
| `Enclosed Porch` | int64 | Continua | Superficie de porche cerrado |
| `3-Ssn Porch` | int64 | Continua | Superficie de porche de tres estaciones |
| `Screen Porch` | int64 | Continua | Superficie de porche con mosquitero |
| `Pool Area` | int64 | Continua | Superficie de piscina |
| `Misc Val` | int64 | Continua | Valor en dólares de alguna característica miscelánea no cubierta en otras variables |
| `Mo Sold` | int64 | Discreta | Mes de venta |
| `Yr Sold` | int64 | Discreta | Año de venta |

**Variables predictoras categóricas (43):**

| Variable | Tipo de dato | Tipo de variable | Descripción |
|---|---|---|---|
| `MS Zoning` | object | Nominal | Clasificación general de zonificación de la venta (residencial, comercial, etc.) |
| `Street` | object | Nominal | Tipo de acceso vial a la propiedad (pavimentado/tierra) |
| `Alley` | object | Nominal | Tipo de acceso por callejón (o "None" si no tiene) |
| `Lot Shape` | object | Ordinal | Regularidad de la forma del lote (regular a muy irregular) |
| `Land Contour` | object | Nominal | Planitud del terreno |
| `Utilities` | object | Ordinal | Tipo de servicios básicos disponibles |
| `Lot Config` | object | Nominal | Configuración del lote (esquina, interior, cul-de-sac, etc.) |
| `Land Slope` | object | Ordinal | Pendiente del terreno |
| `Neighborhood` | object | Nominal (alta dimensionalidad) | Barrio dentro de los límites de la ciudad de Ames |
| `Condition 1` | object | Nominal | Proximidad a condiciones relevantes (vía principal, ferrocarril, parque, etc.) |
| `Condition 2` | object | Nominal | Segunda condición relevante, si aplica |
| `Bldg Type` | object | Nominal | Tipo de vivienda (unifamiliar, dúplex, townhouse, etc.) |
| `House Style` | object | Nominal | Estilo de la vivienda (1 piso, 2 pisos, con altillo, etc.) |
| `Roof Style` | object | Nominal | Tipo de techo |
| `Roof Matl` | object | Nominal | Material del techo |
| `Exterior 1st` | object | Nominal (alta dimensionalidad) | Material de revestimiento exterior principal |
| `Exterior 2nd` | object | Nominal (alta dimensionalidad) | Segundo material de revestimiento exterior, si aplica |
| `Mas Vnr Type` | object | Nominal | Tipo de revestimiento de mampostería |
| `Exter Qual` | object | Ordinal | Calidad del material exterior (Ex/Gd/TA/Fa/Po) |
| `Exter Cond` | object | Ordinal | Condición actual del material exterior |
| `Foundation` | object | Nominal | Tipo de fundación de la vivienda |
| `Bsmt Qual` | object | Ordinal | Altura/calidad del sótano |
| `Bsmt Cond` | object | Ordinal | Condición general del sótano |
| `Bsmt Exposure` | object | Ordinal | Exposición del sótano (walkout, jardín, sin exposición) |
| `BsmtFin Type 1` | object | Ordinal | Calidad del área terminada del sótano, tipo 1 |
| `BsmtFin Type 2` | object | Ordinal | Calidad del área terminada del sótano, tipo 2 |
| `Heating` | object | Nominal | Tipo de sistema de calefacción |
| `Heating QC` | object | Ordinal | Calidad y condición de la calefacción |
| `Central Air` | object | Nominal (binaria) | Aire acondicionado central (sí/no) |
| `Electrical` | object | Ordinal | Tipo de sistema eléctrico |
| `Kitchen Qual` | object | Ordinal | Calidad de la cocina |
| `Functional` | object | Ordinal | Funcionalidad de la vivienda (de típica a severamente dañada) |
| `Fireplace Qu` | object | Ordinal | Calidad de la(s) chimenea(s) |
| `Garage Type` | object | Nominal | Ubicación/tipo del garage |
| `Garage Finish` | object | Ordinal | Terminación interior del garage |
| `Garage Qual` | object | Ordinal | Calidad del garage |
| `Garage Cond` | object | Ordinal | Condición del garage |
| `Paved Drive` | object | Ordinal | Si el acceso vehicular está pavimentado |
| `Pool QC` | object | Ordinal | Calidad de la piscina (o "None" si no tiene) |
| `Fence` | object | Ordinal | Calidad del cierre perimetral (o "None" si no tiene) |
| `Misc Feature` | object | Nominal | Característica miscelánea no cubierta en otras categorías |
| `Sale Type` | object | Nominal | Tipo de venta (convencional, hipoteca asumida, nueva construcción, etc.) |
| `Sale Condition` | object | Nominal | Condición de la venta (normal, familiar, liquidación, etc.) |
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

### Detección sistemática de outliers (todas las variables numéricas)

Para evitar limitar el análisis a solo un par de variables elegidas de forma manual, se aplicó el método del **rango intercuartílico (IQR)** a las **37 variables numéricas** del dataset: un valor se considera atípico si cae fuera de `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`.

**Resultado:** 33 de las 37 variables numéricas presentan al menos un outlier según este criterio. Sin embargo, el número de outliers **no debe interpretarse como un indicador uniforme de "mala calidad de datos"**, ya que responde a distintos patrones según el tipo de variable:

- **Variables "zero-inflated" (concentradas en 0):** `Enclosed Porch`, `BsmtFin SF 2`, `Screen Porch`, `Mas Vnr Area`, `Open Porch SF` y `Misc Val` encabezan el ranking, con entre 100 y 459 outliers. Esto ocurre porque son características que **la mayoría de las viviendas simplemente no tiene** (no todas tienen porche cerrado, revestimiento de mampostería, o una característica miscelánea con valor comercial). El IQR marca como "atípico" a cualquier vivienda que sí la tenga, pero no se trata de errores de datos: es la distribución esperada de una variable binaria/rara expresada en superficie o valor monetario. No se eliminan estos registros, ya que esa información (que la vivienda tenga o no cierta característica) es valiosa para el modelo.
- **`Overall Cond` (252 outliers, 8,6%)** es un caso particular: al ser una variable **ordinal discreta** (escala 1 a 10) fuertemente concentrada en el valor "Average" (5), el método IQR —pensado para variables continuas— tiende a sobre-marcar los extremos de la escala como atípicos. Esto no implica un problema de calidad, sino una **limitación conocida del método IQR al aplicarse a variables discretas/ordinales**, y refuerza por qué la elección del método de detección de outliers debe considerar el tipo de cada variable (ver sección 9, Preparación de Datos).
- **`SalePrice` (137 outliers, 4,68%)** confirma de forma cuantitativa y sistemática lo que ya se había observado visualmente en la sección 8.3 (distribución asimétrica) y 8.6 (outliers evidentes de viviendas grandes y baratas): existe un grupo de viviendas de alto valor que se aleja del comportamiento típico del mercado.

**Conclusión:** la presencia de outliers según IQR es generalizada en el dataset, pero mayormente **explicable por la naturaleza de cada variable** y no por errores de registro. Por esta razón, la estrategia de tratamiento no es eliminar sistemáticamente todos estos valores, sino evaluarlos caso a caso: se mantuvieron como información válida las variables zero-inflated y `Overall Cond`, y solo se removieron (sección 9.1) los dos casos de outliers evidentes en `SalePrice` vs. `Gr Liv Area` identificados en el EDA inicial, por corresponder a condiciones de venta no representativas del mercado.

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
