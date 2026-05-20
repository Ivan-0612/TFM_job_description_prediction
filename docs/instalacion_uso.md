# Guía de instalación y uso

Este documento explica cómo instalar, configurar y ejecutar el sistema completo desde cero.

---

## Requisitos previos

- Python 3.10 o superior.
- Una clave de API de OpenAI (`OPENAI_API_KEY`) con acceso a `gpt-4o-mini` y `text-embedding-3-small`.
- Al menos 4 GB de RAM libres para cargar el modelo SBERT en CPU.
- Conexión a internet para las llamadas a la API de OpenAI.

---

## 1. Instalación de dependencias

Se recomienda usar un entorno virtual:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

Instalar dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Las dependencias principales que se necesitan para el sistema completo son:

```
# Interfaz
streamlit

# LLM y RAG
langchain
langchain-openai
langchain-community
langgraph
openai

# Embeddings semánticos
sentence-transformers

# Búsqueda vectorial
faiss-cpu

# Modelos ML
xgboost
lightgbm
scikit-learn

# Datos
pandas
numpy
joblib

# Utilidades
python-dotenv
```

> **Nota:** `requirements.txt` contiene las dependencias del proyecto completo incluyendo notebooks. Para ejecutar solo la aplicación Streamlit, las dependencias listadas arriba son suficientes.

---

## 2. Configuración de variables de entorno

Crea un fichero `.env` en la raíz del proyecto (junto a `app.py`):

```env
OPENAI_API_KEY=sk-proj-...tu-clave-aquí...
```

Este fichero **nunca debe subirse a un repositorio público**. Está incluido en `.gitignore`.

La clave se usa automáticamente por LangChain a través de `load_dotenv()` en `src/rag/vectorstore.py`, `src/rag/generator.py` y `src/graph/nodes.py`.

---

## 3. Verificar los artefactos de los modelos

Los artefactos de los modelos de predicción deben estar presentes en:

```
artifacts/
├── salary/
│   ├── pca.joblib
│   ├── scaler.joblib
│   └── feature_columns.json
└── seniority/
    ├── pca.joblib
    ├── scaler.joblib
    └── feature_columns.json
```

Y los modelos entrenados en:

```
models/
├── salary/
│   └── set_2_XGBClassifier.joblib
└── seniority/
    └── set_1_XGBClassifier.joblib
```

Si estos ficheros no están presentes, es necesario ejecutar los notebooks `06_salary_model.ipynb` y `07_seniority_model.ipynb` en orden (ver sección 5).

---

## 4. Construir el vectorstore (primera vez)

Si el directorio `vectorstore/ofertas_index/` no existe o está vacío:

```bash
python build_vectorstore.py
```

Este script:
1. Lee `data/interim/df_merged.csv` (~36.000 ofertas).
2. Vectoriza en lotes de 500 con `text-embedding-3-small`.
3. Guarda progreso en disco tras cada lote (tolerante a fallos).
4. Tarda entre 5 y 15 minutos dependiendo de la velocidad de la API.
5. Coste estimado: menos de 0,05 € en la API de OpenAI.

Si el proceso se interrumpe, se puede relanzar y reanudará desde donde se quedó.

Al terminar deben existir:
```
vectorstore/ofertas_index/index.faiss
vectorstore/ofertas_index/index.pkl
```

---

## 5. Ejecutar la aplicación

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en el navegador en `http://localhost:8501`.

### Primer arranque

El primer arranque es más lento porque Streamlit carga:
1. El vectorstore FAISS en memoria.
2. El modelo SBERT `intfloat/multilingual-e5-large` (~1,1 GB).
3. Los dos modelos XGBoost.

Gracias a `@st.cache_resource`, estas cargas solo ocurren una vez por sesión del servidor.

---

## 6. Uso de la aplicación

### Bloque 1 — Perfil de la oferta

Selecciona los parámetros del puesto que quieres generar:
- **Formación requerida:** nivel educativo mínimo del candidato.
- **Sector:** sector de actividad de la empresa.
- **Tipo de empleo:** jornada completa, media jornada, prácticas, etc.
- **Ciudad:** ciudad española donde se ubica el puesto.
- **Descripción:** texto libre describiendo el perfil buscado. Cuanto más detallado, mejor el resultado.

Pulsa **"Generar oferta"** para lanzar el proceso.

### Bloque 2 — Resultados

Tras la generación aparecen:
- **Oferta generada:** texto completo del anuncio de trabajo.
- **Título y habilidades extraídas:** campo interno usado para la predicción (visible en el expander).
- **Rango salarial predicho:** con barra de progreso indicando la posición en la escala de 10 rangos.
- **Nivel de seniority predicho:** los tres niveles (Intern / Junior / Senior) resaltando el predicho.

### Bloque 3 — Ajuste

Si el salario o seniority predicho no coincide con lo que necesitas:
1. Selecciona el **Salario objetivo** y/o el **Seniority objetivo**.
2. Pulsa **"Ajustar oferta"**.
3. El sistema lanzará el grafo LangGraph, que intentará reescribir la oferta hasta 4 veces para acercarse al objetivo.
4. La página se actualiza automáticamente con la nueva oferta y predicciones.

Si el objetivo no se alcanza tras 4 iteraciones, se muestra el mejor resultado obtenido.

---

## 7. Re-entrenar los modelos (opcional)

Si quieres re-entrenar los modelos con datos actualizados:

1. Ejecuta los notebooks en orden:
   ```
   01_get_data.ipynb
   02_build_data.ipynb
   03_process_data_ai.ipynb
   04_process_data.ipynb
   05_EDA.ipynb (opcional)
   06_salary_model.ipynb
   07_seniority_model.ipynb
   ```

2. Los notebooks 06 y 07 guardan automáticamente los artefactos en `artifacts/salary/` y `artifacts/seniority/` y los modelos en `models/salary/` y `models/seniority/`.

3. Después de re-entrenar, los nuevos artefactos se usarán automáticamente en la siguiente ejecución de la app.

**Importante:** si se modifican las features del modelo (por ejemplo añadir o quitar variables OHE), los artefactos (`pca.joblib`, `scaler.joblib`, `feature_columns.json`) y el modelo `.joblib` deben regenerarse todos juntos en el mismo notebook. Mezclar artefactos de entrenamientos distintos causará errores de shape.

---

## 8. Solución de problemas frecuentes

### La app tarda mucho en arrancar

Normal en el primer arranque. SBERT descarga el modelo (~1,1 GB) si no está en caché local. En arranques posteriores ya está en caché de HuggingFace.

### Error: `OPENAI_API_KEY not found`

Verifica que el fichero `.env` existe en la raíz del proyecto y contiene la clave correcta.

### Error de shape en predicción (`X has N features but expecting M`)

Los artefactos en `artifacts/` no son compatibles con el modelo en `models/`. Hay que re-ejecutar el notebook de entrenamiento correspondiente para regenerar ambos a la vez.

### Error: `Failed to fetch dynamically imported module`

Este error aparece en Streamlit cuando se usan componentes dinámicos (`st.metric`, `st.progress`, `st.success`) en ciertos entornos. La app ya usa HTML con `unsafe_allow_html=True` para evitarlo. Si aparece, verifica que `app.py` no tiene llamadas a esos componentes.

### XGBoost crash en inferencia con GPU

Si el modelo fue entrenado con GPU y se intenta hacer inferencia en CPU, puede crashear. La app ya aplica `set_params(device='cpu', tree_method='hist')` al cargar los modelos. Si el problema persiste, verifica la versión de XGBoost (`pip install xgboost --upgrade`).

### El vectorstore tarda demasiado en construirse

Es normal: procesa ~36.000 documentos. Con los límites de TPM de la API de OpenAI (tier gratuito), puede tardar hasta 30 minutos. El proceso guarda progreso tras cada lote de 500 documentos, por lo que se puede interrumpir y reanudar.
