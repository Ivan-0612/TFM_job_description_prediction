# TFM: Predicción de salario y seniority + Generación de ofertas de trabajo con RAG

Proyecto de Trabajo Fin de Máster que combina un pipeline completo de datos, modelos de Machine Learning y un sistema de Inteligencia Artificial generativa para el mercado laboral español.

**Autor:** Iván Benito Sánchez

---

## Descripción general

El sistema permite a un usuario describir el perfil de una oferta de trabajo (sector, ciudad, tipo de empleo, formación requerida y descripción libre) y obtiene:

1. **Una oferta de trabajo generada automáticamente** mediante RAG (Retrieval-Augmented Generation) con GPT-4o-mini, basándose en ofertas reales del mercado como contexto.
2. **Predicción del rango salarial** (10 categorías) usando un modelo XGBoost entrenado con embeddings semánticos + variables estructurales.
3. **Predicción del nivel de seniority** (Intern / Junior / Senior) usando otro modelo XGBoost con el mismo enfoque.
4. **Bucle de ajuste inteligente** mediante LangGraph: si el salario o seniority predicho no coincide con el objetivo del usuario, el sistema reescribe la oferta iterativamente hasta acercarse al objetivo.

---

## Estructura del repositorio

```
TFM proyecto/
├── app.py                        # Interfaz Streamlit
├── build_vectorstore.py          # Script one-time para construir el índice FAISS
├── requirements.txt
├── .env                          # Variables de entorno (OPENAI_API_KEY)
│
├── data/
│   ├── raw/                      # Datos originales de Adzuna, Apify/LinkedIn, Glassdoor
│   ├── interim/
│   │   ├── df_merged.csv         # Dataset consolidado y limpio (fuente del vectorstore)
│   │   └── df_embeddings.csv     # Dataset con embeddings SBERT precalculados
│   └── processed/
│
├── notebooks/
│   ├── 01_get_data.ipynb         # Ingesta de datos (Adzuna API, Apify, Glassdoor scraping)
│   ├── 02_build_data.ipynb       # Integración y consolidación de fuentes
│   ├── 03_process_data_ai.ipynb  # Extracción de campos con IA (sector, ciudad, etc.)
│   ├── 04_process_data.ipynb     # Limpieza, normalización y dataset final
│   ├── 05_EDA.ipynb              # Análisis exploratorio de datos
│   ├── 06_salary_model.ipynb     # Entrenamiento del modelo de predicción de salario
│   └── 07_seniority_model.ipynb  # Entrenamiento del modelo de predicción de seniority
│
├── models/
│   ├── salary/
│   │   ├── set_2_XGBClassifier.joblib   # Modelo activo (sin tipo_empleo)
│   │   ├── set_1_XGBClassifier.joblib
│   │   ├── set_1_LGBMClassifier.joblib
│   │   ├── set_2_LGBMClassifier.joblib
│   │   ├── set_1_RandomForestClassifier.joblib
│   │   └── set_2_RandomForestClassifier.joblib
│   └── seniority/
│       ├── set_1_XGBClassifier.joblib   # Modelo activo (con todas las features)
│       └── set_1_LGBMClassifier.joblib
│
├── artifacts/
│   ├── salary/
│   │   ├── pca.joblib            # PCA entrenado sobre embeddings (87 componentes)
│   │   ├── scaler.joblib         # MinMaxScaler sobre variables OHE+numéricas
│   │   └── feature_columns.json  # Columnas exactas que espera el modelo
│   └── seniority/
│       ├── pca.joblib            # PCA entrenado sobre embeddings (144 componentes)
│       ├── scaler.joblib
│       └── feature_columns.json
│
├── vectorstore/
│   └── ofertas_index/
│       ├── index.faiss           # Índice FAISS con ~36.000 ofertas vectorizadas
│       └── index.pkl             # Metadatos del docstore
│
└── src/
    ├── predictor/
    │   ├── embedder.py           # Wrapper SBERT (intfloat/multilingual-e5-large)
    │   ├── preprocessor.py       # OHE + PCA + Scaler → vector de features
    │   └── predictor.py          # Orquestador de inferencia (salario + seniority)
    ├── rag/
    │   ├── vectorstore.py        # Construcción y carga del índice FAISS
    │   └── generator.py          # Cadena RAG + LLM para generación de ofertas
    └── graph/
        ├── state.py              # Definición del estado (TypedDict)
        ├── nodes.py              # Nodos del grafo (generar, predecir, ajustar, verificar)
        └── graph.py              # Construcción del StateGraph con LangGraph
```

---

## Arquitectura del sistema

El sistema tiene dos procesos completamente separados que se coordinan a través del grafo:

**Proceso 1 — Generación RAG** (usa OpenAI `text-embedding-3-small` + GPT-4o-mini):
- El usuario describe el perfil → se buscan 5 ofertas similares en FAISS → el LLM genera una oferta nueva con ese contexto.
- Salida: `titulo_habilidades` (título + lista de habilidades) + `oferta_completa` (texto del anuncio).

**Proceso 2 — Predicción** (usa SBERT `intfloat/multilingual-e5-large` + XGBoost):
- El `titulo_habilidades` se vectoriza con SBERT (1024 dims) → se reduce con PCA → se combina con variables OHE escaladas → XGBoost predice salario y seniority.

**Proceso 3 — Ajuste LangGraph** (usa GPT-4o-mini):
- Si el resultado no cumple los objetivos, el LLM reescribe el `titulo_habilidades` → se vuelve a predecir → bucle hasta 4 iteraciones.
- Si hubo ajuste, se regenera el texto completo de la oferta con coherencia.

Consulta [docs/arquitectura.md](docs/arquitectura.md) para el diagrama detallado.

---


## Notebooks: flujo de datos

| Notebook | Descripción |
|---|---|
| `01_get_data` | Descarga datos de Adzuna API, scraping de Glassdoor por ciudad/puesto, descarga de Apify/LinkedIn |
| `02_build_data` | Unifica las tres fuentes, normaliza columnas y genera el dataset base |
| `03_process_data_ai` | Usa un LLM para extraer sector, ciudad, tipo de empleo y formación de las descripciones crudas |
| `04_process_data` | Limpieza final, categorización de salarios en rangos, generación de `df_merged.csv` |
| `05_EDA` | Análisis exploratorio: distribuciones, correlaciones, calidad de datos |
| `06_salary_model` | Embeddings SBERT → PCA → XGBoost para clasificación salarial. Guarda artefactos en `artifacts/salary/` |
| `07_seniority_model` | Mismo pipeline para predicción de seniority. Guarda artefactos en `artifacts/seniority/` |

---

## Modelos

### Predicción de salario

- **Tarea:** clasificación multiclase (10 rangos salariales).
- **Features:** embeddings SBERT reducidos con PCA (87 componentes) + OHE de formación, sector y ciudad, escalados con MinMaxScaler.
- **Modelo activo:** `set_2_XGBClassifier` (sin `tipo_empleo`, que reduce ruido).
- **Clases:** `<15.000`, `15.000-22.000`, `22.000-30.000`, `30.000-40.000`, `40.000-52.000`, `52.000-65.000`, `65.000-80.000`, `80.000-100.000`, `100.000-150.000`, `>150.000`.

### Predicción de seniority

- **Tarea:** clasificación multiclase (3 niveles).
- **Features:** embeddings SBERT reducidos con PCA (144 componentes) + OHE de formación, sector, tipo_empleo, escalados con MinMaxScaler.
- **Modelo activo:** `set_1_XGBClassifier` (con todas las features incluyendo `tipo_empleo`).
- **Clases:** `Intern` (0), `Junior` (1), `Senior` (2).

Ambos modelos se fuerzan a CPU en inferencia: `set_params(device='cpu', tree_method='hist')`.

---

## Fuentes de datos

| Fuente | Descripción |
|---|---|
| **Adzuna API** | Ofertas de empleo vía API REST, cubriendo múltiples sectores |
| **Apify / LinkedIn** | Scraping de LinkedIn Jobs por nivel de seniority (entry, associate, mid, director, internship) |
| **Glassdoor** | Scraping por ciudad (54 ciudades españolas) y categoría de puesto (15 categorías) |

---

## Documentación adicional

- [docs/arquitectura.md](docs/arquitectura.md) — Diagrama de arquitectura y flujo de datos.
- [docs/pipeline_inferencia.md](docs/pipeline_inferencia.md) — Pipeline de predicción en detalle.
- [docs/rag_generacion.md](docs/rag_generacion.md) — Sistema RAG: vectorstore y generación.
- [docs/grafo_ajuste.md](docs/grafo_ajuste.md) — Grafo LangGraph y bucle de ajuste.
- [docs/instalacion_uso.md](docs/instalacion_uso.md) — Guía completa de instalación y uso.
