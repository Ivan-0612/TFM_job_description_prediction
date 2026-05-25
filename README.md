# TFM: Predicción de salario y seniority + Generación de ofertas de trabajo con RAG

Proyecto de Trabajo Fin de Máster que contiene un trabajo de Data Science: recogida de datos, procesado y limpieza, modelos de Machine Learning y un sistema de Inteligencia Artificial generativa para el mercado laboral español.

**Autor:** Iván Benito Sánchez
## Demostración / Caso de uso

La aplicación está desplegada en Hugging Face Spaces:

🔗 [https://huggingface.co/spaces/ivan1212a/tfm-generador-ofertas](https://huggingface.co/spaces/ivan1212a/tfm-generador-ofertas)

---

## Propósito del repositorio

Esta carpeta contiene los datos, notebooks y datos necesarios para **el entrenamiento** de los modelos de salario y seniority.


### Contenido principal

```text
TFM proyecto/
├── notebooks/        # Notebooks para ingestión, procesado y entrenamiento
├── data/             # Datos raw / interim / processed utilizados en entrenamiento
├── artifacts/        # Artefactos generados por los notebooks (PCA, scalers, columnas)
├── models/           # Modelos entrenados (.joblib)
└── README.md
```


## Notebooks y flujo de entrenamiento

Los notebooks en `notebooks/` cubren todo el flujo para generar el dataset, procesarlo y entrenar los modelos. Ejecuta los notebooks en el siguiente orden para reproducir un entrenamiento completo:

1. `01_get_data.ipynb`
2. `02_build_data.ipynb`
3. `03_process_data_ai.ipynb`
4. `04_process_data.ipynb`
5. `05_EDA.ipynb` 
6. `06_salary_model.ipynb`
7. `07_seniority_model.ipynb`


## Notebooks: flujo de datos

| Notebook | Descripción |
|---|---|
| `01_get_data` | Descarga datos de Adzuna API, scraping de Glassdoor por ciudad/puesto, descarga de Apify/LinkedIn |
| `02_build_data` | Unifica las tres fuentes, normaliza columnas y genera el dataset base |
| `03_process_data_ai` | Usa un LLM para extraer sector, ciudad, tipo de empleo y formación de las descripciones crudas |
| `04_process_data` | Limpieza final, categorización de salarios en rangos, generación de `df_merged.csv` |
| `05_EDA` | Análisis exploratorio: distribuciones, correlaciones, calidad de datos |
| `06_salary_model` | Embeddings SBERT → PCA → XGBoost para clasificación salarial|
| `07_seniority_model` | Mismo pipeline para predicción de seniority. Guarda artefactos en `artifacts/seniority/` |

---

## Modelos

### Predicción de salario

- **Tarea:** clasificación multiclase (10 rangos salariales).
- **Features:** embeddings SBERT reducidos con PCA + OHE de formación, sector y ciudad, escalados con MinMaxScaler.
- **Modelo activo:** `XGBClassifier` .
- **Clases:** `<15.000`, `15.000-22.000`, `22.000-30.000`, `30.000-40.000`, `40.000-65.000`, `65.000-80.000`, `80.000-100.000`, `>100.000`.

### Predicción de seniority

- **Tarea:** clasificación multiclase (3 niveles).
- **Features:** embeddings SBERT reducidos con PCA + OHE de formación, sector, tipo_empleo, escalados con MinMaxScaler.
- **Modelo activo:** `XGBClassifier`.
- **Clases:** `Intern` (0), `Junior` (1), `Senior` (2).

---

## Fuentes de datos

| Fuente | Descripción |
|---|---|
| **Adzuna API** | Ofertas de empleo vía API REST, cubriendo múltiples sectores |
| **Glassdoor** | Scraping por ciudad (54 ciudades españolas) y categoría de puesto (15 categorías) |

---

