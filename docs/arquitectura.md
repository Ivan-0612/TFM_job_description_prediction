# Arquitectura del sistema

Este documento describe la arquitectura completa del sistema de generación y predicción de ofertas de trabajo.

---

## Visión general

El sistema tiene tres procesos principales que se ejecutan de forma coordinada:

```
Usuario
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│  BLOQUE 1 — INPUTS (Streamlit)                              │
│  Sector · Ciudad · Tipo de empleo · Formación · Descripción │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  PROCESO 1 — GENERACIÓN RAG                                 │
│                                                             │
│  Consulta FAISS (text-embedding-3-small)                    │
│    → 5 ofertas similares del corpus                         │
│  Prompt + contexto → GPT-4o-mini                            │
│    → titulo_habilidades (título + habilidades)              │
│    → oferta_completa (texto del anuncio)                    │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  PROCESO 2 — PREDICCIÓN (XGBoost)                           │
│                                                             │
│  titulo_habilidades → SBERT (multilingual-e5-large)         │
│    → embedding 1024 dims                                    │
│    → PCA → N componentes                                    │
│  + OHE (formacion, sector, tipo_empleo, ciudad)             │
│    → MinMaxScaler                                           │
│  XGBoost Salario  → rango salarial (0-9)                    │
│  XGBoost Seniority → nivel seniority (0-2)                  │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  PROCESO 3 — AJUSTE (LangGraph)   [opcional]                │
│                                                             │
│  ¿predicción == objetivo?                                   │
│    SÍ → fin                                                 │
│    NO → GPT-4o-mini reescribe titulo_habilidades            │
│         → volver a PROCESO 2                                │
│         → hasta 4 iteraciones                               │
│  Si hubo ajuste → regenerar oferta_completa                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Separación de vectorizaciones

El sistema usa **dos modelos de embeddings completamente separados**, cada uno con su propósito:

| Vectorización | Modelo | Dónde se usa | Por qué |
|---|---|---|---|
| **RAG** | `text-embedding-3-small` (OpenAI) | FAISS vectorstore | Búsqueda semántica de ofertas similares |
| **Predicción** | `intfloat/multilingual-e5-large` (SBERT) | Pipeline de inferencia XGBoost | Mismo modelo usado en entrenamiento |

Estos dos procesos son completamente independientes y no interfieren entre sí.

---

## Módulos del sistema

### `src/predictor/`

| Fichero | Clase | Responsabilidad |
|---|---|---|
| `embedder.py` | `Embedder` | Carga SBERT y vectoriza texto con prefijo `"passage: "` |
| `preprocessor.py` | `Preprocessor` | Carga artefactos (PCA, Scaler), aplica OHE y construye el vector final |
| `predictor.py` | `Predictor` | Orquesta embedder + dos preprocessors + dos modelos XGBoost |

### `src/rag/`

| Fichero | Función/Clase | Responsabilidad |
|---|---|---|
| `vectorstore.py` | `construir_vectorstore()` | Procesa `df_merged.csv` en lotes y construye el índice FAISS |
| `vectorstore.py` | `cargar_vectorstore()` | Carga el índice FAISS desde disco |
| `generator.py` | `GeneradorOfertas` | Búsqueda FAISS + llamada a GPT-4o-mini para generar oferta |

### `src/graph/`

| Fichero | Elemento | Responsabilidad |
|---|---|---|
| `state.py` | `OfertaState` | TypedDict con todo el estado del flujo |
| `nodes.py` | `nodo_generar` | Llama al generador RAG |
| `nodes.py` | `nodo_predecir` | Llama al predictor |
| `nodes.py` | `nodo_verificar` | Decide si ajustar o terminar (nodo condicional) |
| `nodes.py` | `nodo_ajustar` | Pide al LLM que reescriba titulo_habilidades |
| `graph.py` | `construir_grafo()` | Ensambla el StateGraph de LangGraph |

---

## Artefactos de los modelos

Cada modelo tiene su propio directorio de artefactos que deben ser compatibles entre sí (generados por el mismo notebook de entrenamiento):

```
artifacts/
├── salary/
│   ├── pca.joblib            ← PCA ajustado sobre embeddings SBERT del train set
│   ├── scaler.joblib         ← MinMaxScaler ajustado sobre features OHE del train set
│   │                            (guardado ANTES de concatenar embeddings)
│   └── feature_columns.json  ← Lista de columnas en el orden exacto que espera el modelo
│                                (se sobreescribe con model.get_booster().feature_names)
└── seniority/
    ├── pca.joblib            ← PCA con 144 componentes
    ├── scaler.joblib
    └── feature_columns.json
```

**Punto crítico:** el `scaler.joblib` debe guardarse **antes** de concatenar las componentes PCA al DataFrame de entrenamiento, porque el scaler solo debe operar sobre las variables OHE/numéricas, no sobre los embeddings reducidos.

---

## Flujo del StateGraph (LangGraph)

```
[START]
   │
   ▼
[nodo_generar]  →  genera titulo_habilidades + oferta_completa
   │
   ▼
[nodo_predecir]  →  predice salario_idx + seniority_idx
   │
   ▼
[nodo_verificar]  ─── ¿objetivos cumplidos o max_iter?
   │                         │
   │ "ajustar"               │ "fin"
   ▼                         ▼
[nodo_ajustar]            [END]
   │  (modifica titulo_habilidades)
   └──→ [nodo_predecir]  (bucle)
```

El estado `OfertaState` fluye a través de todos los nodos y acumula el historial de intentos anteriores, que se pasa al LLM de ajuste para evitar que repita los mismos errores.

---

## Inicialización lazy en producción

Para evitar cargar SBERT y los modelos XGBoost al importar el módulo (lo que provocaría doble carga en Streamlit), `nodes.py` usa inicialización lazy con variables globales:

```python
_generador = None
_predictor = None

def _get_generador():
    global _generador
    if _generador is None:
        _generador = GeneradorOfertas()
    return _generador
```

En `app.py`, Streamlit usa `@st.cache_resource` para garantizar que el `GeneradorOfertas` y el `Predictor` se instancian una sola vez por sesión del servidor.
