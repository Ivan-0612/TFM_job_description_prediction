# Pipeline de inferencia: predicción de salario y seniority

Este documento describe en detalle el pipeline de predicción (`src/predictor/`), que transforma el texto de una oferta de trabajo en predicciones de rango salarial y nivel de seniority.

---

## Visión general

```
titulo_habilidades (str)
        │
        ▼
   [ Embedder ]
   SBERT: intfloat/multilingual-e5-large
   → embedding: array(1024,)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
[ Preprocessor salary ]          [ Preprocessor seniority ]
  PCA (87 componentes)              PCA (144 componentes)
  + OHE sin tipo_empleo             + OHE con tipo_empleo
  + MinMaxScaler                    + MinMaxScaler
  → array(N_salary_features,)       → array(N_seniority_features,)
        │                                  │
        ▼                                  ▼
[ XGBoost set_2 ]                [ XGBoost set_1 ]
→ salario_idx (0-9)              → seniority_idx (0-2)
```

---

## `Embedder` (`src/predictor/embedder.py`)

### Propósito

Vectorizar el texto de la oferta usando el mismo modelo que se usó durante el entrenamiento, garantizando que el espacio de embeddings sea compatible.

### Modelo

`intfloat/multilingual-e5-large` — modelo multilingüe de SentenceTransformers optimizado para recuperación semántica. Produce embeddings de **1024 dimensiones**.

### Uso del prefijo

El modelo E5 requiere un prefijo específico según el tipo de texto:
- **`"passage: "`** → para documentos/contenido a indexar (usado en entrenamiento e inferencia).
- **`"query: "`** → para consultas de búsqueda.

En este proyecto siempre se usa `"passage: "` tanto en entrenamiento como en inferencia, garantizando consistencia.

```python
class Embedder:
    def __init__(self, device: str = "cpu"):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

    def embed(self, texto: str) -> np.ndarray:
        texto_preparado = "passage: " + texto
        embedding = self.model.encode([texto_preparado], show_progress_bar=False, batch_size=1)
        return embedding[0]  # array 1D de 1024 dimensiones
```

**Nota:** En inferencia se fuerza `device="cpu"` para compatibilidad en entornos sin GPU.

---

## `Preprocessor` (`src/predictor/preprocessor.py`)

### Propósito

Transformar el embedding SBERT y las variables estructurales (formación, sector, tipo de empleo, ciudad) en el vector exacto de features que espera cada modelo XGBoost.

### Artefactos cargados

- `pca.joblib` — PCA ajustado sobre los embeddings del train set.
- `scaler.joblib` — MinMaxScaler ajustado sobre las columnas OHE del train set.
- `feature_columns.json` — lista ordenada de columnas finales del modelo (sobreescrita por `model.get_booster().feature_names` en el `Predictor`).

### Pasos del preprocesado

**Paso 1 — Reducción de dimensionalidad (PCA)**

```python
embedding_reducido = self.pca.transform([embedding])
# → array (1, N_componentes)
df_embedding = pd.DataFrame(embedding_reducido, columns=[f"PC{i+1}" for i in range(...)])
```

El PCA reduce las 1024 dimensiones del embedding a un número menor de componentes (87 para salario, 144 para seniority), eliminando ruido y reduciendo el coste computacional.

**Paso 2 — One-Hot Encoding (OHE)**

Las variables categóricas se codifican con `pd.get_dummies` usando prefijos explícitos que coinciden exactamente con los usados en el entrenamiento:

```python
df_ohe = pd.concat([
    pd.get_dummies(df_struct["formación_académica"], prefix="formacion", drop_first=True, dummy_na=True),
    pd.get_dummies(df_struct["sector"], prefix="sector", drop_first=True),
    pd.get_dummies(df_struct["tipo_de_empleo"], prefix="tipo_empleo", drop_first=True, dummy_na=True),
    pd.get_dummies(df_struct["Ciudad"], prefix="ciudad", drop_first=True),
], axis=1).astype(int)
```

Las comas en los nombres de categorías (ej. `"sector_Salud, Farmacia..."`) se reemplazan por guiones bajos para evitar conflictos:

```python
df_ohe.columns = df_ohe.columns.str.replace(",", "_")
```

**Paso 3 — Escalado (MinMaxScaler)**

Se usa el scaler ajustado durante el entrenamiento. El reindex previo garantiza que el orden de columnas coincide exactamente con el que vio el scaler, rellenando con 0 las categorías no vistas:

```python
scaler_cols = [c.replace(",", "_") for c in self.scaler.feature_names_in_]
df_ohe_scaled = pd.DataFrame(
    self.scaler.transform(df_ohe.reindex(columns=scaler_cols, fill_value=0).values),
    columns=scaler_cols,
)
```

**Paso 4 — Concatenación**

```python
df_full = pd.concat([df_ohe_scaled, df_embedding], axis=1)
```

El orden es importante: primero las variables OHE escaladas, luego las componentes PCA. El scaler NO se aplicó sobre los embeddings PCA durante el entrenamiento.

**Paso 5 — Reindex a las features del modelo**

```python
df_final = df_full.reindex(columns=self.feature_columns, fill_value=0)
return df_final.values
```

Garantiza el orden exacto de columnas que espera el modelo XGBoost.

---

## `Predictor` (`src/predictor/predictor.py`)

### Propósito

Orquestador que ensambla todos los componentes y expone una única interfaz de predicción.

### Inicialización

```python
class Predictor:
    def __init__(self, device: str = "cpu"):
        self.embedder = Embedder(device=device)
        self.preprocessor_salary = Preprocessor(project_root / "artifacts/salary")
        self.preprocessor_seniority = Preprocessor(project_root / "artifacts/seniority")
        self.modelo_salary = joblib.load(project_root / "models/salary/set_2_XGBClassifier.joblib")
        self.modelo_seniority = joblib.load(project_root / "models/seniority/set_1_XGBClassifier.joblib")

        # Forzar CPU para evitar crash si el modelo fue entrenado con GPU
        self.modelo_salary.set_params(device="cpu", tree_method="hist")
        self.modelo_seniority.set_params(device="cpu", tree_method="hist")

        # La fuente de verdad de las features es el propio modelo, no el JSON
        self.preprocessor_salary.feature_columns = list(self.modelo_salary.get_booster().feature_names)
        self.preprocessor_seniority.feature_columns = list(self.modelo_seniority.get_booster().feature_names)
```

**Por qué `get_booster().feature_names`:** el `feature_columns.json` se genera al final del notebook de entrenamiento e incluye todas las columnas del DataFrame, pero `set_2` descarta `tipo_empleo`. El booster del modelo guarda internamente las features exactas con las que fue entrenado, por lo que es la fuente de verdad definitiva.

### Método de predicción

```python
def predecir(self, titulo_habilidades, formacion, sector, tipo_empleo, ciudad) -> dict:
    embedding = self.embedder.embed(titulo_habilidades)

    X_salary = self.preprocessor_salary.procesar(embedding, formacion, sector, tipo_empleo, ciudad)
    X_seniority = self.preprocessor_seniority.procesar(embedding, formacion, sector, tipo_empleo, ciudad)

    pred_salary = int(self.modelo_salary.predict(X_salary)[0])
    pred_seniority = int(self.modelo_seniority.predict(X_seniority)[0])

    return {
        "salario_idx": pred_salary,          # 0-9
        "seniority_idx": pred_seniority,      # 0-2
        "salario_label": RANGOS_SALARIO[pred_salary],
        "seniority_label": NIVELES_SENIORITY[pred_seniority],
    }
```

---

## Diferencias entre modelos de salario y seniority

| Aspecto | Salario (set_2) | Seniority (set_1) |
|---|---|---|
| `tipo_empleo` en OHE | ❌ No (descartado) | ✅ Sí |
| `ciudad` en OHE | ✅ Sí | ❌ No |
| Componentes PCA | 87 | 144 |
| Clases | 10 rangos salariales | 3 niveles |

Aunque el `Preprocessor` siempre recibe `tipo_empleo` y `ciudad`, las columnas que no usa el modelo simplemente se descartan en el paso de reindex final gracias a `feature_columns`.

---

## Rangos y etiquetas

### Salario (índice 0-9)

```
0:  <15.000 €
1:  15.000 - 22.000 €
2:  22.000 - 30.000 €
3:  30.000 - 40.000 €
4:  40.000 - 52.000 €
5:  52.000 - 65.000 €
6:  65.000 - 80.000 €
7:  80.000 - 100.000 €
8:  100.000 - 150.000 €
9:  >150.000 €
```

### Seniority (índice 0-2)

```
0:  Intern
1:  Junior
2:  Senior
```
