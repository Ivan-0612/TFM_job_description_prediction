# Sistema RAG: vectorstore y generación de ofertas

Este documento describe el sistema de Retrieval-Augmented Generation (RAG) implementado en `src/rag/`, que combina un índice FAISS de ofertas reales con un LLM para generar ofertas de trabajo sintéticas y realistas.

---

## Visión general

```
Inputs del usuario
(sector, ciudad, tipo_empleo, formacion, descripcion)
        │
        ▼
  Consulta FAISS
  text-embedding-3-small (OpenAI)
  → top-5 documentos similares
        │
        ▼
  Construcción del prompt
  + contexto de las 5 ofertas reales
  + perfil solicitado
        │
        ▼
  GPT-4o-mini (temperature=0.3)
        │
        ▼
  Parser de respuesta
        │
        ├──→ titulo_habilidades (título + lista de habilidades)
        └──→ oferta_completa (texto del anuncio, 100-150 palabras)
```

---

## Vectorstore FAISS (`src/rag/vectorstore.py`)

### Construcción (`build_vectorstore.py`)

El índice FAISS se construye una única vez ejecutando `python build_vectorstore.py`. El proceso:

1. Lee `data/interim/df_merged.csv` y elimina filas sin `Titulo_Habilidades`.
2. Procesa las ofertas en lotes de 500 para respetar los límites de TPM de la API de OpenAI.
3. Vectoriza cada oferta con `text-embedding-3-small` (modelo de OpenAI, 1536 dims).
4. El `page_content` de cada documento es el campo `Titulo_Habilidades` (título + habilidades de la oferta original).
5. Los metadatos incluyen: `sector`, `ciudad`, `tipo_empleo`, `formacion`, `salario`, `seniority`.
6. Guarda el índice en `vectorstore/ofertas_index/` tras cada lote (tolerante a fallos: si se interrumpe, reanuda desde donde se quedó).
7. Pausa de 8 segundos entre lotes para evitar superar el límite de TPM.

El índice resultante contiene **~36.000 ofertas** vectorizadas. Se generan dos ficheros:
- `index.faiss` — el índice propiamente dicho.
- `index.pkl` — el docstore con los documentos y sus metadatos.

Ambos ficheros son necesarios. **Coste estimado: < 0,05 € en la API de OpenAI.**

### Por qué `Titulo_Habilidades` como `page_content`

Este campo es la concatenación del título del puesto y las habilidades requeridas, extraído durante el preprocesado con IA (notebook `03_process_data_ai`). Concentra la información semántica más relevante para la búsqueda, sin el ruido de los textos largos de las descripciones completas.

### Carga del vectorstore

```python
def cargar_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    return vectorstore
```

`allow_dangerous_deserialization=True` es necesario porque FAISS usa pickle internamente. Es seguro en este contexto ya que el índice lo generamos nosotros mismos.

---

## `GeneradorOfertas` (`src/rag/generator.py`)

### Inicialización

```python
class GeneradorOfertas:
    def __init__(self):
        self.vectorstore = cargar_vectorstore()     # carga el índice FAISS
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.parser = StrOutputParser()
```

`temperature=0.3` produce respuestas más consistentes y predecibles, manteniendo algo de variabilidad para que las ofertas no sean idénticas.

### Método `generar()`

**Paso 1 — Recuperación (Retrieval)**

Se construye una query combinando sector, ciudad y descripción del usuario, y se recuperan las 5 ofertas más similares del índice:

```python
query = f"{sector} {ciudad} {descripcion}"
documentos = self.vectorstore.similarity_search(query, k=5)
```

**Paso 2 — Construcción del contexto**

Las 5 ofertas recuperadas se formatean en texto para pasarlas al prompt:

```python
contexto = "\n---\n".join([
    f"Título y habilidades: {doc.page_content}\n"
    f"Sector: {doc.metadata['sector']} | Ciudad: {doc.metadata['ciudad']} | Seniority: {doc.metadata['seniority']}"
    for doc in documentos
])
```

**Paso 3 — Generación (Generation)**

La cadena LangChain `prompt | llm | parser` genera la respuesta:

```python
chain = self.prompt | self.llm | self.parser
respuesta = chain.invoke({
    "contexto": contexto,
    "sector": sector,
    "ciudad": ciudad,
    "tipo_empleo": tipo_empleo,
    "formacion": formacion,
    "descripcion": descripcion,
})
```

### Prompt de generación

```
Eres un experto en recursos humanos del mercado laboral español.
Tu tarea es generar una oferta de trabajo realista basándote en el perfil solicitado.

Estas son ofertas reales similares del mercado como referencia:
{contexto}

Genera una oferta para el siguiente perfil:
- Sector: {sector}
- Ciudad: {ciudad}
- Tipo de empleo: {tipo_empleo}
- Formación requerida: {formacion}
- Descripción adicional: {descripcion}

Responde EXACTAMENTE en este formato, sin añadir nada más:

TITULO: <título del puesto de trabajo>
HABILIDADES: <lista de habilidades técnicas y blandas separadas por comas>
OFERTA:
<texto completo del anuncio de trabajo, entre 100 y 150 palabras>
```

El formato estructurado con etiquetas (`TITULO:`, `HABILIDADES:`, `OFERTA:`) facilita el parsing determinista de la respuesta.

**Paso 4 — Parsing**

```python
def _parsear_respuesta(self, respuesta: str) -> dict:
    # Extrae líneas TITULO: y HABILIDADES:
    # Acumula todo lo que hay tras OFERTA:
    return {
        "titulo_habilidades": f"{titulo} {habilidades}",
        "oferta_completa": "\n".join(oferta_lines).strip(),
    }
```

El `titulo_habilidades` concatena título y habilidades en una sola cadena, que es la entrada al pipeline de predicción. Esta concatenación es coherente con cómo se construyó el campo en el dataset de entrenamiento.

---

## Método `regenerar_oferta()`

Después de que el bucle de ajuste LangGraph modifica el `titulo_habilidades`, el texto del anuncio (`oferta_completa`) puede haber quedado incoherente. Este método regenera solo el texto del anuncio sin hacer búsqueda en el vectorstore:

```python
def regenerar_oferta(self, titulo_habilidades, sector, ciudad, tipo_empleo, formacion) -> str:
    prompt = ChatPromptTemplate.from_template("""
Eres un experto en recursos humanos del mercado laboral español.
Redacta el texto completo de un anuncio de trabajo (entre 100 y 150 palabras)
basándote en el siguiente título y habilidades:

Titulo y habilidades: {titulo_habilidades}
...
""")
    chain = prompt | self.llm | self.parser
    return chain.invoke({...})
```

Este método se llama desde `app.py` cuando `resultado_ajustado["iteracion"] > 0`, es decir, cuando el bucle hizo al menos un ajuste.

---

## Separación de responsabilidades RAG vs. Predictor

| Aspecto | RAG (`src/rag/`) | Predictor (`src/predictor/`) |
|---|---|---|
| **Embedding** | `text-embedding-3-small` (OpenAI, 1536 dims) | `multilingual-e5-large` (SBERT, 1024 dims) |
| **Propósito** | Buscar ofertas similares para dar contexto al LLM | Representar semánticamente la oferta para XGBoost |
| **API** | OpenAI (requiere clave) | Local (no requiere API) |
| **Cuándo se usa** | En generación inicial y carga de contexto | En cada predicción |
| **Coste** | Proporcional al uso | Gratuito (inferencia local) |

Esta separación es deliberada: el modelo de predicción fue entrenado con SBERT y no puede usar embeddings de OpenAI en inferencia sin reentrenar.
