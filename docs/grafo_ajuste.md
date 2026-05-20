# Grafo LangGraph: bucle de ajuste de ofertas

Este documento describe el grafo de estados implementado con LangGraph (`src/graph/`) que permite ajustar iterativamente una oferta de trabajo hasta que las predicciones se alineen con los objetivos del usuario.

---

## Motivación

Una vez generada la oferta y predichos el salario y seniority, puede ocurrir que la predicción no coincida con lo que el usuario necesita. Por ejemplo, la empresa quiere una oferta que prediga nivel "Senior" pero el modelo devuelve "Junior". El grafo resuelve esto reescribiendo el `titulo_habilidades` iterativamente hasta que la predicción converja al objetivo.

---

## Estado (`src/graph/state.py`)

Todo el flujo de información entre nodos se comunica a través de `OfertaState`, un `TypedDict` con los siguientes campos:

```python
class OfertaState(TypedDict):
    # Inputs del usuario (inmutables durante el flujo)
    formacion: str
    sector: str
    tipo_empleo: str
    ciudad: str
    descripcion: str

    # Objetivos de ajuste (None = sin preferencia)
    target_salario: Optional[int]    # índice 0-9
    target_seniority: Optional[int]  # 0=Intern, 1=Junior, 2=Senior

    # Output del generador RAG
    titulo_habilidades: str
    oferta_completa: str

    # Output del predictor
    prediccion_salario: Optional[int]
    prediccion_seniority: Optional[int]

    # Control del bucle
    iteracion: int
    max_iteraciones: int
    historial: List[dict]   # historial de intentos anteriores
```

El campo `historial` acumula los intentos anteriores con su `titulo_habilidades`, salario y seniority predichos. Se pasa al LLM de ajuste para que no repita los mismos errores.

---

## Nodos (`src/graph/nodes.py`)

### `nodo_generar`

Llama al `GeneradorOfertas` con los parámetros del usuario y almacena el resultado en el estado.

```python
def nodo_generar(state: OfertaState) -> OfertaState:
    resultado = _get_generador().generar(
        sector=state["sector"],
        ciudad=state["ciudad"],
        tipo_empleo=state["tipo_empleo"],
        formacion=state["formacion"],
        descripcion=state["descripcion"],
    )
    return {
        **state,
        "titulo_habilidades": resultado["titulo_habilidades"],
        "oferta_completa": resultado["oferta_completa"],
        "iteracion": 0,
        "historial": [],
    }
```

### `nodo_predecir`

Llama al `Predictor` y actualiza las predicciones en el estado.

```python
def nodo_predecir(state: OfertaState) -> OfertaState:
    resultado = _get_predictor().predecir(
        titulo_habilidades=state["titulo_habilidades"],
        ...
    )
    return {
        **state,
        "prediccion_salario": resultado["salario_idx"],
        "prediccion_seniority": resultado["seniority_idx"],
    }
```

### `nodo_verificar` (nodo condicional)

Devuelve un string (`"ajustar"` o `"fin"`) que LangGraph usa para decidir el siguiente nodo. No modifica el estado.

```python
def nodo_verificar(state: OfertaState) -> str:
    # Terminar si se alcanzó el límite de iteraciones
    if state["iteracion"] >= state["max_iteraciones"]:
        return "fin"

    # Terminar si no hay objetivos definidos
    if state["target_salario"] is None and state["target_seniority"] is None:
        return "fin"

    salario_ok = (state["target_salario"] is None or
                  state["prediccion_salario"] == state["target_salario"])
    seniority_ok = (state["target_seniority"] is None or
                    state["prediccion_seniority"] == state["target_seniority"])

    return "fin" if (salario_ok and seniority_ok) else "ajustar"
```

### `nodo_ajustar`

Pide al LLM que reescriba el `titulo_habilidades` para acercarse al objetivo, usando el historial de intentos anteriores para evitar repetir errores.

El prompt incluye:
- El `titulo_habilidades` actual.
- Las predicciones actuales (en etiquetas legibles, no índices).
- Los objetivos en etiquetas legibles.
- El historial de intentos anteriores.

```python
def nodo_ajustar(state: OfertaState) -> OfertaState:
    # ...construye historial_texto con intentos anteriores...
    llm_ajuste = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    # ...llama al LLM y parsea TITULO: + HABILIDADES:...
    return {
        **state,
        "titulo_habilidades": nuevo_titulo_habilidades,
        "iteracion": state["iteracion"] + 1,
        "historial": nuevo_historial,
    }
```

`temperature=0.5` es ligeramente mayor que en la generación inicial, para que el LLM explore más variantes al ajustar.

**Por qué el LLM de ajuste se instancia dentro del nodo:** si se instanciara a nivel de módulo, se cargaría al importar `nodes.py`, lo que causaría doble carga de recursos en Streamlit.

---

## Grafo (`src/graph/graph.py`)

```python
def construir_grafo():
    workflow = StateGraph(OfertaState)

    workflow.add_node("generar", nodo_generar)
    workflow.add_node("predecir", nodo_predecir)
    workflow.add_node("ajustar", nodo_ajustar)

    workflow.set_entry_point("generar")
    workflow.add_edge("generar", "predecir")

    # Nodo condicional: tras predecir, verificar si ajustar o terminar
    workflow.add_conditional_edges(
        "predecir",
        nodo_verificar,
        {"ajustar": "ajustar", "fin": END}
    )

    # Bucle: tras ajustar, volver a predecir
    workflow.add_edge("ajustar", "predecir")

    return workflow.compile()
```

El grafo compila un objeto `CompiledStateGraph` con soporte explícito para ciclos, lo que permite el bucle `ajustar → predecir → verificar → ajustar...`.

---

## Flujo completo

```
[START]
   │
   ▼
[generar]
  Genera titulo_habilidades + oferta_completa con RAG
  Inicializa iteracion=0, historial=[]
   │
   ▼
[predecir]
  Predice salario_idx y seniority_idx
   │
   ▼
[verificar] ──── ¿objetivos cumplidos o max_iter alcanzado?
   │                                │
   │ "ajustar"                      │ "fin"
   ▼                                ▼
[ajustar]                        [END]
  LLM reescribe titulo_habilidades
  Incrementa iteracion
  Actualiza historial
   │
   └──────────────────────────────▶ [predecir]
```

---

## Uso desde `app.py`

En la aplicación Streamlit, el grafo se usa en el bloque de ajuste. Se invoca con el estado actual de la sesión (incluyendo la oferta ya generada y las predicciones previas), lo que evita regenerar la oferta desde cero:

```python
from src.graph.graph import construir_grafo
grafo = construir_grafo()

resultado_ajustado = grafo.invoke({
    "formacion": r["formacion"],
    "sector": r["sector"],
    "tipo_empleo": r["tipo_empleo"],
    "ciudad": r["ciudad"],
    "descripcion": r["descripcion"],
    "titulo_habilidades": r["titulo_habilidades"],   # oferta actual
    "oferta_completa": r["oferta_completa"],
    "prediccion_salario": r["prediccion_salario"],   # predicciones actuales
    "prediccion_seniority": r["prediccion_seniority"],
    "target_salario": target_salario,                # objetivos del usuario
    "target_seniority": target_seniority,
    "max_iteraciones": 4,
    "iteracion": 0,
    "historial": [],
})
```

**Nota importante:** el grafo parte desde `nodo_generar`, lo que implica que siempre regernera la oferta desde cero internamente. Sin embargo, dado que `nodo_generar` hace una llamada RAG, la oferta resultante puede diferir ligeramente. El ajuste posterior sobre `titulo_habilidades` es el que realmente empuja hacia el objetivo.

Tras el ajuste, si `resultado_ajustado["iteracion"] > 0`, se llama a `regenerar_oferta()` para reescribir el texto del anuncio de forma coherente con el nuevo `titulo_habilidades`.

---

## Límites y comportamiento esperado

- **Máximo de iteraciones:** 4. Si no se alcanza el objetivo, se devuelve el mejor resultado obtenido.
- **Convergencia:** el LLM no garantiza converger siempre al objetivo exacto, especialmente si el objetivo es muy específico o contradictorio con el sector/formación. El historial ayuda a evitar oscilaciones.
- **Coste por ajuste:** cada iteración consume ~2-3 llamadas a la API de OpenAI (verificación + ajuste). Con max_iteraciones=4, el coste máximo por ajuste es de ~8-12 llamadas.
