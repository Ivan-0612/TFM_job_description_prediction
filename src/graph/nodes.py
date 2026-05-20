from src.rag.generator import GeneradorOfertas
from src.predictor.predictor import Predictor
from src.graph.state import OfertaState
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Lazy initialization — se instancian solo cuando se usan por primera vez
_generador = None
_predictor = None


def _get_generador():
    global _generador
    if _generador is None:
        _generador = GeneradorOfertas()
    return _generador


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = Predictor(device="cpu")
    return _predictor


RANGOS_SALARIO = [
    "<15.000",
    "15.000-22.000",
    "22.000-30.000",
    "30.000-40.000",
    "40.000-52.000",
    "52.000-65.000",
    "65.000-80.000",
    "80.000-100.000",
    "100.000-150.000",
    ">150.000",
]
NIVELES_SENIORITY = ["Intern", "Junior", "Senior"]


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


def nodo_predecir(state: OfertaState) -> OfertaState:
    resultado = _get_predictor().predecir(
        titulo_habilidades=state["titulo_habilidades"],
        formacion=state["formacion"],
        sector=state["sector"],
        tipo_empleo=state["tipo_empleo"],
        ciudad=state["ciudad"],
    )
    return {
        **state,
        "prediccion_salario": resultado["salario_idx"],
        "prediccion_seniority": resultado["seniority_idx"],
    }


def nodo_verificar(state: OfertaState) -> str:
    """Nodo condicional: devuelve 'ajustar' o 'fin'."""
    if state["iteracion"] >= state["max_iteraciones"]:
        return "fin"

    if state["target_salario"] is None and state["target_seniority"] is None:
        return "fin"

    salario_ok = (
        state["target_salario"] is None
        or state["prediccion_salario"] == state["target_salario"]
    )
    seniority_ok = (
        state["target_seniority"] is None
        or state["prediccion_seniority"] == state["target_seniority"]
    )

    return "fin" if (salario_ok and seniority_ok) else "ajustar"


def nodo_ajustar(state: OfertaState) -> OfertaState:
    """Pide al LLM que reescriba el titulo_habilidades para acercarse al objetivo."""

    historial_texto = ""
    for intento in state["historial"]:
        historial_texto += (
            f"- Intento {intento['iteracion']}: '{intento['titulo_habilidades']}' "
            f"→ salario {intento['salario']} / seniority {intento['seniority']}\n"
        )

    llm_ajuste = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    prompt = ChatPromptTemplate.from_template("""
Eres un experto en redacción de ofertas de trabajo del mercado laboral español.

La oferta actual tiene este título y habilidades:
{titulo_habilidades_actual}

El modelo de predicción ha predicho:
- Salario: {salario_actual}
- Seniority: {seniority_actual}

El objetivo es conseguir:
- Salario objetivo: {salario_objetivo}
- Seniority objetivo: {seniority_objetivo}

{historial}

Modifica el título y las habilidades para que se acerquen al objetivo.
Ten en cuenta los intentos anteriores para no repetir los mismos errores.

Responde EXACTAMENTE en este formato:
TITULO: <nuevo título>
HABILIDADES: <nuevas habilidades separadas por comas>
""")

    chain = prompt | llm_ajuste | StrOutputParser()
    respuesta = chain.invoke(
        {
            "titulo_habilidades_actual": state["titulo_habilidades"],
            "salario_actual": RANGOS_SALARIO[state["prediccion_salario"]],
            "seniority_actual": NIVELES_SENIORITY[state["prediccion_seniority"]],
            "salario_objetivo": (
                RANGOS_SALARIO[state["target_salario"]]
                if state["target_salario"] is not None
                else "Sin cambio"
            ),
            "seniority_objetivo": (
                NIVELES_SENIORITY[state["target_seniority"]]
                if state["target_seniority"] is not None
                else "Sin cambio"
            ),
            "historial": (
                f"Intentos anteriores:\n{historial_texto}" if historial_texto else ""
            ),
        }
    )

    titulo, habilidades = "", ""
    for linea in respuesta.strip().split("\n"):
        if linea.startswith("TITULO:"):
            titulo = linea.replace("TITULO:", "").strip()
        elif linea.startswith("HABILIDADES:"):
            habilidades = linea.replace("HABILIDADES:", "").strip()

    nuevo_titulo_habilidades = f"{titulo} {habilidades}"

    nuevo_historial = state["historial"] + [
        {
            "iteracion": state["iteracion"],
            "titulo_habilidades": state["titulo_habilidades"],
            "salario": RANGOS_SALARIO[state["prediccion_salario"]],
            "seniority": NIVELES_SENIORITY[state["prediccion_seniority"]],
        }
    ]

    return {
        **state,
        "titulo_habilidades": nuevo_titulo_habilidades,
        "iteracion": state["iteracion"] + 1,
        "historial": nuevo_historial,
    }
