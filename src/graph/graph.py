from langgraph.graph import StateGraph, END
from src.graph.state import OfertaState
from src.graph.nodes import nodo_generar, nodo_predecir, nodo_verificar, nodo_ajustar

"""
Este módulo define el grafo de estados para el proceso de generación y ajuste de ofertas de trabajo.
El grafo tiene el siguiente flujo:
1. El nodo "generar" crea una oferta inicial a partir del perfil del usuario.
2. El nodo "predecir" evalúa la oferta generada y predice su salario y seniority.
3. El nodo "verificar" decide si la oferta cumple los objetivos o si necesita ajuste
4. Si se necesita ajuste, el nodo "ajustar" modifica el título y habilidades para acercarse a los objetivos, y luego vuelve a predecir.
El proceso se repite hasta que la oferta cumple los objetivos o se alcanza el máximo de iteraciones.
"""


def construir_grafo():
    workflow = StateGraph(OfertaState)

    # Registrar nodos
    workflow.add_node("generar", nodo_generar)
    workflow.add_node("predecir", nodo_predecir)
    workflow.add_node("ajustar", nodo_ajustar)

    # Definir el flujo
    workflow.set_entry_point("generar")
    workflow.add_edge("generar", "predecir")

    # Nodo condicional: después de predecir, verificar si hay que ajustar
    workflow.add_conditional_edges(
        "predecir", nodo_verificar, {"ajustar": "ajustar", "fin": END}
    )

    # El bucle: después de ajustar, volver a predecir
    workflow.add_edge("ajustar", "predecir")

    return workflow.compile()


# Instancia del grafo lista para usar
app = construir_grafo()
