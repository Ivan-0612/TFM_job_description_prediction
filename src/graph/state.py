from typing import TypedDict, Optional, List

"""
Este módulo define la estructura del estado de la oferta de trabajo a lo largo del proceso de generación y ajuste.
"""


class OfertaState(TypedDict):
    # Inputs del usuario (no cambian durante el flujo)
    formacion: str
    sector: str
    tipo_empleo: str
    ciudad: str
    descripcion: str

    # Targets de ajuste (None = sin preferencia)
    target_salario: Optional[int]  # 0-9
    target_seniority: Optional[int]  # 0=intern, 1=junior, 2=senior

    # Output del Proceso 1 (generación)
    titulo_habilidades: str
    oferta_completa: str

    # Output del Proceso 2 (predicción)
    prediccion_salario: Optional[int]
    prediccion_seniority: Optional[int]

    # Control del bucle de ajuste
    iteracion: int
    max_iteraciones: int
    historial: List[dict]
