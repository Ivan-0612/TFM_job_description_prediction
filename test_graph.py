import sys

sys.path.append(".")
from src.graph.graph import app

resultado = app.invoke(
    {
        "formacion": "Grado Universitario",
        "sector": "Tecnología",
        "tipo_empleo": "Jornada completa",
        "ciudad": "Madrid",
        "descripcion": "Perfil backend con experiencia en APIs REST",
        "target_salario": None,
        "target_seniority": None,
        "max_iteraciones": 4,
        "titulo_habilidades": "",
        "oferta_completa": "",
        "prediccion_salario": None,
        "prediccion_seniority": None,
        "iteracion": 0,
        "historial": [],
    }
)

print("=== OFERTA GENERADA ===")
print(resultado["oferta_completa"])
print("\n=== PREDICCIONES ===")
print(f"Salario:   {resultado['prediccion_salario']}")
print(f"Seniority: {resultado['prediccion_seniority']}")
