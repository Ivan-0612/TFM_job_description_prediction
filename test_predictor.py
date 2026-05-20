import sys

sys.path.append(".")

from src.predictor.predictor import Predictor

predictor = Predictor(device="cpu")

resultado = predictor.predecir(
    titulo_habilidades="Data Engineer Python Spark SQL Airflow AWS",
    formacion="Grado Universitario",
    sector="Tecnología",
    tipo_empleo="Jornada completa",
    ciudad="Madrid",
)

print("=== Resultado ===")
print(f"Salario predicho:   {resultado['salario_label']}")
print(f"Seniority predicho: {resultado['seniority_label']}")
print(f"Índice salario:     {resultado['salario_idx']}")
print(f"Índice seniority:   {resultado['seniority_idx']}")
