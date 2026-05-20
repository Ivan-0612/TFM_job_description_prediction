import sys

sys.path.append(".")
from src.rag.generator import GeneradorOfertas

generador = GeneradorOfertas()
resultado = generador.generar(
    sector="Tecnología",
    ciudad="Madrid",
    tipo_empleo="Jornada completa",
    formacion="Grado Universitario",
    descripcion="Necesito un perfil de backend con experiencia en APIs REST",
)

print("=== TITULO + HABILIDADES ===")
print(resultado["titulo_habilidades"])
print("\n=== OFERTA COMPLETA ===")
print(resultado["oferta_completa"])
