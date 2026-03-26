# TFM proyecto: Predicción de salario, seniority y sistema RAG para ofertas de empleo

Proyecto de Trabajo Fin de Máster orientado a la **recogida, limpieza y preparación de ofertas de empleo**, con tres objetivos aplicados de IA/NLP:

1. **Entrenar un modelo para predecir salario**.
2. **Entrenar un modelo para predecir seniority**.
3. **Construir un sistema RAG** con ofertas de empleo + LLM para **generar ofertas optimizadas**.

## 1. Objetivo del proyecto
Este TFM busca desarrollar un pipeline completo de datos y modelado que permita:

1. Extraer y consolidar ofertas de empleo desde múltiples fuentes.
2. Preparar datasets limpios y estructurados para tareas supervisadas.
3. Entrenar y evaluar:
   - Un modelo de **predicción de salario**.
   - Un modelo de **clasificación/predicción de seniority**.
4. Implementar un sistema **RAG (Retrieval-Augmented Generation)** que recupere ofertas relevantes y, apoyado por un LLM, proponga **versiones optimizadas de ofertas de trabajo**.

## 2. Estructura del repositorio
```text
README.md
requirements.txt
data/
  raw/         # Datos originales por fuente (CSV, parciales, etc.)
  interim/     # Ficheros intermedios tras primeras transformaciones
  processed/   # Datos finales limpios para análisis/modelado
  external/    # Datos externos auxiliares
docs/          # Documentación adicional
models/        # Modelos entrenados y artefactos
notebooks/     # Notebooks del flujo principal (00, 01, 02, 03...)
references/    # Material de referencia
reports/
  figures/     # Gráficas y visualizaciones para informe/memoria
src/
  scraping_functions.py  # Funciones de scraping y utilidades
```

## 3. Fuentes de datos (resumen)
Fuentes integradas en el proyecto:

- **Adzuna API**
- **Apify / LinkedIn jobs scraper**
- **Glassdoor** 

## 4. Flujo de trabajo (alto nivel)
1. **01_get_data.ipynb**  
   Ingesta y recopilación de ofertas.
2. **02_build_data.ipynb**  
   Integración de fuentes y construcción del dataset base.
3. **03_process_data.ipynb**  
   Limpieza, transformación y generación del dataset final para modelado.
4. **Modelado ML**  
   Entrenamiento y evaluación de modelos de salario y seniority.
5. **Sistema RAG + LLM**  
   Indexación/recuperación de ofertas y generación de ofertas optimizadas.

## 5. Líneas de modelado
### 5.1 Predicción de salario
- Tipo de tarea: regresión.
- Variable objetivo: salario (normalizado según disponibilidad/calidad de campo).
- Salida esperada: estimación salarial para nuevas ofertas.

### 5.2 Predicción de seniority
- Tipo de tarea: clasificación multiclase (p. ej. `intern`, `junior`, `senior`).
- Variable objetivo: nivel de experiencia/seniority.
- Salida esperada: nivel de seniority estimado para una oferta.

### 5.3 RAG para generación de ofertas optimizadas
- Base documental: corpus de ofertas limpias y estructuradas.
- Recuperación: búsqueda de ofertas similares/relevantes.
- Generación (LLM): redacción de una oferta optimizada usando contexto recuperado.
- Resultado: borradores de ofertas más claros, completos y alineados con el mercado.

## 6. Requisitos
Dependencias principales en:

- requirements.txt

## 7. Cómo ejecutar (versión básica)
1. Clonar el repositorio.
2. Crear entorno virtual (recomendado).
3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar notebooks en orden:
   - `00_pruebas.ipynb` (opcional)
   - `01_get_data.ipynb`
   - `02_build_data.ipynb`
   - `03_process_data.ipynb`
5. Ejecutar scripts/notebooks de modelado 

## 8. Resultados esperados
- Dataset consolidado y limpio para tareas de ML/NLP.
- Modelo de predicción de salario con métricas de evaluación.
- Modelo de predicción de seniority con métricas de clasificación.
- Prototipo funcional de sistema RAG + LLM para generación de ofertas optimizadas.

## 10. Autoría
Proyecto desarrollado como parte del **Trabajo Fin de Máster (TFM)**.  
Autor: *Iván Benito Sánchez*  
