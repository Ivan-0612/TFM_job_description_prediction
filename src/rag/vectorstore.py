import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

"""
Este módulo se encarga de construir y cargar el vectorstore de FAISS a partir del dataset de ofertas.
El vectorstore se utiliza para recuperar las ofertas más similares a la consulta del usuario, y así
dar contexto al LLM para la generación de respuestas.
"""

load_dotenv()

VECTORSTORE_PATH = "vectorstore/ofertas_index"


import time


def construir_vectorstore(
    csv_path: str = "data/interim/df_merged.csv", batch_size: int = 500
):
    """
    Construye el vectorstore de forma incremental guardando el progreso en disco.
    Evita que la API se bloquee y permite reanudar si se congela.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Titulo_Habilidades"])
    df["Titulo_Habilidades"] = df["Titulo_Habilidades"].astype(str)

    total_filas = len(df)
    print(f"Total de ofertas válidas en el CSV: {total_filas}")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        chunk_size=500,
    )

    # Intentar cargar el progreso guardado anteriormente
    if os.path.exists(VECTORSTORE_PATH):
        print(f"Cargando progreso desde {VECTORSTORE_PATH}...")
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        filas_procesadas = len(vectorstore.index_to_docstore_id)
        print(f"Reanudando la ejecución desde la fila {filas_procesadas}...")
    else:
        print("No se encontró progreso previo. Iniciando desde cero...")
        vectorstore = None
        filas_procesadas = 0

    # Procesar secuencialmente desde donde se quedó
    for i in range(filas_procesadas, total_filas, batch_size):
        batch_df = df.iloc[i : i + batch_size]
        print(
            f"Procesando filas {i} a {min(i + batch_size, total_filas)} de {total_filas}..."
        )

        # Crear documentos
        documentos_batch = batch_df.apply(
            lambda row: Document(
                page_content=row["Titulo_Habilidades"],
                metadata={
                    "sector": str(row.get("sector", "")),
                    "ciudad": str(row.get("Ciudad", "")),
                    "tipo_empleo": str(row.get("tipo_de_empleo", "")),
                    "formacion": str(row.get("formación_académica", "")),
                    "salario": str(row.get("Rango_Salarial", "")),
                    "seniority": str(row.get("experiencia", "")),
                },
            ),
            axis=1,
        ).tolist()

        # Indexar en FAISS
        if vectorstore is None:
            vectorstore = FAISS.from_documents(documentos_batch, embeddings)
        else:
            vectorstore.add_documents(documentos_batch)

        # Guardar en disco
        os.makedirs(os.path.dirname(VECTORSTORE_PATH), exist_ok=True)
        vectorstore.save_local(VECTORSTORE_PATH)
        print(f"Progreso guardado")

        # Pausa obligatoria para resetear la ventana de Tokens por Minuto (TPM) de OpenAI
        time.sleep(8)

    print(f"Finalizado. {total_filas} ofertas indexadas.")
    return vectorstore


def cargar_vectorstore():
    """
    Carga el índice FAISS ya construido desde disco.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore
