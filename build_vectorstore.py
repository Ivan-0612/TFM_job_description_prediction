import sys

"""
Este script se encarga de construir el vectorstore de FAISS a partir del dataset de ofertas utilizando vectorstor.py
"""

sys.path.append(".")
from src.rag.vectorstore import construir_vectorstore

construir_vectorstore()
