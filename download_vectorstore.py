"""
Script para descargar el vectorstore de Google Drive si no existe en el disco.
Se ejecuta automáticamente al arrancar la app en Render.

INSTRUCCIONES PARA CONFIGURAR:
1. Sube 'index.faiss' e 'index.pkl' a Google Drive.
2. Haz clic derecho en cada fichero → "Compartir" → "Cualquier persona con el enlace".
3. Copia el enlace. Tendrá el formato:
      https://drive.google.com/file/d/XXXXXXXXXXXXXXXXXXXXXXXX/view?usp=sharing
4. El ID del fichero es la parte XXXXXXXXXXXXXXXXXXXXXXXX.
5. Sustituye los valores de FAISS_FILE_ID e INDEX_PKL_FILE_ID abajo.
"""

import os
import gdown

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
# Sustituye estos IDs por los de tus ficheros en Google Drive
FAISS_FILE_ID = "1I14K3SELzt09GYflZFFKvP_bNdx7g44r"
PKL_FILE_ID = "1okFdMXQW2jWi_VVHdv_m6ozSYizOmwdR"
# ──────────────────────────────────────────────────────────────────────────────

VECTORSTORE_DIR = "vectorstore/ofertas_index"
FAISS_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
PKL_PATH = os.path.join(VECTORSTORE_DIR, "index.pkl")


def descargar_vectorstore():
    """Descarga el vectorstore desde Google Drive si no existe en disco."""

    if os.path.exists(FAISS_PATH) and os.path.exists(PKL_PATH):
        print("✅ Vectorstore ya existe en disco, no es necesario descargarlo.")
        return

    print("📥 Vectorstore no encontrado. Descargando desde Google Drive...")
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    if not os.path.exists(FAISS_PATH):
        print("   Descargando index.faiss (~212 MB)...")
        url = f"https://drive.google.com/uc?id={FAISS_FILE_ID}"
        gdown.download(url, FAISS_PATH, quiet=False)
        print("   ✅ index.faiss descargado.")

    if not os.path.exists(PKL_PATH):
        print("   Descargando index.pkl (~10 MB)...")
        url = f"https://drive.google.com/uc?id={PKL_FILE_ID}"
        gdown.download(url, PKL_PATH, quiet=False)
        print("   ✅ index.pkl descargado.")

    print("✅ Vectorstore listo.")


if __name__ == "__main__":
    descargar_vectorstore()
