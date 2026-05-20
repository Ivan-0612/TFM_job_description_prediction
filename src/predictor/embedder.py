import os
import numpy as np

"""
Embedder con doble modo:
- API (Render/producción): usa la Inference API de Hugging Face cuando HF_TOKEN está en el entorno.
  No descarga el modelo (~1,1 GB), hace llamadas HTTP al endpoint de HF.
- Local (desarrollo): carga sentence-transformers localmente si HF_TOKEN no está definido.

En ambos casos se usa el mismo modelo (intfloat/multilingual-e5-large) con el
mismo prefijo "passage: " y normalización L2, para que los embeddings sean
compatibles con los artefactos de entrenamiento.
"""


class Embedder:
    def __init__(self, device: str = "cpu"):
        self.hf_token = os.getenv("HF_TOKEN")

        if self.hf_token:
            # ── MODO API (Render) ──────────────────────────────────────────────
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.hf_token)
            self._modo = "api"
            print("Embedder: usando Hugging Face Inference API")
        else:
            # ── MODO LOCAL (desarrollo) ────────────────────────────────────────
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(
                "intfloat/multilingual-e5-large", device=device
            )
            self._modo = "local"
            print("Embedder: usando SBERT local")

    def embed(self, texto: str) -> np.ndarray:
        texto_preparado = "passage: " + texto
        if self._modo == "api":
            return self._embed_api(texto_preparado)
        else:
            return self._embed_local(texto_preparado)

    def _embed_api(self, texto: str) -> np.ndarray:
        """Llama a la HF Inference API y devuelve un array 1D normalizado de 1024 dims."""
        resultado = self.client.feature_extraction(
            texto,
            model="intfloat/multilingual-e5-large",
        )
        embedding = np.array(resultado)

        # La API puede devolver (seq_len, hidden) o (1, seq_len, hidden) según el modelo.
        # Aplicamos mean pooling para obtener un único vector de frase.
        if embedding.ndim == 3:          # (batch, seq_len, hidden) → quitar dim batch
            embedding = embedding.squeeze(0)
        if embedding.ndim == 2:          # (seq_len, hidden) → mean pooling
            embedding = embedding.mean(axis=0)
        # Ahora embedding.shape == (1024,)

        # Normalización L2 (igual que sentence-transformers con normalize_embeddings=True)
        norma = np.linalg.norm(embedding)
        if norma > 0:
            embedding = embedding / norma

        return embedding

    def _embed_local(self, texto: str) -> np.ndarray:
        """Usa sentence-transformers localmente. Devuelve array 1D de 1024 dims."""
        embedding = self.model.encode(
            [texto], show_progress_bar=False, batch_size=1, normalize_embeddings=True
        )
        return embedding[0]
