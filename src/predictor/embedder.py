from sentence_transformers import SentenceTransformer

"""
Este modulo tiene como objetivo recibir un texto y devolver su embedding utilizando el mismo modelo que se ha utilizado en el entrenamiento de modelos.
"""


class Embedder:
    def __init__(self, device: str = "cpu"):  # en inferencia no se necesita gpu
        self.model = SentenceTransformer(
            "intfloat/multilingual-e5-large", device=device
        )

    def embed(self, texto: str) -> list:
        texto_preparado = "passage: " + texto
        embedding = self.model.encode(
            [texto_preparado], show_progress_bar=False, batch_size=1
        )
        return embedding[0]  # devuelve un array 1D de 1024 dimensiones
