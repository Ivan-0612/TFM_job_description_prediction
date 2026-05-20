import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

"""
Preprocesa los datos de entrada para el modelo de predicción de salario y el modelo de seniority
"""


class Preprocessor:
    def __init__(self, artifacts_dir: str):
        artifacts_path = Path(artifacts_dir)

        self.pca = joblib.load(artifacts_path / "pca.joblib")
        self.scaler = joblib.load(artifacts_path / "scaler.joblib")

        with open(artifacts_path / "feature_columns.json", "r") as f:
            self.feature_columns = json.load(f)

    def procesar(
        self,
        embedding: np.ndarray,
        formacion: str,
        sector: str,
        tipo_empleo: str,
        ciudad: str,
    ) -> np.ndarray:

        # 1. Reducir embedding con PCA
        embedding_reducido = self.pca.transform([embedding])
        df_embedding = pd.DataFrame(
            embedding_reducido,
            columns=[f"PC{i+1}" for i in range(embedding_reducido.shape[1])],
        )

        # 2. Construir OHE con los mismos prefijos que en entrenamiento
        df_struct = pd.DataFrame(
            [
                {
                    "formación_académica": formacion,
                    "sector": sector,
                    "tipo_de_empleo": tipo_empleo,
                    "Ciudad": ciudad,
                }
            ]
        )
        df_ohe = pd.concat(
            [
                pd.get_dummies(
                    df_struct["formación_académica"],
                    prefix="formacion",
                    drop_first=True,
                    dummy_na=True,
                ).astype(int),
                pd.get_dummies(
                    df_struct["sector"], prefix="sector", drop_first=True
                ).astype(int),
                pd.get_dummies(
                    df_struct["tipo_de_empleo"],
                    prefix="tipo_empleo",
                    drop_first=True,
                    dummy_na=True,
                ).astype(int),
                pd.get_dummies(
                    df_struct["Ciudad"], prefix="ciudad", drop_first=True
                ).astype(int),
            ],
            axis=1,
        )
        df_ohe.columns = df_ohe.columns.str.replace(",", "_")

        # 3. Escalar usando exactamente las columnas que vio el scaler
        scaler_cols = [c.replace(",", "_") for c in self.scaler.feature_names_in_]
        df_ohe_scaled = pd.DataFrame(
            self.scaler.transform(
                df_ohe.reindex(columns=scaler_cols, fill_value=0).values
            ),
            columns=scaler_cols,
        )

        # 4. Concatenar OHE escalado + embeddings reducidos
        df_full = pd.concat([df_ohe_scaled, df_embedding], axis=1)

        # 5. Seleccionar solo las features que espera el modelo
        df_final = df_full.reindex(columns=self.feature_columns, fill_value=0)

        return df_final.values
