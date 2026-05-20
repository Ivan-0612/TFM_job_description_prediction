import numpy as np
from src.predictor.embedder import Embedder
from src.predictor.preprocessor import Preprocessor
import joblib
from pathlib import Path

RANGOS_SALARIO = [
    "<15.000",
    "15.000-22.000",
    "22.000-30.000",
    "30.000-40.000",
    "40.000-52.000",
    "52.000-65.000",
    "65.000-80.000",
    "80.000-100.000",
    "100.000-150.000",
    ">150.000",
]

NIVELES_SENIORITY = ["Intern", "Junior", "Senior"]


class Predictor:
    def __init__(self, device: str = "cpu"):
        project_root = Path(__file__).resolve().parents[2]

        self.embedder = Embedder(device=device)

        self.preprocessor_salary = Preprocessor(project_root / "artifacts" / "salary")
        self.preprocessor_seniority = Preprocessor(
            project_root / "artifacts" / "seniority"
        )

        self.modelo_salary = joblib.load(
            project_root / "models" / "salary" / "set_2_XGBClassifier.joblib"
        )
        self.modelo_seniority = joblib.load(
            project_root / "models" / "seniority" / "set_1_XGBClassifier.joblib"
        )

        # Forzar CPU para evitar crash por mismatch CUDA/CPU en inferencia
        self.modelo_salary.set_params(device="cpu", tree_method="hist")
        self.modelo_seniority.set_params(device="cpu", tree_method="hist")

        self.preprocessor_salary.feature_columns = list(
            self.modelo_salary.get_booster().feature_names
        )
        self.preprocessor_seniority.feature_columns = list(
            self.modelo_seniority.get_booster().feature_names
        )

    def predecir(
        self,
        titulo_habilidades: str,
        formacion: str,
        sector: str,
        tipo_empleo: str,
        ciudad: str,
    ) -> dict:

        embedding = self.embedder.embed(titulo_habilidades)

        X_salary = self.preprocessor_salary.procesar(
            embedding, formacion, sector, tipo_empleo, ciudad
        )
        X_seniority = self.preprocessor_seniority.procesar(
            embedding, formacion, sector, tipo_empleo, ciudad
        )

        pred_salary = int(self.modelo_salary.predict(X_salary)[0])
        pred_seniority = int(self.modelo_seniority.predict(X_seniority)[0])

        return {
            "salario_idx": pred_salary,
            "seniority_idx": pred_seniority,
            "salario_label": RANGOS_SALARIO[pred_salary],
            "seniority_label": NIVELES_SENIORITY[pred_seniority],
        }
