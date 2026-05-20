import sys
import os

sys.path.append(".")

import streamlit as st

FAISS_PATH = "vectorstore/ofertas_index/index.faiss"
PKL_PATH = "vectorstore/ofertas_index/index.pkl"


def _mostrar_descarga_vectorstore():
    """Muestra animación de descarga si el vectorstore no está en disco."""
    if os.path.exists(FAISS_PATH) and os.path.exists(PKL_PATH):
        return  # ya está, no hace falta nada

    from download_vectorstore import descargar_vectorstore, VECTORSTORE_DIR

    st.info(
        "⏳ **Primera ejecución detectada.** El índice de búsqueda (~220 MB) "
        "no está en disco y se descargará ahora desde Google Drive. "
        "Esto solo ocurre una vez.",
        icon="📥",
    )

    with st.status("Descargando base de conocimiento...", expanded=True) as status:
        st.write("📦 Descargando `index.faiss` (~212 MB)...")
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)

        import gdown

        gdown.download(
            f"https://drive.google.com/uc?id={__import__('download_vectorstore').FAISS_FILE_ID}",
            FAISS_PATH,
            quiet=False,
        )
        st.write("✅ `index.faiss` descargado.")

        st.write("📦 Descargando `index.pkl` (~10 MB)...")
        gdown.download(
            f"https://drive.google.com/uc?id={__import__('download_vectorstore').PKL_FILE_ID}",
            PKL_PATH,
            quiet=False,
        )
        st.write("✅ `index.pkl` descargado.")

        status.update(
            label="✅ Base de conocimiento lista. Recargando...", state="complete"
        )

    st.rerun()


@st.cache_resource
def cargar_recursos():
    from src.rag.generator import GeneradorOfertas
    from src.predictor.predictor import Predictor

    generador = GeneradorOfertas()
    predictor = Predictor(device="cpu")
    return generador, predictor


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
SECTORES = [
    "Agricultura",
    "Bienes de consumo",
    "Bienes raíces",
    "Comercio electrónico",
    "Comercio minorista y comercio",
    "Construcción",
    "Deportes y recreación",
    "Economía y política",
    "Energía y medio ambiente",
    "Finanzas y seguros",
    "Internet",
    "Medios de comunicación",
    "Metales y electrónica",
    "Productos químicos y recursos",
    "Publicidad y Marketing",
    "Salud, Farmacia y Tecnología Médica",
    "Servicios",
    "Sociedad",
    "Tecnología y telecomunicaciones",
    "Transporte y Logística",
    "Viajes, turismo y hostelería",
]
FORMACIONES = [
    "Grado Universitario",
    "Postgrado",
    "FP Superior",
    "FP Medio",
    "ESO",
    "Ninguna",
]
TIPOS_EMPLEO = [
    "Jornada completa",
    "Media jornada",
    "Prácticas",
    "Temporal",
    "Contrato por obra",
    "Otro",
]
CIUDADES = [
    "A Coruña",
    "Albacete",
    "Alicante",
    "Almería",
    "Badajoz",
    "Barcelona",
    "Bilbao",
    "Burgos",
    "Castellón de la Plana",
    "Ciudad Real",
    "Cuenca",
    "Cáceres",
    "Cádiz",
    "Córdoba",
    "España",
    "Gijón",
    "Girona",
    "Granada",
    "Guadalajara",
    "Huelva",
    "Huesca",
    "Jaén",
    "Jerez de la Frontera",
    "Las Palmas de Gran Canaria",
    "León",
    "Lleida",
    "Logroño",
    "Lugo",
    "Madrid",
    "Murcia",
    "Málaga",
    "Ourense",
    "Oviedo",
    "Palencia",
    "Palma de Mallorca",
    "Pamplona",
    "Pontevedra",
    "Salamanca",
    "San Sebastián",
    "Santa Cruz de Tenerife",
    "Santander",
    "Segovia",
    "Sevilla",
    "Soria",
    "Tarragona",
    "Teruel",
    "Toledo",
    "Valencia",
    "Valladolid",
    "Vigo",
    "Vitoria",
    "Zamora",
    "Zaragoza",
    "Ávila",
]

st.set_page_config(page_title="Generador de Ofertas", layout="wide")
st.title("Generador de Ofertas de Trabajo")
st.caption("Genera una oferta realista y predice su salario y nivel de seniority")

# Descarga el vectorstore si no está en disco (solo ocurre la primera vez)
_mostrar_descarga_vectorstore()

# ── BLOQUE 1: INPUTS ──────────────────────────────────────────────────────────
st.subheader("📋 Perfil de la oferta")

col1, col2, col3, col4 = st.columns(4)
with col1:
    formacion = st.selectbox("Formación requerida", FORMACIONES)
with col2:
    sector = st.selectbox("Sector", SECTORES)
with col3:
    tipo_empleo = st.selectbox("Tipo de empleo", TIPOS_EMPLEO)
with col4:
    ciudad = st.selectbox("Ciudad", CIUDADES)

descripcion = st.text_area(
    "Descripción del perfil buscado",
    placeholder="Ej: Necesito un perfil de backend con experiencia en APIs REST y bases de datos...",
    height=100,
)

generar_clicked = st.button("Generar oferta", type="primary", use_container_width=True)

if generar_clicked:
    if not descripcion.strip():
        st.warning("Por favor escribe una descripción del perfil.")
    else:
        generador, predictor = cargar_recursos()
        with st.status("Procesando...", expanded=True) as status:
            st.write("🔍 Buscando ofertas similares y generando con IA...")
            resultado_rag = generador.generar(
                sector=sector,
                ciudad=ciudad,
                tipo_empleo=tipo_empleo,
                formacion=formacion,
                descripcion=descripcion,
            )
            st.write("✅ Oferta generada")
            st.write("Calculando salario y seniority...")
            resultado_pred = predictor.predecir(
                titulo_habilidades=resultado_rag["titulo_habilidades"],
                formacion=formacion,
                sector=sector,
                tipo_empleo=tipo_empleo,
                ciudad=ciudad,
            )
            st.write("✅ Predicciones completadas")
            status.update(label="¡Listo!", state="complete")

        st.session_state.resultado = {
            **resultado_rag,
            "prediccion_salario": resultado_pred["salario_idx"],
            "prediccion_seniority": resultado_pred["seniority_idx"],
            "formacion": formacion,
            "sector": sector,
            "tipo_empleo": tipo_empleo,
            "ciudad": ciudad,
            "descripcion": descripcion,
        }

# ── BLOQUE 2: OFERTA + PREDICCIONES ───────────────────────────────────────────
if "resultado" in st.session_state:
    r = st.session_state.resultado
    salario_idx = r["prediccion_salario"]
    seniority_idx = r["prediccion_seniority"]

    st.divider()
    st.subheader("📄 Oferta generada")
    st.markdown(r["oferta_completa"])
    with st.expander("Ver título y habilidades extraídas"):
        st.text(r["titulo_habilidades"])

    st.divider()
    st.subheader("Predicciones")

    col_sal, col_sen = st.columns(2)

    with col_sal:
        st.markdown("**Rango salarial predicho**")
        st.markdown(
            f"<h2 style='color:#4CAF50;margin:0'>{RANGOS_SALARIO[salario_idx]}</h2>",
            unsafe_allow_html=True,
        )
        porcentaje = int(((salario_idx + 1) / len(RANGOS_SALARIO)) * 100)
        st.markdown(
            f"""
        <div style='background:#333;border-radius:8px;height:20px;margin-top:10px'>
            <div style='background:#4CAF50;width:{porcentaje}%;height:20px;border-radius:8px'></div>
        </div>
        <p style='color:#aaa;font-size:12px;margin-top:4px'>Rango {salario_idx+1} de {len(RANGOS_SALARIO)}</p>
        """,
            unsafe_allow_html=True,
        )

    with col_sen:
        st.markdown("**Nivel de seniority**")
        colores = []
        for i in range(3):
            if i == seniority_idx:
                colores.append(
                    f"<div style='background:#4CAF50;color:white;text-align:center;padding:12px;border-radius:8px;font-weight:bold;font-size:16px'>{NIVELES_SENIORITY[i]}</div>"
                )
            else:
                colores.append(
                    f"<div style='background:#333;color:#aaa;text-align:center;padding:12px;border-radius:8px;font-size:16px'>{NIVELES_SENIORITY[i]}</div>"
                )
        st.markdown(
            f"""
        <div style='display:flex;gap:10px;margin-top:8px'>
            {colores[0]}{colores[1]}{colores[2]}
        </div>
        """,
            unsafe_allow_html=True,
        )

    # ── BLOQUE 3: AJUSTE ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔧 Ajustar predicción")
    st.caption(
        "Si el salario o seniority predicho no se ajusta a lo que buscas, selecciona el objetivo y pulsa ajustar."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        target_salario_label = st.selectbox(
            "Salario objetivo", ["Sin preferencia"] + RANGOS_SALARIO, index=0
        )
    with col_b:
        target_seniority_label = st.selectbox(
            "Seniority objetivo", ["Sin preferencia"] + NIVELES_SENIORITY, index=0
        )

    ajustar_clicked = st.button(
        "🔄 Ajustar oferta", type="secondary", use_container_width=True
    )

    if ajustar_clicked:
        target_salario = (
            None
            if target_salario_label == "Sin preferencia"
            else RANGOS_SALARIO.index(target_salario_label)
        )
        target_seniority = (
            None
            if target_seniority_label == "Sin preferencia"
            else NIVELES_SENIORITY.index(target_seniority_label)
        )

        if target_salario is None and target_seniority is None:
            st.warning("Selecciona al menos un objetivo de salario o seniority.")
        else:
            from src.graph.graph import construir_grafo

            grafo = construir_grafo()

            with st.status("Ajustando oferta...", expanded=True) as status:
                st.write("🔍 Analizando la oferta actual y calculando ajustes...")
                resultado_ajustado = grafo.invoke(
                    {
                        "formacion": r.get("formacion", formacion),
                        "sector": r.get("sector", sector),
                        "tipo_empleo": r.get("tipo_empleo", tipo_empleo),
                        "ciudad": r.get("ciudad", ciudad),
                        "descripcion": r.get("descripcion", descripcion),
                        "titulo_habilidades": r["titulo_habilidades"],
                        "oferta_completa": r["oferta_completa"],
                        "prediccion_salario": r["prediccion_salario"],
                        "prediccion_seniority": r["prediccion_seniority"],
                        "target_salario": target_salario,
                        "target_seniority": target_seniority,
                        "max_iteraciones": 4,
                        "iteracion": 0,
                        "historial": [],
                    }
                )

                iteraciones = resultado_ajustado["iteracion"]
                sal_final = RANGOS_SALARIO[resultado_ajustado["prediccion_salario"]]
                sen_final = NIVELES_SENIORITY[
                    resultado_ajustado["prediccion_seniority"]
                ]

                if iteraciones > 0:
                    st.write("Regenerando texto de la oferta con los ajustes...")
                    generador, _ = cargar_recursos()
                    nueva_oferta = generador.regenerar_oferta(
                        titulo_habilidades=resultado_ajustado["titulo_habilidades"],
                        sector=r.get("sector", sector),
                        ciudad=r.get("ciudad", ciudad),
                        tipo_empleo=r.get("tipo_empleo", tipo_empleo),
                        formacion=r.get("formacion", formacion),
                    )
                    resultado_ajustado["oferta_completa"] = nueva_oferta

                if (
                    target_salario is None
                    or resultado_ajustado["prediccion_salario"] == target_salario
                ) and (
                    target_seniority is None
                    or resultado_ajustado["prediccion_seniority"] == target_seniority
                ):
                    status.update(
                        label=f"✅ Objetivo alcanzado en {iteraciones} iteración/es",
                        state="complete",
                    )
                else:
                    status.update(
                        label=f"⚠️ Mejor resultado tras {iteraciones} iteración/es: {sal_final} / {sen_final}",
                        state="complete",
                    )

            st.session_state.resultado = {
                **resultado_ajustado,
                "formacion": r.get("formacion", formacion),
                "sector": r.get("sector", sector),
                "tipo_empleo": r.get("tipo_empleo", tipo_empleo),
                "ciudad": r.get("ciudad", ciudad),
                "descripcion": r.get("descripcion", descripcion),
            }
            st.rerun()
