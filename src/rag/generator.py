from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag.vectorstore import cargar_vectorstore
from dotenv import load_dotenv

"""
Este módulo se encarga de generar ofertas de trabajo sintéticas utilizando un LLM, con el contexto de ofertas reales recuperadas de un vectorstore.
El proceso es el siguiente:
1. El usuario proporciona un perfil de trabajo (sector, ciudad, tipo de empleo, formación requerida, descripción adicional).
2. Se recuperan las ofertas más similares del vectorstore para dar contexto al LLM.
3. Se construye un prompt con el contexto y el perfil solicitado, y se llama al LLM para generar una oferta de trabajo.
4. Se parsea la respuesta del LLM para extraer el título, habilidades y la oferta completa en un formato estructurado.
"""

load_dotenv()


class GeneradorOfertas:
    def __init__(self):
        self.vectorstore = cargar_vectorstore()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.parser = StrOutputParser()

        self.prompt = ChatPromptTemplate.from_template("""
Eres un experto en recursos humanos del mercado laboral español.
Tu tarea es generar una oferta de trabajo realista basándote en el perfil solicitado.

Estas son ofertas reales similares del mercado como referencia:
{contexto}

Genera una oferta para el siguiente perfil:
- Sector: {sector}
- Ciudad: {ciudad}
- Tipo de empleo: {tipo_empleo}
- Formación requerida: {formacion}
- Descripción adicional: {descripcion}

Responde EXACTAMENTE en este formato, sin añadir nada más:

TITULO: <título del puesto de trabajo>
HABILIDADES: <lista de habilidades técnicas y blandas separadas por comas>
OFERTA:
<texto completo del anuncio de trabajo, entre 100 y 150 palabras>
""")

    def generar(
        self,
        sector: str,
        ciudad: str,
        tipo_empleo: str,
        formacion: str,
        descripcion: str,
    ) -> dict:

        # Recuperar las 5 ofertas más similares del vectorstore
        query = f"{sector} {ciudad} {descripcion}"
        documentos = self.vectorstore.similarity_search(query, k=5)

        # Formatear el contexto con las ofertas recuperadas
        contexto = "\n---\n".join(
            [
                f"Título y habilidades: {doc.page_content}\n"
                f"Sector: {doc.metadata['sector']} | "
                f"Ciudad: {doc.metadata['ciudad']} | "
                f"Seniority: {doc.metadata['seniority']}"
                for doc in documentos
            ]
        )

        # Llamar al LLM
        chain = self.prompt | self.llm | self.parser
        respuesta = chain.invoke(
            {
                "contexto": contexto,
                "sector": sector,
                "ciudad": ciudad,
                "tipo_empleo": tipo_empleo,
                "formacion": formacion,
                "descripcion": descripcion,
            }
        )

        # Parsear la respuesta
        return self._parsear_respuesta(respuesta)

    def _parsear_respuesta(self, respuesta: str) -> dict:
        lineas = respuesta.strip().split("\n")
        titulo = ""
        habilidades = ""
        oferta_lines = []
        en_oferta = False

        for linea in lineas:
            if linea.startswith("TITULO:"):
                titulo = linea.replace("TITULO:", "").strip()
            elif linea.startswith("HABILIDADES:"):
                habilidades = linea.replace("HABILIDADES:", "").strip()
            elif linea.startswith("OFERTA:"):
                en_oferta = True
            elif en_oferta:
                oferta_lines.append(linea)

        return {
            "titulo_habilidades": f"{titulo} {habilidades}",
            "oferta_completa": "\n".join(oferta_lines).strip(),
        }

    def regenerar_oferta(self, titulo_habilidades: str, sector: str, ciudad: str,
                         tipo_empleo: str, formacion: str) -> str:
        """
        Regenera solo el texto completo de la oferta a partir de un titulo_habilidades
        ya ajustado. No hace búsqueda RAG, solo reescribe el anuncio.
        """
        prompt = ChatPromptTemplate.from_template("""
Eres un experto en recursos humanos del mercado laboral español.
Redacta el texto completo de un anuncio de trabajo (entre 100 y 150 palabras) 
basándote en el siguiente título y habilidades:

Titulo y habilidades: {titulo_habilidades}

Datos del puesto:
- Sector: {sector}
- Ciudad: {ciudad}
- Tipo de empleo: {tipo_empleo}
- Formacion requerida: {formacion}

Escribe solo el texto del anuncio, sin encabezados ni etiquetas.
""")
        chain = prompt | self.llm | self.parser
        return chain.invoke({
            "titulo_habilidades": titulo_habilidades,
            "sector": sector,
            "ciudad": ciudad,
            "tipo_empleo": tipo_empleo,
            "formacion": formacion
        })
