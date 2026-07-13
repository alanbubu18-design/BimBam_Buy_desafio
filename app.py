import os
from pathlib import Path
from typing import TypedDict, Optional, Literal, List, Dict
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Cargar variables de entorno del archivo .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validar API Key antes de iniciar
if not GEMINI_API_KEY:
    st.error("Error: No se encontró GEMINI_API_KEY en el archivo .env")
    st.stop()

# Importaciones de LangChain, LangGraph y Google
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import START, END, StateGraph

# ==========================================
# 1. CONFIGURACIÓN DEL LLM
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

PROMPT_TRIAJE = """
Eres un especialista en triaje del Service Desk de la empresa BimBam Buy,
un e-commerce multiplataforma enfocado en la experiencia de compra digital ágil y segura.

Dado el mensaje del usuario, devuelve SÓLO un JSON con:
{
 "decision": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_TICKET",
 "urgencia": "BAJA" | "MEDIANA" | "ALTA",
 "campos_faltantes": ["..."]
}

Reglas:
- **AUTO_RESOLVER**: Preguntas claras sobre las políticas internas de BimBam Buy
  (Ej.: "¿Cuál es el tiempo estimado de entrega?", "¿Qué cubre la garantía de productos?",
  "¿Cómo funciona el programa de afiliados?", "¿Qué métodos de pago aceptan?").
- **PEDIR_INFO**: Mensajes imprecisos o sin información suficiente para identificar el tema
  (Ej.: "Necesito ayuda con una política", "Tengo una duda general", "No entiendo cómo funciona").
- **ABRIR_TICKET**: Solicitudes de excepciones, autorizaciones, aprobaciones o accesos especiales,
  o cuando el usuario pide explícitamente abrir un ticket
  (Ej.: "Quiero una excepción en el reembolso", "Solicito autorización para un pago especial",
  "Por favor, abre un ticket con soporte de logística").
Analiza el mensaje y decide la acción más adecuada.
"""

class TriajeOut(BaseModel):
    decision: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_TICKET"]
    urgencia: Literal["BAJA", "MEDIANA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)

chain_de_triaje = llm.with_structured_output(TriajeOut)

def triaje(mensaje: str) -> Dict:
    salida: TriajeOut = chain_de_triaje.invoke(
        [
            SystemMessage(content=PROMPT_TRIAJE),
            HumanMessage(content=mensaje)
        ]
    )
    return salida.model_dump()

# ==========================================
# 2. PROCESAMIENTO DE DOCUMENTOS (RAG)
# ==========================================
@st.cache_resource
def inicializar_retriever():
    """Carga documentos locales y crea el almacén de vectores FAISS."""
    docs = []
    # Ruta adaptada a entorno local (carpeta "documentos" en el proyecto)
    ruta_docs = Path("./documentos")
    
    if not ruta_docs.exists() or not list(ruta_docs.glob("*.pdf")):
        return None

    for documento in ruta_docs.glob("*.pdf"):
        try:
            loader = PyMuPDFLoader(str(documento))
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error cargando archivo {documento.name}: {e}")

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_splits = splitter.split_documents(docs)

    modelo_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )

    vectorstore = FAISS.from_documents(docs_splits, modelo_embeddings)
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 4}
    )

retriever = inicializar_retriever()

# ==========================================
# 3. CADENA DE DOCUMENTOS
# ==========================================
prompt_rag = ChatPromptTemplate(
    [
        ("system",
            """Eres el especialista en soporte de la empresa BimBam Buy,
            un e-commerce multiplataforma enfocado en la experiencia de compra digital ágil y segura.

            Responde siempre utilizando únicamente la información contenida en el contexto que se te proporcione.

            Si la respuesta está en el contexto, explícalo de forma clara y breve.
            Si la información no aparece en el contexto, responde exactamente: 'No lo sé'.
            No inventes ni supongas información fuera del contexto.
            """
        ),
        ("human", "Contexto: {context}\nPregunta del usuario: {input}")
    ]
)

document_chain = create_stuff_documents_chain(llm, prompt_rag)

def busqueda_de_respuestas_RAG(pregunta) -> Dict:
    if not retriever:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}
        
    documentos_relacionados = retriever.invoke(pregunta)

    if not documentos_relacionados:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}
        
    answer = document_chain.invoke({
        "input": pregunta,
        "context": documentos_relacionados
    })
    
    if answer.rstrip(".!?") == "No lo sé":
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}
        
    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True
    }

# ==========================================
# 4. CONFIGURACIÓN DEL AGENTE (LANGGRAPH)
# ==========================================
class AgentState(TypedDict, total=False):
    pregunta: str
    triaje: dict
    respuesta: Optional[str]
    citaciones: Optional[list]
    rag_exito: bool
    accion_final: str

def nodo_triaje(state: AgentState) -> AgentState:
    return {"triaje": triaje(state["pregunta"])}

def nodo_auto_resolver(state: AgentState) -> AgentState:
    respuesta_RAG = busqueda_de_respuestas_RAG(state["pregunta"])
    update: AgentState = {
        "respuesta": respuesta_RAG["respuesta"],
        "citaciones": respuesta_RAG["citaciones"],
        "rag_exito": respuesta_RAG["documentos_encontrados"]
    }
    if respuesta_RAG["documentos_encontrados"]:
        update["accion_final"] = "AUTO_RESOLVER"
    return update

def nodo_pedir_info(state: AgentState) -> AgentState:
    return {
        "respuesta": "Necesito más información sobre tu solicitud para poder ayudarte adecuadamente.",
        "citaciones": [],
        "accion_final": "PEDIR_INFO"
    }

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    tri = state["triaje"]
    return {
        "respuesta": f"Se ha determinado que tu solicitud requiere atención humana. Abriendo ticket en soporte con prioridad {tri.get('urgencia', 'MEDIA')}.",
        "citaciones": [],
        "accion_final": "ABRIR_TICKET"
    }

def arista_decision_triaje(state: AgentState) -> str:
    tri = state["triaje"]
    if tri["decision"] == "AUTO_RESOLVER":
        return "rag"
    elif tri["decision"] == "PEDIR_INFO":
        return "info"
    else:
        return "ticket"

def arista_auto_rag(state: AgentState) -> str:
    if state.get("rag_exito"):
        return "ok"
    KEYWORDS_ABRIR_TICKET = ["aprobación", "aprobar", "excepción", "liberación", "autorizacion", "autorizar", "abrir ticket", "acceso especial"]
    if any(keyword in state["pregunta"].lower() for keyword in KEYWORDS_ABRIR_TICKET):
        return "ticket"
    return "info"

workflow = StateGraph(AgentState)
workflow.add_node("triaje", nodo_triaje)
workflow.add_node("auto_resolver", nodo_auto_resolver)
workflow.add_node("pedir_info", nodo_pedir_info)
workflow.add_node("abrir_ticket", nodo_abrir_ticket)

workflow.add_edge(START, "triaje")
workflow.add_conditional_edges("triaje", arista_decision_triaje, {
    "rag": "auto_resolver",
    "info": "pedir_info",
    "ticket": "abrir_ticket"
})
workflow.add_conditional_edges("auto_resolver", arista_auto_rag, {
    "info": "pedir_info",
    "ticket": "abrir_ticket",
    "ok": END
})
workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_ticket", END)

grafo = workflow.compile()

# ==========================================
# 5. INTERFAZ EN LA NUBE CON STREAMLIT
# ==========================================
st.set_page_config(page_title="BimBam Buy - Service Desk Agent", page_icon="🤖")

# Encabezado estilizado
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🤖 BimBam Buy - Service Desk Agent</h1>
    <p style='text-align: center; font-size:18px;'>Agente automatizado inteligente para la gestión de solicitudes y soporte</p>
    """,
    unsafe_allow_html=True
)

# Áreas de conocimiento disponibles
st.markdown("### 📚 Áreas de Conocimiento Disponibles")
st.info("""
- Garantías y políticas de devolución  
- Tiempos de entrega y logística  
- Métodos de pago aceptados  
- Atención al cliente y soporte técnico  
""")

# Ejemplos de preguntas
with st.expander("💡 Ejemplos de Preguntas que puedes hacer"):
    st.markdown("""
    - ¿Qué métodos de pago aceptan?  
    - ¿Cuál es el tiempo estimado de entrega?  
    - ¿Cómo funciona la garantía de productos electrónicos?  
    - ¿Qué debo hacer para solicitar una devolución?  
    """)

# Explicación del triaje
with st.expander("🧭 ¿Qué es el Triaje del Agente?"):
    st.markdown("""
    El **triaje** es el proceso inicial que realiza el agente para analizar tu consulta y decidir cómo atenderla.  
    Su función es **clasificar la solicitud** según su urgencia y tipo de acción requerida.

    **¿Para qué fue diseñado?**
    - Priorizar solicitudes urgentes (ejemplo: problemas de acceso o fallas críticas).
    - Diferenciar entre consultas simples y aquellas que requieren búsqueda en documentos (RAG).
    - Guiar al agente hacia la acción más adecuada.

    **¿Qué haría en cada caso?**
    - 🔴 **Alta urgencia:** Escalar la solicitud o dar respuesta inmediata con pasos críticos.  
    - 🟡 **Media urgencia:** Responder con detalle, apoyándose en políticas y documentos.  
    - 🟢 **Baja urgencia:** Proporcionar información general o guiar al usuario hacia recursos disponibles.  
    """)
    

if not retriever:
    st.warning("⚠️ No se encontraron archivos PDF en la carpeta './documentos'. El sistema responderá en base al triaje, pero las búsquedas de políticas detalladas (RAG) se saltarán.")

pregunta_usuario = st.text_input("Ingresa tu consulta:", placeholder="¿Qué métodos de pago aceptan?")

if st.button("Enviar Consulta", type="primary"):
    if pregunta_usuario.strip():
        with st.spinner("El agente está procesando tu solicitud..."):
            # Invocar al grafo de LangGraph
            resultado = grafo.invoke({"pregunta": pregunta_usuario})
            
            # Mostrar Resultados en la Interfaz Web
            st.subheader("Respuesta del Agente:")
            st.info(resultado["respuesta"])
            
            # Datos técnicos del proceso en un contenedor colapsable
            with st.expander("Ver Detalles de la Operación (Metadatos)"):
                st.write(f"**Decisión Inicial de Triaje:** {resultado['triaje']['decision']}")
                st.write(f"**Urgencia Estimada:** {resultado['triaje']['urgencia']}")
                st.write(f"**Acción Final Tomada:** {resultado['accion_final']}")
                
                if resultado.get("citaciones"):
                    st.write("**Fuentes utilizadas (RAG):**")
                    for idx, citacion in enumerate(resultado["citaciones"]):
                        st.markdown(f"**Doc {idx+1}:** {citacion.page_content[:200]}...")
    else:
        st.error("Por favor, ingresa una pregunta válida.")
