from typing import TypedDict, Optional
from langgraph.graph import START, END, StateGraph
from agente.triaje import triaje
from agente.rag_pipeline import busqueda_de_respuestas_RAG

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
    return {"respuesta": "Necesito más información.", "citaciones": [], "accion_final": "PEDIR_INFO"}

def nodo_abrir_ticket(state: AgentState) -> AgentState:
    tri = state["triaje"]
    return {"respuesta": f"Tu solicitud requiere atención humana. Ticket con prioridad {tri.get('urgencia','MEDIA')}.",
            "citaciones": [], "accion_final": "ABRIR_TICKET"}

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
