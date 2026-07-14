import streamlit as st
from agente.agente import grafo
from agente.rag_pipeline import retriever

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

# Advertencia si no hay documentos PDF
if not retriever:
    st.warning("⚠️ No se encontraron archivos PDF en la carpeta './documentos'. El sistema responderá en base al triaje, pero las búsquedas de políticas detalladas (RAG) se saltarán.")

# Entrada del usuario
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
