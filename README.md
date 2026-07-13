# 🤖 BimBam Buy - Service Desk Agent

Agente automatizado inteligente para la gestión de solicitudes y soporte, diseñado para responder preguntas frecuentes, clasificar consultas mediante **triaje automático** y escalar casos complejos a soporte humano.

---

## 📖 Descripción General
Este proyecto implementa un **agente de Service Desk** para la empresa BimBam Buy.  
El agente combina **IA + RAG (Retrieval Augmented Generation)** para responder preguntas basadas en documentos internos (PDFs) y aplicar un sistema de **triaje inteligente** que clasifica las solicitudes según su urgencia y tipo de acción requerida.

---

## 🏗️ Arquitectura de la Solución
- **Interfaz Web:** Construida con Streamlit para interacción intuitiva.  
- **Pipeline RAG:** Recuperación de información desde PDFs mediante FAISS.  
- **Agente IA:** Implementado con LangChain/LangGraph y modelos Gemini.  
- **Triaje:** Clasificación automática de consultas en tres categorías:
  - `AUTO_RESOLVER` → Preguntas frecuentes resueltas con RAG.  
  - `PEDIR_INFO` → Consultas imprecisas que requieren aclaración.  
  - `ABRIR_TICKET` → Escalamiento a soporte humano.  

---

## 🛠️ Tecnologías y Herramientas
- **Python 3.10+**
- **Streamlit** (interfaz web)
- **LangChain / LangGraph** (orquestación de agentes)
- **FAISS** (vector store para RAG)
- **Google Generative AI (Gemini)** (modelo de lenguaje)
- **dotenv** (manejo de variables de entorno)

---

## 🚀 Instrucciones de Ejecución

### 🔹 Opción 1: Usar la versión en la nube
Accede directamente sin instalar nada:  
👉 [Abrir aplicación en Streamlit Cloud](https://bimbambuydesafio-cweagv4gp3nnxpjffef3st.streamlit.app/)

### 🔹 Opción 2: Ejecutar localmente
1. Clona el repositorio:
   ```bash
   git clone https://github.com/alanbubu18-design/bimbambuy-agent.git
   cd bimbambuy-agent
