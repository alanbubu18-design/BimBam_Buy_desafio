# 🤖 BimBam Buy - Service Desk Agent

Agente automatizado inteligente para la gestión de solicitudes y soporte, diseñado para responder preguntas frecuentes, clasificar consultas mediante **triaje automático** y escalar casos complejos a soporte humano.

Descripción General
Este proyecto implementa un **agente de Service Desk** para la empresa BimBam Buy.  
El agente combina **IA + RAG (Retrieval Augmented Generation)** para responder preguntas basadas en documentos internos (PDFs) y aplicar un sistema de **triaje inteligente** que clasifica las solicitudes según su urgencia y tipo de acción requerida.

Arquitectura de la Solución
- **Interfaz Web:** Construida con Streamlit para interacción intuitiva.  
- **Pipeline RAG:** Recuperación de información desde PDFs mediante FAISS.  
- **Agente IA:** Implementado con LangChain/LangGraph y modelos Gemini.  
- **Triaje:** Clasificación automática de consultas en tres categorías:
  - `AUTO_RESOLVER` → Preguntas frecuentes resueltas con RAG.  
  - `PEDIR_INFO` → Consultas imprecisas que requieren aclaración.  
  - `ABRIR_TICKET` → Escalamiento a soporte humano.  

Tecnologías y Herramientas
- **Python 3.13+**
- **Streamlit** (interfaz web)
- **LangChain / LangGraph** (orquestación de agentes)
- **FAISS** (vector store para RAG)
- **Google Generative AI (Gemini)** (modelo de lenguaje)
- **dotenv** (manejo de variables de entorno)

Instrucciones de Ejecución

Opción 1: Usar la versión en la nube
Accede directamente sin instalar nada:  
👉 [Abrir aplicación en Streamlit Cloud](https://bimbambuydesafio-cweagv4gp3nnxpjffef3st.streamlit.app/)

Opción 2: Ejecutar localmente
1. Clona el repositorio:
   ```bash
   git clone https://github.com/alanbubu18-design/bimbambuy-agent.git
   cd bimbambuy-agent
2. Instala dependencias: pip install -r requirements.txt
3. Crea un archivo .env en la raíz del proyecto con tu API key: GOOGLE_API_KEY=tu_api_key_aqui
4. Asegurate de tener descargada la carpeta de documentos y sus respectivos PDF's
5. Ejecuta la aplicación:streamlit run app.py

## 🌐 Evidencia de Deploy en la Nube

Durante el proceso de despliegue se intentó utilizar **Oracle Cloud Infrastructure (OCI)**; sin embargo, la plataforma presentó limitaciones de capacidad en la región asignada, lo que impidió completar el deploy de manera estable.  

Ante esta situación, se evaluaron alternativas y se seleccionó **Render** como solución complementaria. Render ofrece un **plan gratuito permanente**, soporte para aplicaciones Python y despliegue sencillo de proyectos con **Streamlit**, lo que permitió resolver el problema y garantizar la disponibilidad del agente en la nube.  

**URL pública del servicio:**  
[https://bimbam-buy-desafio.onrender.com](https://bimbam-buy-desafio.onrender.com)

**Motivos de elección de Render:**  
- Sustituye la función de OCI al brindar un entorno confiable para ejecutar aplicaciones en la nube.  
- Permite visualizar **logs en tiempo real**, facilitando la depuración y validación del despliegue.  
- Genera una **URL pública inmediata**, requisito esencial para la evidencia del proyecto.  
- Compatible con **Streamlit**, que es el framework utilizado para la interfaz del agente.  
- Plan gratuito sin límite de tiempo, ideal para proyectos académicos y demostrativos.  

**Framework utilizado:** Streamlit  
---

## 📸 Evidencia
Se adjunta captura de pantalla del servicio en ejecución en Render, mostrando la URL pública y los logs de despliegue exitoso, que se anexa en nuestra carpeta de evidencias.

