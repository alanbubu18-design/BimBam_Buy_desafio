from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from config.settings import GEMINI_API_KEY

def inicializar_retriever():
    docs = []
    ruta_docs = Path("./documentos")
    if not ruta_docs.exists() or not list(ruta_docs.glob("*.pdf")):
        return None

    for documento in ruta_docs.glob("*.pdf"):
        loader = PyMuPDFLoader(str(documento))
        docs.extend(loader.load())

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

prompt_rag = ChatPromptTemplate([
    ("system", "Eres el especialista en soporte de BimBam Buy. Usa solo el contexto proporcionado."),
    ("human", "Contexto: {context}\nPregunta del usuario: {input}")
])

document_chain = create_stuff_documents_chain(
    ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY),
    prompt_rag
)

def busqueda_de_respuestas_RAG(pregunta):
    if not retriever:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}
    documentos_relacionados = retriever.invoke(pregunta)
    if not documentos_relacionados:
        return {"respuesta": "No lo sé", "citaciones": [], "documentos_encontrados": False}
    answer = document_chain.invoke({"input": pregunta, "context": documentos_relacionados})
    return {
        "respuesta": answer,
        "citaciones": documentos_relacionados,
        "documentos_encontrados": True
    }
