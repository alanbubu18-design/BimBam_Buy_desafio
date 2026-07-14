import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Error: No se encontró GEMINI_API_KEY en el archivo .env")
