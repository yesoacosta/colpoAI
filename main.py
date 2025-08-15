import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Configura tu clave de API de Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

app = FastAPI()

# Configuración para permitir CORS (Intercambio de Recursos de Origen Cruzado)
# Esto es necesario para que tu archivo HTML local pueda comunicarse con este servidor
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AHORA EL ORDEN ES IMPORTANTE: primero definimos la ruta de la API
# Modelo de datos para la solicitud de análisis
class AnalysisRequest(BaseModel):
    image_data: str
    medical_history: str

@app.post("/analyze_colposcopy/")
async def analyze_colposcopy(request: AnalysisRequest):
    # Verificación inicial de la clave de API
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="Error: La clave de API de Gemini no está configurada en el archivo .env.")

    try:
        # Decodifica la imagen de base64
        image_bytes = base64.b64decode(request.image_data)
        
        # Convierte los bytes de la imagen a un formato de imagen que Gemini pueda procesar
        img = Image.open(io.BytesIO(image_bytes))

        # Carga el modelo de Gemini Vision
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # Prepara el prompt para la IA
        prompt = f"""Genera un informe colposcópico técnico y conciso basado en la siguiente imagen. Considera la Unión Escamocolumnar, la Zona de Transformación y el Exocérvix. Integra el historial médico del paciente si es relevante: "{request.medical_history}". El informe debe ser profesional, estructurado en secciones y estar en español. Evita frases introductorias o conversacionales. Utiliza el siguiente formato exacto para las secciones y viñetas:
Observaciones Principales:
- [Observación 1]
- [Observación 2]

Posible Diagnóstico:
- [Diagnóstico 1]

Recomendaciones:
- [Recomendación 1]
- [Recomendación 2]"""

        # Envía la solicitud a la IA
        response = model.generate_content([prompt, img])
        
        # Aquí validamos si la respuesta de la API es válida
        if not hasattr(response, 'text') or not response.text:
            raise Exception("La API de Gemini no devolvió un informe válido. Esto puede ocurrir si la imagen no es adecuada para el análisis.")

        # Extrae el texto del informe de la respuesta
        report_text = response.text

        return {"report": report_text}
    except genai.types.BlockedPromptException as e:
        # Manejo de errores de seguridad
        raise HTTPException(status_code=400, detail=f"Error de la API: El contenido fue bloqueado por una razón de seguridad. Detalle: {e}")
    except Exception as e:
        # Cualquier otro error se manejará aquí de forma clara
        raise HTTPException(status_code=500, detail=f"Ocurrió un error al analizar la imagen. Esto podría ser un problema con la clave de API o con la conexión. Detalle: {str(e)}")

# Y ahora, después, le decimos que sirva los archivos estáticos (HTML, CSS, JS)
app.mount("/", StaticFiles(directory="static", html=True), name="static")