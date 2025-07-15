import os
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from base64 import b64encode

# Importar HumanMessage para manejar el contenido multimodal correctamente
from langchain_core.messages import HumanMessage

# Importar el agente y sus componentes
from agent import (
    agent_executor,
    add_to_history,
    get_context,
    test_connection,
    llm
)

# Definir el modelo de datos para la entrada de la API
class ChatRequest(BaseModel):
    query: str

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Agente Conversacional LangChain (Backend API)",
    description="Backend API para interactuar con el agente LangChain multi-fuente, potenciado por Google Gemini.",
    version="1.0.0",
)

# --- CONFIGURACIÓN CORS ---
# Añade explícitamente tu origen de desarrollo (localhost)
# Para producción, deberías reemplazar "*" con la URL de tu frontend desplegado
origins = [
    "http://localhost:3000", # Si usas python -m http.server 3000
    "http://127.0.0.1:3000", # Otra forma de localhost
    "http://localhost:5500", # Si usas Live Server u otro en 5500
    "http://127.0.0.1:5500", # Tu origen actual que está dando problemas
    "*" # Mantener el comodín para otras pruebas, pero sé específico en producción
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # <--- CAMBIO CLAVE AQUÍ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Evento de inicio: probar la conexión con Gemini
@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando backend API...")
    if not test_connection():
        print("❌ La conexión con Gemini falló al iniciar. El backend podría no funcionar correctamente.")
    else:
        print("✅ Backend API iniciado y conectado a Gemini.")

# Ruta principal para verificar que el backend está funcionando (ahora solo una API)
@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Backend API del Agente Conversacional LangChain. Usa /api/chat o /api/upload-cedula para interactuar."}

# Ruta para interactuar con el agente (conversación)
@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    user_query = request.query
    
    try:
        context = get_context()
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            agent_executor.invoke,
            {"input": user_query, "conversation_context": context}
        )
        
        agent_response = response["output"]
        add_to_history(user_query, agent_response)
        
        return {"response": agent_response}
    except Exception as e:
        print(f"❌ Error al procesar la petición de chat: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# --- RUTA PARA SUBIR Y LEER CÉDULAS ---
@app.post("/api/upload-cedula")
async def upload_cedula(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    try:
        image_bytes = await file.read()
        image_base64 = b64encode(image_bytes).decode("utf-8")

        # Crear el contenido multimodal para Gemini
        prompt_content = [
            {
                "type": "text",
                "text": "Extrae la siguiente información de esta cédula de identidad, si está presente: Nombre completo, Número de Identificación (Cédula), Fecha de Nacimiento, Fecha de Expedición, Fecha de Vencimiento, Género, Nacionalidad. Presenta la información de forma clara y estructurada, por ejemplo, en una lista. Si algún dato no se encuentra, indícalo."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{file.content_type};base64,{image_base64}"
                }
            }
        ]
        
        # Invocar al LLM directamente para procesar la imagen
        loop = asyncio.get_running_loop()
        gemini_response = await loop.run_in_executor(
            None,
            llm.invoke,
            [HumanMessage(content=prompt_content)]
        )
        
        extracted_data = gemini_response.content
        
        return {"message": "Cédula procesada exitosamente", "extracted_data": extracted_data}

    except Exception as e:
        print(f"❌ Error al procesar la imagen de la cédula: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

# Para ejecutar la aplicación directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
