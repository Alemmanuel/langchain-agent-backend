from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Importar el agente y sus componentes desde el archivo modificado
# Asegúrate de que 'agent_gemini_current.py' esté en el mismo directorio
from agent import (
    agent_executor,
    add_to_history,
    get_context,
    test_connection,
)

# Definir el modelo de datos para la entrada de la API
class ChatRequest(BaseModel):
    query: str

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Agente Conversacional LangChain (Gemini Backend)",
    description="Backend para interactuar con el agente LangChain multi-fuente, potenciado por Google Gemini.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las orígenes. En producción, deberías restringirlo a tu dominio de frontend.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# Evento de inicio: probar la conexión con Gemini
@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando backend...")
    if not test_connection():
        print("❌ La conexión con Gemini falló al iniciar. El backend podría no funcionar correctamente.")
        # Opcional: podrías detener la aplicación si la conexión es crítica
        # raise RuntimeError("No se pudo conectar con Gemini. Verifica tu API key y cuota.")
    else:
        print("✅ Backend iniciado y conectado a Gemini.")

# Ruta principal para verificar que el backend está funcionando
@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Agente Conversacional LangChain. Usa /chat para interactuar."}

# Ruta para interactuar con el agente
@app.post("/chat")
async def chat_with_agent(request: ChatRequest):
    user_query = request.query
    
    try:
        # Obtener el contexto de conversación actual
        context = get_context()
        
        # Invocar al agente con la pregunta del usuario y el contexto
        # LangChain AgentExecutor.invoke es síncrono, así que lo ejecutamos en un executor
        # para no bloquear el bucle de eventos de FastAPI.
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, # Usa el ThreadPoolExecutor por defecto
            agent_executor.invoke,
            {"input": user_query, "conversation_context": context}
        )
        
        agent_response = response["output"]
        
        # Añadir el intercambio al historial de conversación
        add_to_history(user_query, agent_response)
        
        return {"response": agent_response}
    except Exception as e:
        print(f"❌ Error al procesar la petición: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

# Para ejecutar la aplicación directamente (opcional, uvicorn se usa más comúnmente)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
