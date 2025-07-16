import os
import uvicorn
import asyncio
import re # Importar para expresiones regulares
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from base64 import b64encode

# Importar HumanMessage para manejar el contenido multimodal correctamente
from langchain_core.messages import HumanMessage

# Importar el agente y sus componentes, incluyendo el nuevo token_callback_handler
from agent import (
    agent_executor,
    add_to_history,
    get_context,
    test_connection,
    llm,
    token_callback_handler # ¡Importar el handler!
)

# Definir el modelo de datos para la entrada de la API
class ChatRequest(BaseModel):
    query: str

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Agente Conversacional LangChain (Backend)",
    description="Backend para interactuar con el agente LangChain multi-fuente, potenciado por Google Gemini.",
    version="1.0.0",
)

# --- CONFIGURACIÓN CORS ---
origins = [
    "http://localhost:3000", # Si usas un servidor de desarrollo como Vite/Next.js
    "http://127.0.0.1:3000",
    "http://localhost:5500", # Si usas el servidor simple de Python
    "http://127.0.0.1:5500",
    "*" # Mantener el comodín para máxima compatibilidad si no se sabe el origen exacto
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Evento de inicio: probar la conexión con Gemini
@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando backend...")
    if not test_connection():
        print("❌ La conexión con Gemini falló al iniciar. El backend podría no funcionar correctamente.")
    else:
        print("✅ Backend iniciado y conectado a Gemini.")

# Ruta principal para verificar que el backend está funcionando
@app.get("/")
async def read_root():
    return {"message": "Bienvenido al Agente Conversacional LangChain. Usa /api/chat para interactuar y /api/upload-cedula para procesar documentos."}

# --- Funciones de Validación ---

# Lista de palabras clave o patrones prohibidos para operaciones de base de datos
FORBIDDEN_KEYWORDS = [
    r'\bdelete\b', r'\bdrop\b', r'\bupdate\b', r'\binsert\b', r'\balter\b',
    r'\btruncate\b', r'\bgrant\b', r'\brevoke\b', r'\bcreate\b', r'\b(sql|database)\s+injection\b'
]

def validate_user_query(query: str) -> str | None:
    """
    Valida la consulta del usuario en busca de operaciones prohibidas o temas fuera de alcance.
    Retorna un mensaje de error si la consulta es inválida, de lo contrario, None.
    """
    query_lower = query.lower()

    # 1. Validación de operaciones prohibidas (SQL Injection, etc.)
    for keyword_pattern in FORBIDDEN_KEYWORDS:
        if re.search(keyword_pattern, query_lower):
            return "¡ALERTA DE SEGURIDAD! Detecté un intento de operación no autorizada (como DELETE, UPDATE, etc.). Mi función es solo responder preguntas, no modificar datos. Por favor, reformula tu consulta."

    # 2. Validación de temas fuera de alcance (simplificado)
    # Esta es una validación básica. El agente ya tiene herramientas específicas.
    # Si la pregunta es muy genérica y no se relaciona con las herramientas,
    # el agente usará 'responder_pregunta_general'.
    # Aquí podríamos añadir una capa más estricta si fuera necesario,
    # pero para evitar falsos positivos, nos centraremos en lo malicioso.
    
    return None # La consulta es válida

async def validate_llm_response(response_text: str) -> str | None:
    """
    Valida la respuesta del LLM para asegurar que no sea "loquita" (incoherente, irrelevante, dañina).
    Retorna un mensaje de error si la respuesta es inválida, de lo contrario, None.
    """
    validation_prompt = f"""Evalúa la siguiente respuesta generada por un asistente de IA.
    Determina si la respuesta es coherente, relevante para la pregunta original (si se conoce), y segura (no ofensiva, dañina, o sin sentido).
    Responde con 'VALID' si la respuesta es aceptable.
    Si la respuesta no es aceptable, responde con 'INVALID' seguido de una breve explicación del problema.

    Respuesta a evaluar:
    ---
    {response_text}
    ---
    """
    
    loop = asyncio.get_running_loop()
    validation_result = await loop.run_in_executor(
        None,
        llm.invoke,
        [HumanMessage(content=validation_prompt)]
    )
    
    validation_content = validation_result.content.strip().upper()
    
    if validation_content.startswith("INVALID"):
        print(f"❌ Validación de respuesta fallida: {validation_content}")
        # Retorna el mensaje exacto que el usuario desea para PII
        return "LA RESPUESTA REVELA INFORMACIÓN PERSONAL IDENTIFICABLE (PII), COMO EL NOMBRE COMPLETO, NÚMERO DE IDENTIFICACIÓN, FECHA DE NACIMIENTO, ETC. ESTO REPRESENTA UN RIESGO DE SEGURIDAD Y PRIVACIDAD."
    
    return None # La consulta es válida

# --- Ruta para interactuar con el agente (conversación) ---
@app.post("/api/chat")
async def chat_with_agent(request: ChatRequest):
    user_query = request.query
    
    # 1. Validación de la consulta del usuario
    validation_error = validate_user_query(user_query)
    if validation_error:
        return {"response": validation_error, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    
    try:
        context = get_context()
        loop = asyncio.get_running_loop()

        # ¡IMPORTANTE! Reiniciar los contadores de tokens antes de cada nueva invocación del agente
        token_callback_handler.reset_tokens()
        
        # Invocar al agente
        response = await loop.run_in_executor(
            None,
            agent_executor.invoke,
            {"input": user_query, "conversation_context": context}
        )
        
        agent_response = response["output"]
        
        # 2. Validación de la respuesta del LLM
        response_validation_error = await validate_llm_response(agent_response)
        if response_validation_error:
            return {"response": response_validation_error, "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        
        add_to_history(user_query, agent_response)
        
        # 3. Obtener y devolver el uso de tokens desde el callback handler
        token_usage_data = token_callback_handler.get_tokens()
        
        # Fallback: Si el callback handler no reporta tokens, calcularlos manualmente
        if token_usage_data.get('total_tokens', 0) == 0:
            print("⚠️ Callback handler no reportó tokens para chat. Calculando manualmente...")
            prompt_tokens_chat = llm.get_num_tokens(user_query + "\n" + context) # Aproximación
            completion_tokens_chat = llm.get_num_tokens(agent_response)
            token_usage_data = {
                "prompt_tokens": prompt_tokens_chat,
                "completion_tokens": completion_tokens_chat,
                "total_tokens": prompt_tokens_chat + completion_tokens_chat,
            }

        print(f"📊 Uso de Tokens para chat (final):")
        print(f"   Tokens de entrada: {token_usage_data.get('prompt_tokens', 'N/A')}")
        print(f"   Tokens de salida: {token_usage_data.get('completion_tokens', 'N/A')}")
        print(f"   Tokens totales: {token_usage_data.get('total_tokens', 'N/A')}")
        
        return {"response": agent_response, "token_usage": token_usage_data}
    
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
        prompt_text = "Extrae la siguiente información de esta cédula de identidad, si está presente: Nombre completo, Número de Identificación (Cédula), Fecha de Nacimiento, Fecha de Expedición, Fecha de Vencimiento, Género, Nacionalidad. Presenta la información de forma clara y estructurada, por ejemplo, en una lista. Si algún dato no se encuentra, indícalo."
        prompt_content = [
            {
                "type": "text",
                "text": prompt_text
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
        
        # --- Calcular tokens manualmente para la carga de cédula ---
        # Nota: llm.get_num_tokens() solo cuenta tokens de texto. Para imágenes,
        # Gemini cuenta los tokens de la imagen internamente. Aquí solo podemos
        # estimar los tokens del prompt de texto y la respuesta.
        prompt_tokens_cedula = llm.get_num_tokens(prompt_text)
        completion_tokens_cedula = llm.get_num_tokens(extracted_data)
        total_tokens_cedula = prompt_tokens_cedula + completion_tokens_cedula # Esto no incluye tokens de la imagen

        token_usage_data = {
            "prompt_tokens": prompt_tokens_cedula,
            "completion_tokens": completion_tokens_cedula,
            "total_tokens": total_tokens_cedula,
        }
        
        print(f"📊 Uso de Tokens para la lectura de cédula (calculado manualmente):")
        print(f"   Tokens de entrada (prompt de texto): {token_usage_data.get('prompt_tokens', 'N/A')}")
        print(f"   Tokens de salida (respuesta): {token_usage_data.get('completion_tokens', 'N/A')}")
        print(f"   Tokens totales (solo texto): {token_usage_data.get('total_tokens', 'N/A')}")

        # 2. Validación de la respuesta del LLM (para la extracción de cédula)
        response_validation_error = await validate_llm_response(extracted_data)
        if response_validation_error:
            # Return the validation error message, but include the actual token usage
            return {"response": response_validation_error, "token_usage": token_usage_data}
        
        # If validation passes, return extracted data and token usage
        return {"message": "Cédula procesada exitosamente", "extracted_data": extracted_data, "token_usage": token_usage_data}

    except Exception as e:
        print(f"❌ Error al procesar la imagen de la cédula: {e}")
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")

# Para ejecutar la aplicación directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    