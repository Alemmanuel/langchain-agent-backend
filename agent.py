import os, sqlite3, requests, re
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.callbacks import BaseCallbackHandler  # Importar BaseCallbackHandler
from langchain_core.outputs import LLMResult  # Importar LLMResult para tipado
from markupsafe import escape  # Asegúrate de importar esto al inicio si no está

# --- CONFIGURACIÓN GEMINI ---
GOOGLE_API_KEY = "AIzaSyApNBtnuGOQFMf-jFz4m6M1UYkhIob8lSU"  # Asegúrate de que esta sea tu API Key válida
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- MODELO GEMINI (manteniendo gemini-2.0-flash) ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",  # ¡Este es el modelo Gemini que estamos usando!
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY,
)

# --- HISTORIAL DE CONVERSACIÓN ---
history = []


def add_to_history(u_in, a_res):
    history.extend([f"Usuario: {u_in}", f"Asistente: {a_res}"])
    if len(history) > 20:
        history.pop(0)
        history.pop(0)


def get_context():
    return "\n".join(history[-10:]) if history else ""


# --- PRUEBA DE CONEXIÓN ---
def test_connection():
    try:
        print("🔄 Probando conexión con Gemini...")
        print(f"✅ Conexión exitosa! Respuesta: {llm.invoke('Hola').content}")
        return True
    except Exception as e:
        print(
            f"❌ Error de conexión: {e}\nAsegúrate de que tu API Key de Gemini sea válida y que no hayas excedido tu cuota."
        )
        return False


# --- EMBEDDINGS & VECTORSTORE ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=GOOGLE_API_KEY
)


def load_vectorstore():
    texts = [
        "LangChain permite combinar múltiples fuentes de datos.",
        "FAISS es una librería para búsquedas vectoriales eficientes.",
        "Los modelos de lenguaje pueden responder preguntas generales y usar herramientas específicas.",
        "Gemini es un modelo generativo avanzado de Google que puede procesar texto y mantener conversaciones.",
        "Los agentes conversacionales pueden mantener contexto a lo largo de múltiples intercambios.",
    ]
    return FAISS.from_documents([Document(page_content=t) for t in texts], embeddings)


vectorstore = load_vectorstore()


@tool
def buscar_vector(query: str) -> str:
    """Busca información en la base de datos vectorial sobre LangChain, FAISS y tecnologías relacionadas."""
    return "<h2>Resultados encontrados:</h2><ul>" + "".join([f"<li>{doc.page_content}</li>" for doc in vectorstore.similarity_search(query, k=3)]) + "</ul>"



# --- BASE DE DATOS SQL ---
def create_sqlite():
    conn = sqlite3.connect("demo.db")
    c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS empleados")
    c.execute(
        "CREATE TABLE empleados (id INTEGER PRIMARY KEY, nombre TEXT, rol TEXT, salario INTEGER)"
    )
    c.executemany(
        "INSERT INTO empleados VALUES (?, ?, ?, ?)",
        [
            (1, "Carlos", "Desarrollador", 50000),
            (2, "Ana", "Diseñadora", 45000),
            (3, "Luis", "Manager", 60000),
            (4, "María", "Desarrolladora", 52000),
            (5, "Pedro", "Tester", 40000),
        ],
    )
    conn.commit()
    conn.close()
    print("✅ Base de datos SQLite creada correctamente")


create_sqlite()


@tool
def buscar_sql(pregunta: str) -> str:
    """Consulta la base de datos de empleados. Puede buscar por rol, nombre, salario o ID de empleado.
    Ejemplos: 'empleado con ID 2', 'quien es el desarrollador', 'salario de Ana'."""
    conn = sqlite3.connect("demo.db")
    c = conn.cursor()
    p_lower = pregunta.lower()

    try:
        match_id = re.search(r"id\s*(\d+)", p_lower)
        if match_id:
            emp_id = int(match_id.group(1))
            c.execute(
                "SELECT nombre, rol, salario FROM empleados WHERE id = ?", (emp_id,)
            )
            res = c.fetchone()
            conn.close()
            if res:
                return f"Empleado con ID {emp_id}: {res[0]}, Rol: {res[1]}, Salario: ${res[2]:,}"
            else:
                return f"No se encontró empleado con ID {emp_id}."

        if "desarrollador" in p_lower or "dev" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Desarrollador%'"
            )
        elif "diseñador" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Diseñador%'"
            )
        elif "manager" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Manager%'"
            )
        elif "tester" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Tester%'"
            )
        elif "salario" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados ORDER BY salario DESC"
            )
        elif "mayor" in p_lower and "salario" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados ORDER BY salario DESC LIMIT 1"
            )
        elif "menor" in p_lower and "salario" in p_lower:
            c.execute(
                "SELECT nombre, rol, salario FROM empleados ORDER BY salario ASC LIMIT 1"
            )
        elif "salario de" in p_lower:
            name_match = re.search(r"salario de (\w+)", p_lower)
            if name_match:
                name = name_match.group(1).capitalize()
                c.execute(
                    "SELECT nombre, rol, salario FROM empleados WHERE nombre = ?",
                    (name,),
                )
                res = c.fetchone()
                conn.close()
                if res:
                    return f"El salario de {res[0]} ({res[1]}) es ${res[2]:,}"
                else:
                    return f"No se encontró el salario para {name}."
            else:
                c.execute("SELECT nombre, rol, salario FROM empleados")
        else:
            c.execute("SELECT nombre, rol, salario FROM empleados")

        res = c.fetchall()
        conn.close()
        if not res:
            return "No se encontraron empleados con esos criterios."
        return "<h2>Empleados encontrados:</h2><ul>" + "".join([f"<li><strong>{e[0]}</strong>: {e[1]} – <code>${e[2]:,}</code></li>" for e in res]) + "</ul>".join(
            [f"- {e[0]}: {e[1]} (Salario: ${e[2]:,})" for e in res]
        )
    except Exception as e:
        conn.close()
        return f"Error en la consulta: {str(e)}"


# --- API Rick and Morty ---
@tool
def buscar_rick_morty(query: str) -> str:
    """Busca personajes de Rick and Morty por ID o nombre.
    Ejemplos: 'personaje 2', 'Rick Sanchez'."""
    try:
        match_id = re.search(r"\b(\d+)\b", query)
        if match_id:
            char_id = int(match_id.group(1))
            url = f"https://rickandmortyapi.com/api/character/{char_id}"
        else:
            url = f"https://rickandmortyapi.com/api/character/?name={query}"

        r = requests.get(url)
        data = r.json()
        if r.status_code != 200:
            return f"No se encontró información para: {query}"
        char = data["results"][0] if "results" in data and data["results"] else data

        if not char:
            return f"No se encontró información para: {query}"

        return f"""
            <h2>{char['name']}</h2>
            <p><strong>Especie:</strong> {char['species']}<br>
            <strong>Género:</strong> {char['gender']}<br>
            <strong>Origen:</strong> {char['origin']['name']}<br>
            <strong>Ubicación actual:</strong> {char['location']['name']}<br>
            <strong>Episodios:</strong> {len(char['episode'])} apariciones<br>
            <strong>Estado:</strong> {char['status']}</p>
            """

    except Exception as e:
        return f"Error al buscar el personaje: {str(e)}"


# --- CONOCIMIENTO GENERAL ---
@tool
def responder_pregunta_general(pregunta: str) -> str:
    """Responde preguntas generales usando el conocimiento del modelo de lenguaje."""
    p_general = f"""
        Contexto:
        {get_context()}

        Instrucciones:
        Responde directamente como si hablaras con "The Special One". Devuelve solo contenido HTML válido, sin ```html ni ``` ni etiquetas de bloque de código.
        Usa <p>, <strong> y <br> para dar estilo a tu respuesta. Usa emojis si van bien con el tono. No expliques, no saludes, no repitas la pregunta. Solo responde directo con HTML.
        Pregunta: {pregunta}
        """



    try:
        return llm.invoke(p_general).content
    except Exception as e:
        return f"No pude procesar esa pregunta: {str(e)}"


# --- TOKEN USAGE TRACKING ---
class CustomTokenCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Se ejecuta al finalizar una llamada a un LLM."""
        print(f"\n--- DEBUG: on_llm_end called ---")
        print(f"Full LLMResult object: {response}")  # Print the full object
        print(f"LLM Output: {response.llm_output}")
        print(f"Generations: {response.generations}")

        # Helper para buscar 'token_usage' recursivamente en diccionarios/listas
        def _find_token_usage_recursively(data):
            if isinstance(data, dict):
                if "token_usage" in data and isinstance(data["token_usage"], dict):
                    print(
                        f"DEBUG: Found token_usage directly in dict: {data['token_usage']}"
                    )
                    return data["token_usage"]
                for key, value in data.items():
                    # print(f"DEBUG: Searching in dict key: {key}") # Demasiado verboso, comentar
                    result = _find_token_usage_recursively(value)
                    if result:
                        return result
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    # print(f"DEBUG: Searching in list item index: {i}") # Demasiado verboso, comentar
                    result = _find_token_usage_recursively(item)
                    if result:
                        return result
            return None

        # 1. Intentar encontrar token_usage en llm_output (puede estar anidado)
        usage = _find_token_usage_recursively(response.llm_output)
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            print(f"DEBUG: Tokens accumulated from llm_output: {usage}")
            print(
                f"DEBUG: Current accumulated tokens: P={self.prompt_tokens}, C={self.completion_tokens}, T={self.total_tokens}"
            )
            return

        # 2. Si no se encontró en llm_output, intentar en generation_info de cada generación
        for i, generation in enumerate(response.generations):
            print(f"DEBUG: Checking generation {i}: {generation}")
            if generation.generation_info:
                usage = _find_token_usage_recursively(generation.generation_info)
                if usage:
                    self.prompt_tokens += usage.get("prompt_tokens", 0)
                    self.completion_tokens += usage.get("completion_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)
                    print(f"DEBUG: Tokens accumulated from generation_info: {usage}")
                    print(
                        f"DEBUG: Current accumulated tokens: P={self.prompt_tokens}, C={self.completion_tokens}, T={self.total_tokens}"
                    )
                    return  # Salir después de encontrar el primer uso de tokens

        print("DEBUG: No token usage found after exhaustive search for this LLM call.")
        print(
            f"DEBUG: Current accumulated tokens: P={self.prompt_tokens}, C={self.completion_tokens}, T={self.total_tokens}"
        )
        print(f"--- END DEBUG ---")

    def reset_tokens(self):
        """Reinicia los contadores de tokens."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def get_tokens(self):
        """Retorna el uso de tokens acumulado."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


# Instanciar el handler globalmente para que el backend_app pueda acceder a él
token_callback_handler = CustomTokenCallbackHandler()

# --- TOOLS & PROMPT ---
tools = [buscar_vector, buscar_sql, buscar_rick_morty, responder_pregunta_general]
prompt = PromptTemplate.from_template(
    """Eres un asistente conversacional inteligente.

Contexto de conversación:
{conversation_context}

Tienes acceso a las siguientes herramientas:
{tools}

INSTRUCCIONES:
- Siempre piensa paso a paso.
- Si la pregunta es sobre un ID, intenta determinar si es un ID de empleado o de personaje de Rick and Morty.
- Para preguntas generales o conversacionales, usa "responder_pregunta_general".
- Para información sobre empleados (incluyendo por ID), usa "buscar_sql".
- Para personajes de Rick and Morty (incluyendo por ID), usa "buscar_rick_morty".
- Para información técnica sobre LangChain/FAISS, usa "buscar_vector".
- Mantén un tono conversacional y amigable.
- Haz referencia a conversaciones anteriores cuando sea relevante.
- Responde como si estuvieras hablando de tú a tú con The Special One.
- Usa HTML simple: envuelve tu respuesta en <p>, resalta frases clave con <strong> y usa <br> para separar ideas o frases.
- Nada de encabezados, listas ni títulos robóticos.
- Usa un tono conversacional, profesional y con personalidad.
- Está bien usar emojis para dar tono, pero sin exagerar.
- Devuelve solo el contenido HTML. No uses bloques de código como ```html o ```.
- No expliques, no introduzcas, no comentes. Solo la respuesta pura en HTML renderizable.


Usa el siguiente formato exacto para tus pensamientos y acciones:

Question: la pregunta de entrada que debes responder
Thought: siempre debes pensar sobre qué hacer, analizando la pregunta y decidiendo la mejor herramienta.
Action: el_nombre_de_la_herramienta_a_usar (debe ser una de [{tool_names}])
Action Input: la_entrada_exacta_para_la_herramienta_seleccionada (ej. 'empleado con ID 2' o '2' para Rick and Morty)
Observation: el resultado de la acción
... (este Thought/Action/Action Input/Observation puede repetirse si es necesario)
Thought: Ahora sé la respuesta final
Final Answer: la respuesta final a la pregunta de entrada original

Comienza!

Question: {input}
Thought:{agent_scratchpad}"""
)

# --- AGENTE ---
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    callbacks=[token_callback_handler],  # ¡Aquí se añade el callback handler!
)

# Las siguientes líneas se comentan o eliminan para que el archivo sea importable
# if __name__ == "__main__":
#     print("🤖 Agente Conversacional Multi-fuente con Gemini")
#     print("=" * 55)
#     if not test_connection(): exit(1)
#     print("\nCapacidades:\n✅ Conversación general\n✅ Base de datos vectorial\n✅ Base de datos SQL (con búsqueda por ID)\n✅ API Rick and Morty (con búsqueda por ID/nombre)\n✅ Mantiene contexto\n✅ Powered by Google Gemini 1.5 Flash\n" + "=" * 55)
#     print("Ejemplos:\n- '¿Cómo estás?'\n- '¿Qué es Python?'\n- '¿Quién gana más?'\n- 'Busca a Morty'\n- '¿Qué es LangChain?'\n- 'Cuéntame un chiste'\n- 'Quién es el ID 2'\n- 'Personaje de Rick and Morty con ID 1'\n" + "=" * 55)
#     print("\nEscribe 'salir' para terminar.\n")
#     while True:
#         q = input("\n💬 Tú: ");
#         if q.lower() in ["salir", "exit", "quit"]: print("👋 ¡Hasta luego!"); break
#         try:
#             res = agent_executor.invoke({"input": q, "conversation_context": get_context()})
#             print(f"\n🤖 Asistente: {res['output']}"); add_to_history(q, res['output'])
#         except Exception as e:
#             err_msg = f"Disculpa, tuve un problema: {str(e)}"; print(f"\n❌ {err_msg}"); add_to_history(q, err_msg)
