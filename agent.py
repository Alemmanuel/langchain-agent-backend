import os, sqlite3, requests, re
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# --- CONFIGURACIÓN GEMINI ---
GOOGLE_API_KEY = "AIzaSyApNBtnuGOQFMf-jFz4m6M1UYkhIob8lSU"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- MODELO GEMINI (manteniendo gemini-2.0-flash-lite) ---
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite", # Este es el modelo funcional y gratuito que estamos usando
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# --- HISTORIAL DE CONVERSACIÓN ---
history = []
def add_to_history(u_in, a_res):
    history.extend([f"Usuario: {u_in}", f"Asistente: {a_res}"])
    if len(history) > 20: history.pop(0); history.pop(0)
def get_context(): return "\n".join(history[-10:]) if history else ""

# --- PRUEBA DE CONEXIÓN ---
def test_connection():
    try:
        print("🔄 Probando conexión con Gemini...")
        print(f"✅ Conexión exitosa! Respuesta: {llm.invoke('Hola').content}")
        return True
    except Exception as e:
        print(f"❌ Error de conexión: {e}\nAsegúrate de que tu API Key de Gemini sea válida y que no hayas excedido tu cuota.")
        return False

# --- EMBEDDINGS & VECTORSTORE ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
def load_vectorstore():
    texts = ["LangChain permite combinar múltiples fuentes de datos.", "FAISS es una librería para búsquedas vectoriales eficientes.", "Los modelos de lenguaje pueden responder preguntas generales y usar herramientas específicas.", "Gemini es un modelo generativo avanzado de Google que puede procesar texto y mantener conversaciones.", "Los agentes conversacionales pueden mantener contexto a lo largo de múltiples intercambios."]
    return FAISS.from_documents([Document(page_content=t) for t in texts], embeddings)
vectorstore = load_vectorstore()

@tool
def buscar_vector(query: str) -> str:
    """Busca información en la base de datos vectorial sobre LangChain, FAISS y tecnologías relacionadas."""
    return "\n".join([doc.page_content for doc in vectorstore.similarity_search(query, k=3)])

# --- BASE DE DATOS SQL ---
def create_sqlite():
    conn = sqlite3.connect("demo.db"); c = conn.cursor()
    c.execute("DROP TABLE IF EXISTS empleados")
    c.execute("CREATE TABLE empleados (id INTEGER PRIMARY KEY, nombre TEXT, rol TEXT, salario INTEGER)")
    c.executemany("INSERT INTO empleados VALUES (?, ?, ?, ?)", [(1, 'Carlos', 'Desarrollador', 50000), (2, 'Ana', 'Diseñadora', 45000), (3, 'Luis', 'Manager', 60000), (4, 'María', 'Desarrolladora', 52000), (5, 'Pedro', 'Tester', 40000)])
    conn.commit(); conn.close(); print("✅ Base de datos SQLite creada correctamente")
create_sqlite()

@tool
def buscar_sql(pregunta: str) -> str:
    """Consulta la base de datos de empleados. Puede buscar por rol, nombre, salario o ID de empleado.
    Ejemplos: 'empleado con ID 2', 'quien es el desarrollador', 'salario de Ana'."""
    conn = sqlite3.connect("demo.db"); c = conn.cursor(); p_lower = pregunta.lower()
    
    try:
        match_id = re.search(r'id\s*(\d+)', p_lower)
        if match_id:
            emp_id = int(match_id.group(1))
            c.execute("SELECT nombre, rol, salario FROM empleados WHERE id = ?", (emp_id,))
            res = c.fetchone()
            conn.close()
            if res: return f"Empleado con ID {emp_id}: {res[0]}, Rol: {res[1]}, Salario: ${res[2]:,}"
            else: return f"No se encontró empleado con ID {emp_id}."

        if "desarrollador" in p_lower or "dev" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Desarrollador%'")
        elif "diseñador" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Diseñador%'")
        elif "manager" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Manager%'")
        elif "tester" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados WHERE rol LIKE '%Tester%'")
        elif "salario" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados ORDER BY salario DESC")
        elif "mayor" in p_lower and "salario" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados ORDER BY salario DESC LIMIT 1")
        elif "menor" in p_lower and "salario" in p_lower: c.execute("SELECT nombre, rol, salario FROM empleados ORDER BY salario ASC LIMIT 1")
        elif "salario de" in p_lower:
            name_match = re.search(r'salario de (\w+)', p_lower)
            if name_match:
                name = name_match.group(1).capitalize()
                c.execute("SELECT nombre, rol, salario FROM empleados WHERE nombre = ?", (name,))
                res = c.fetchone()
                conn.close()
                if res: return f"El salario de {res[0]} ({res[1]}) es ${res[2]:,}"
                else: return f"No se encontró el salario para {name}."
            else: c.execute("SELECT nombre, rol, salario FROM empleados")
        else: c.execute("SELECT nombre, rol, salario FROM empleados")
        
        res = c.fetchall(); conn.close()
        if not res: return "No se encontraron empleados con esos criterios."
        return "Empleados encontrados:\n" + "\n".join([f"- {e[0]}: {e[1]} (Salario: ${e[2]:,})" for e in res])
    except Exception as e: conn.close(); return f"Error en la consulta: {str(e)}"

# --- API Rick and Morty ---
@tool
def buscar_rick_morty(query: str) -> str:
    """Busca personajes de Rick and Morty por ID o nombre.
    Ejemplos: 'personaje 2', 'Rick Sanchez'."""
    try:
        match_id = re.search(r'\b(\d+)\b', query)
        if match_id:
            char_id = int(match_id.group(1))
            url = f"https://rickandmortyapi.com/api/character/{char_id}"
        else: url = f"https://rickandmortyapi.com/api/character/?name={query}"
        
        r = requests.get(url); data = r.json()
        if r.status_code != 200: return f"No se encontró información para: {query}"
        char = data['results'][0] if 'results' in data and data['results'] else data
        
        if not char: return f"No se encontró información para: {query}"

        return f"🎭 Personaje: {char['name']}\n🧬 Especie: {char['species']}\n👤 Género: {char['gender']}\n🌍 Origen: {char['origin']['name']}\n📍 Ubicación: {char['location']['name']}\n📺 Episodios: {len(char['episode'])} apariciones\n💫 Estado: {char['status']}"
    except Exception as e: return f"Error al buscar el personaje: {str(e)}"

# --- CONOCIMIENTO GENERAL ---
@tool
def responder_pregunta_general(pregunta: str) -> str:
    """Responde preguntas generales usando el conocimiento del modelo de lenguaje."""
    p_general = f"Contexto: {get_context()}\nPregunta: {pregunta}\nResponde de manera conversacional y útil."
    try: return llm.invoke(p_general).content
    except Exception as e: return f"No pude procesar esa pregunta: {str(e)}"

# --- TOOLS & PROMPT ---
tools = [buscar_vector, buscar_sql, buscar_rick_morty, responder_pregunta_general]
prompt = PromptTemplate.from_template("""Eres un asistente conversacional inteligente.

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
Thought:{agent_scratchpad}""")

# --- AGENTE ---
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=5)

# --- APP PRINCIPAL ---
if __name__ == "__main__":
    print("🤖 Agente Conversacional Multi-fuente con Gemini")
    print("=" * 55)
    if not test_connection(): exit(1)
    print("\nCapacidades:\n✅ Conversación general\n✅ Base de datos vectorial\n✅ Base de datos SQL (con búsqueda por ID)\n✅ API Rick and Morty (con búsqueda por ID/nombre)\n✅ Mantiene contexto\n✅ Powered by Google Gemini 1.5 Flash\n" + "=" * 55)
    print("Ejemplos:\n- '¿Cómo estás?'\n- '¿Qué es Python?'\n- '¿Quién gana más?'\n- 'Busca a Morty'\n- '¿Qué es LangChain?'\n- 'Cuéntame un chiste'\n- 'Quién es el ID 2'\n- 'Personaje de Rick and Morty con ID 1'\n" + "=" * 55)
    print("\nEscribe 'salir' para terminar.\n")
    while True:
        q = input("\n💬 Tú: ");
        if q.lower() in ["salir", "exit", "quit"]: print("👋 ¡Hasta luego!"); break
        try:
            res = agent_executor.invoke({"input": q, "conversation_context": get_context()})
            print(f"\n🤖 Asistente: {res['output']}"); add_to_history(q, res['output'])
        except Exception as e:
            err_msg = f"Disculpa, tuve un problema: {str(e)}"; print(f"\n❌ {err_msg}"); add_to_history(q, err_msg)
