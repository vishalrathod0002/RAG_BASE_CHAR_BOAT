import os
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from werkzeug.utils import secure_filename

# LangChain Imports
from langchain_community.document_loaders import (
    CSVLoader, PyPDFLoader, TextLoader, Docx2txtLoader,
    UnstructuredMarkdownLoader, UnstructuredExcelLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Configuration
app = Flask(__name__)
app.secret_key = "super_secret_key_for_session"

# Paths
INDEX_PATH = "faiss_index"
DOCS_PATH = os.path.join(os.getcwd(), "documents")
ALLOWED_EXTENSIONS = {'pdf', 'csv', 'docx', 'md', 'txt', 'xlsx', 'json', 'html'}

# User DB
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin'},
    'user':  {'password': 'user123',  'role': 'user'}
}

os.makedirs(DOCS_PATH, exist_ok=True)

# --- GLOBAL VARIABLES (In-Memory Cache) ---
global_retriever = None 
global_llm = None

# --- RAG & Helper Functions ---

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "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")
    return ChatGroq(
        model="openai/gpt-oss-120b",  # <--- CHANGED TO VALID MODEL NAME
        groq_api_key=api_key,
        temperature=0
    )

def refresh_vectorstore():
    """Reads files, processes them, builds index, and updates GLOBAL variable."""
    print("Processing documents for Vector DB...")
    documents = []
    
    for file in os.listdir(DOCS_PATH):
        file_path = os.path.join(DOCS_PATH, file)
        extension = os.path.splitext(file)[1].lower()
        
        try:
            if extension == ".pdf": loader = PyPDFLoader(file_path)
            elif extension == ".csv": loader = CSVLoader(file_path)
            elif extension == ".docx": loader = Docx2txtLoader(file_path)
            elif extension == ".md": loader = UnstructuredMarkdownLoader(file_path)
            elif extension == ".txt": loader = TextLoader(file_path)
            elif extension == ".xlsx": loader = UnstructuredExcelLoader(file_path)
            else: continue

            documents.extend(loader.load())
        except Exception as e: print(f"Error loading {file}: {e}")

    if not documents: return None, 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    
    # Build and Save to Disk
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(INDEX_PATH)
    
    # Update GLOBAL Retriever (This is the important fix)
    global global_retriever
    global_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    return vectorstore, len(documents)

def load_existing_index_into_memory():
    """Loads the saved disk index into the GLOBAL variable."""
    global global_retriever
    if os.path.exists(INDEX_PATH):
        print("Loading existing Vector Store into Memory...")
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        global_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    else:
        print("No existing index found on disk.")

# --- Routes ---

@app.route('/')
def index():
    if 'logged_in' not in session: return redirect(url_for('login_page'))
    return render_template('index.html', role=session.get('role'))

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.json
    user = USERS.get(data.get('username'))
    if user and user['password'] == data.get('password'):
        session['logged_in'] = True
        session['role'] = user['role']
        return jsonify({'success': True, 'role': user['role']})
    return jsonify({'success': False, 'message': 'Invalid credentials'}), 401


# NEW: Register Route
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Missing fields'}), 400
    
    if username in USERS:
        return jsonify({'success': False, 'message': 'User already exists'}), 400
    
    # Create new user (Default role is 'user')
    USERS[username] = {'password': password, 'role': 'user'}
    
    return jsonify({'success': True, 'message': 'Account created successfully'}),

# Add this to your imports if not already there
# (You likely already have them from previous steps)
# from langchain_community.document_loaders import ...

# --- NEW ROUTE TO READ FILE CONTENT ---
@app.route('/api/read_file/<filename>')
def read_file(filename):
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    
    # Secure the filename to prevent directory traversal
    filename = secure_filename(filename)
    file_path = os.path.join(DOCS_PATH, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:
        extension = os.path.splitext(filename)[1].lower()
        
        # Re-use the logic from read_file() but just for one file
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".csv":
            loader = CSVLoader(file_path)
        elif extension == ".docx":
            loader = Docx2txtLoader(file_path)
        elif extension == ".md":
            loader = UnstructuredMarkdownLoader(file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path)
        elif extension == ".xlsx":
            loader = UnstructuredExcelLoader(file_path)
        else:
            return jsonify({'content': 'File format not supported for reading.'})

        documents = loader.load()
        # Join all pages/chunks into one string
        content = "\n\n--- PAGE BREAK ---\n\n".join([doc.page_content for doc in documents])
        
        return jsonify({'filename': filename, 'content': content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
                   
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

@app.route('/api/files', methods=['GET'])
def list_files():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    files = [f for f in os.listdir(DOCS_PATH) if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]
    return jsonify({'files': files})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'logged_in' not in session or session.get('role') != 'admin': return jsonify({'error': 'Unauthorized'}), 403
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file.save(os.path.join(DOCS_PATH, filename))
    
    return jsonify({'message': 'File uploaded (pending index)', 'filename': filename})

@app.route('/api/load_index', methods=['POST'])
def load_index():
    """Admin triggers this to process files and update GLOBAL memory."""
    if 'logged_in' not in session or session.get('role') != 'admin': return jsonify({'error': 'Unauthorized'}), 403
    try:
        vectorstore, doc_count = refresh_vectorstore()
        if vectorstore:
            return jsonify({'message': f'Vector DB updated in memory with {doc_count} documents.'})
        else:
            return jsonify({'message': 'No valid documents found.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    if 'logged_in' not in session: return jsonify({'error': 'Unauthorized'}), 401
    data = request.json
    query = data.get('query')
    if not query: return jsonify({'error': 'No query provided'}), 400

    try:
        # USE GLOBAL RETRIEVER (No disk loading here!)
        if not global_retriever:
            return jsonify({'answer': 'Vector DB is not loaded in memory. Ask Admin to click "Load to Vector DB".'})
            
        docs = global_retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        
        # Use Global LLM as well (cached)
        llm = get_llm()

        # 3. UPDATED PROMPT (Allows General Knowledge)
        prompt = f"""
        You are a helpful AI assistant.
        
        Use the following pieces of retrieved context to answer the question. 
        If the context is relevant, prioritize it. 
        If the context is NOT relevant or does not contain the answer, use your general knowledge to answer the user.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        response = llm.invoke(prompt)
        return jsonify({'answer': response.content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # LOAD EXISTING INDEX INTO MEMORY ON STARTUP
    load_existing_index_into_memory()
    
    print("\n" + "="*50)
    print("RAG APP STARTED")
    print("Vector Store Status:", "Loaded in Memory" if global_retriever else "Empty")
    print("Admin: admin / admin123")
    print("User: user / user123")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000)