import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan di file .env")
genai.configure(api_key=GEMINI_API_KEY)

# --- INISIALISASI DATABASE VEKTOR DAN MODEL AI ---
print("Mempersiapkan database ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(
    name="portfolio",
    embedding_function=chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)
)
print("Database siap.")

print("Mempersiapkan model Generative AI...")
llm = genai.GenerativeModel('gemini-1.5-flash')
print("Model siap.")

# --- MEMBUAT APLIKASI WEB SERVER (API) ---
app = Flask(__name__)
CORS(app) # Mengizinkan akses dari domain lain (website portofolio Anda)

@app.route('/chat', methods=['POST'])
def handle_chat():
    """Endpoint untuk menerima dan menjawab pertanyaan dari pengguna."""
    
    # 1. Ambil pertanyaan pengguna dari request
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    print(f"\nMenerima pertanyaan: {user_question}")

    # 2. Cari konteks yang relevan di database vektor (Retrieval)
    # Kita akan mencari 3 potongan informasi (chunk) yang paling mirip
    results = collection.query(
        query_texts=[user_question],
        n_results=3
    )
    retrieved_context = "\n\n".join(results['documents'][0])
    
    print(f"Konteks yang ditemukan:\n---\n{retrieved_context}\n---")

    # 3. Rakit prompt final untuk LLM (Augmentation)
    prompt_template = f"""
    Anda adalah asisten AI yang ramah dan membantu untuk portofolio Reino Yuris yang bernama BiBoy.
    Tugas Anda adalah menjawab pertanyaan pengguna HANYA berdasarkan informasi yang diberikan di dalam KONTEKS di bawah ini.
    Jika informasi tidak ada di dalam konteks, jawab dengan sopan "Maaf, saya tidak memiliki informasi mengenai hal tersebut."
    
    KONTEKS:
    {retrieved_context}
    
    PERTANYAAN PENGGUNA:
    {user_question}
    
    JAWABAN:
    """

    # 4. Hasilkan jawaban dari LLM (Generation)
    print("Mengirim prompt ke Generative AI untuk mendapatkan jawaban...")
    try:
        response = llm.generate_content(prompt_template)
        final_answer = response.text
        print(f"Jawaban diterima: {final_answer}")
        
        # 5. Kirim jawaban kembali ke pengguna
        return jsonify({"answer": final_answer})
        
    except Exception as e:
        print(f"Error saat memanggil Generative AI: {e}")
        return jsonify({"error": "Terjadi kesalahan saat berkomunikasi dengan AI."}), 500

if __name__ == '__main__':
    # Menjalankan server di port 5001
    app.run(port=5001, debug=True)
