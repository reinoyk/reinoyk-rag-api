import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Konfigurasi API key Google
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY tidak ditemukan")
genai.configure(api_key=GEMINI_API_KEY)

def load_document(filepath):
    print(f"Membaca dokumen dari: {filepath}...")
    content = ""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    print("Dokumen berhasil dibaca.")
    return content

def chunk_document(content):
    print("Memecah dokumen menjadi chunks...")
    
    chunks = content.split('\n\n')
    
    # Membersihkan spasi kosong yang tidak perlu dari setiap chunk
    cleaned_chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50] 
    print(f"Berhasil membuat {len(cleaned_chunks)} chunks.")
    return cleaned_chunks

def create_embeddings_and_store(chunks):
    """Membuat embeddings untuk setiap chunk dan menyimpannya di ChromaDB."""
    print("Mempersiapkan database vektor ChromaDB...")

    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Membuat tabel bernama 'portfolio' dan Model embedding dari Google
    collection = client.get_or_create_collection(
        name="portfolio",
        embedding_function=chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=GEMINI_API_KEY)
    )
    
    print("Membuat embeddings dan menyimpan ke database (proses ini mungkin butuh beberapa saat)...")
    # Menambahkan semua chunk ke dalam koleksi.
    # ChromaDB akan secara otomatis:
    # 1. Memanggil API embedding Google untuk setiap chunk.
    # 2. Menyimpan teks chunk beserta hasil embedding-nya.
    # 3. Membuat ID unik untuk setiap entri.
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("\n==============================================")
    print("✅ Database vektor berhasil dibuat!")
    print(f"Total {collection.count()} potongan informasi telah diindeks.")
    print("Lokasi database: ./chroma_db")
    print("==============================================")

def main():
    filepath = "knowledge-base.md"
    
    # Muat dokumen
    document_content = load_document(filepath)
    
    # Pecah dokumen
    chunks = chunk_document(document_content)
    
    # Buat embeddings dan simpan
    create_embeddings_and_store(chunks)

if __name__ == "__main__":
    main()
