import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()
print("API Keys loaded (though not needed for this script).")

def main():
    """Main function to build and save a standard vector database."""
    # Load all PDF files from the 'data' directory.
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' directory. Please place your PDFs there.")
        return

    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"Error: No PDF files found in '{data_dir}' folder. Please add some PDFs to process.")
        return
        
    print("Loading PDF documents...")
    all_docs = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            return
    
    print(f"Loaded {len(all_docs)} pages from PDFs.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    print(f"Split documents into {len(docs)} chunks.")

    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("\nCreating FAISS vector database directly from document chunks...")
    

    db = FAISS.from_documents(
        documents=docs,
        embedding=embedding_function
    )
    
    db.save_local("faiss_direct_index")
    print("\nVector database created successfully!")
    print("It has been saved to the 'faiss_direct_index' folder.")
    print("Please update your streamlit_app.py to load from this new folder.")

if __name__ == "__main__":
    main()

