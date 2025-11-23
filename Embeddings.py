import os
import glob
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Configuration
MARKDOWN_DIR = "./confluence_docs"  # Where your MD files are
DB_PATH = "./qa_agent_db"           # Where Chroma will store data
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # CPU-optimized, fast, lightweight

# 2. Define Structure-Aware Splitting
# This maps Markdown headers to metadata fields.
# Adjust this based on how your docs are typically written.
headers_to_split_on = [
    ("#", "topic"),          # e.g., "Payment Service Config"
    ("##", "subtopic"),      # e.g., "Timeouts", "Known Errors"
    ("###", "section")       # e.g., "Error 503"
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 3. Secondary Splitter (for long sections)
# If a section is huge (e.g., a giant log dump), we split it further
# but keep the metadata attached.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def load_and_process_markdown():
    print(f"Scanning {MARKDOWN_DIR} for .md files...")
    
    files = glob.glob(f"{MARKDOWN_DIR}/**/*.md", recursive=True)
    all_splits = []

    for file_path in files:
        path_obj = Path(file_path)
        
        # EXTRACT METADATA FROM FILE PATH (Crucial for your "Region/Service" logic)
        # Assuming folder structure: ./docs/payment-service/debugging.md
        # You might need to adjust logic based on your actual folder structure
        try:
            # Example: taking the parent folder as the service name
            service_name = path_obj.parent.name 
            filename = path_obj.stem
        except:
            service_name = "general"
            filename = "unknown"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # STAGE 1: Split by Markdown Headers (The "Smart" Split)
        md_header_splits = markdown_splitter.split_text(content)

        # STAGE 2: Inject File-Level Metadata & Further Split Long Chunks
        for doc in md_header_splits:
            # Add the file-path metadata we extracted earlier
            doc.metadata["service"] = service_name
            doc.metadata["source_file"] = filename
            
            # This handles the "messy" mixed content (tables vs text)
            # If it's a table, it usually fits in one header section.
            
        # Split again if the specific section is too long for the embedding model
        final_splits = text_splitter.split_documents(md_header_splits)
        all_splits.extend(final_splits)

    print(f"Created {len(all_splits)} chunks from {len(files)} files.")
    return all_splits

# 4. Vector Store Ingestion
def ingest_to_chroma(chunks):
    print("Initializing Embedding Model (CPU)...")
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'}
    )

    print("Indexing into ChromaDB...")
    # This will create a persistence folder.
    # collection_metadata={"hnsw:space": "cosine"} optimize for cosine similarity
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=DB_PATH,
        collection_metadata={"hnsw:space": "cosine"} 
    )
    print(f"Success! Database saved to {DB_PATH}")

if __name__ == "__main__":
    chunks = load_and_process_markdown()
    if chunks:
        ingest_to_chroma(chunks)
