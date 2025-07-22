# load_vectorstore.py

import os
import time
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

index = pc.Index(PINECONE_INDEX_NAME)

# Upload, Load, Chunk, Embed, Insert
async def load_vectorestore(uploaded_files):
    embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    file_paths = []

    # Upload
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        file_bytes = await file.read()
        with open(save_path, "wb") as f:
            f.write(file_bytes)
        file_paths.append(str(save_path))

    # Load, chunk, embed, and upload
    for file_path in file_paths:
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        print(f"Extracted text sample: {documents[0].page_content[:800]}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]
        metadata = [{"source": file_path} for _ in range(len(texts))]
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(texts))]

        print(f"\n✅ Total Chunks Created: {len(texts)}")
        for idx, text in enumerate(texts[:5]):
            print(f"\n--- Chunk {idx+1} ---\n{text[:800]}\n---")

        print(f"Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts)
        print(f"\n✅ Sample Embedding Vector Length: {len(embeddings[0])}")

        # Upload to Pinecone
        print("Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            try:
                index.upsert(vectors=zip(ids, embeddings, metadata))
                progress.update(len(embeddings))
            except Exception as e:
                print(f"Pinecone upsert failed: {str(e)}")
                raise
        print(f"✅ Upload and embedding complete for {file_path}\n")
