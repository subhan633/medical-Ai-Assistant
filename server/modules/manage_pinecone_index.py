# manage_pinecone_index.py

import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

# Check existing indexes
existing_indexes = [i["name"] for i in pc.list_indexes()]

# Delete existing index if exists
if PINECONE_INDEX_NAME in existing_indexes:
    print(f"Deleting existing index: {PINECONE_INDEX_NAME}...")
    pc.delete_index(PINECONE_INDEX_NAME)

# Create a new index with correct dimension = 768
print(f"Creating new index: {PINECONE_INDEX_NAME} with dimension 768...")
pc.create_index(
    name=PINECONE_INDEX_NAME,
    dimension=768,
    metric="cosine",   # or "dotproduct"
    spec=spec
)

print("✅ Index recreated successfully. Check your Pinecone console to confirm.") 
