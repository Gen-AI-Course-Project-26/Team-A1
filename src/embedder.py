import os
import json
import time
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

PROCESSED_DIR = "data/processed"
VECTOR_DIR    = "data/vectorstore"

CHUNK_FILES = [
    "constitution_chunks.json",
    "ipc_chunks.json",
    "crpc_chunks.json",
    "bnss_chunks.json",
]

# Free HuggingFace embedding model - runs on their servers, no local install
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

# ── Get embeddings from HuggingFace API ───────────────────────────────────────

def get_embeddings(texts):
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload  = {"inputs": texts, "options": {"wait_for_model": True}}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise Exception(f"HF API error {response.status_code}: {response.text}")

    result = response.json()

    # HF returns list of embeddings directly
    if isinstance(result, list) and isinstance(result[0], list):
        return np.array(result, dtype=np.float32)

    raise Exception(f"Unexpected HF response format: {type(result)}")

# ── Cosine similarity ─────────────────────────────────────────────────────────

def cosine_similarity(query_vec, matrix):
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return matrix_norm @ query_norm

# ── Build vector store ────────────────────────────────────────────────────────

def build_vectorstore():
    os.makedirs(VECTOR_DIR, exist_ok=True)

    all_texts    = []
    all_metadata = []

    for filename in CHUNK_FILES:
        path    = os.path.join(PROCESSED_DIR, filename)
        chunks  = json.load(open(path, encoding="utf-8"))
        doc_key = filename.replace("_chunks.json", "")
        print(f"  Loaded {len(chunks)} chunks from {doc_key}")

        for chunk in chunks:
            text = chunk["text"][:512].strip()
            all_texts.append(text)
            all_metadata.append({
                "doc_key":        chunk["doc_key"],
                "section_id":     chunk["section_id"],
                "section_number": chunk["section_number"],
                "section_title":  chunk["section_title"][:100],
                "text":           chunk["text"][:2000],
            })

    total = len(all_texts)
    print(f"\n  Total chunks to embed: {total}")
    print(f"  This will take 5-10 minutes. Please wait...\n")

    # HF free API handles batches of ~64 safely
    batch_size     = 32
    all_embeddings = []

    for i in range(0, total, batch_size):
        batch = all_texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total - 1) // batch_size + 1
        print(f"  Embedding batch {batch_num}/{total_batches} "
              f"({i+1}-{min(i+batch_size, total)} of {total})...")

        for attempt in range(4):
            try:
                embeddings = get_embeddings(batch)
                all_embeddings.append(embeddings)
                break
            except Exception as e:
                wait = (attempt + 1) * 5
                print(f"    Attempt {attempt+1} failed: {e}")
                if attempt < 3:
                    print(f"    Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise

        # Small delay to avoid rate limiting
        time.sleep(1)

    embedding_matrix = np.vstack(all_embeddings)
    print(f"\n  Embedding matrix shape: {embedding_matrix.shape}")

    embeddings_path = os.path.join(VECTOR_DIR, "embeddings.npy")
    metadata_path   = os.path.join(VECTOR_DIR, "metadata.json")

    np.save(embeddings_path, embedding_matrix)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False)

    print(f"  Saved embeddings -> {embeddings_path}")
    print(f"  Saved metadata   -> {metadata_path}")
    return embedding_matrix, all_metadata

# ── Search ────────────────────────────────────────────────────────────────────

def search(query, top_k=5):
    embeddings_path = os.path.join(VECTOR_DIR, "embeddings.npy")
    metadata_path   = os.path.join(VECTOR_DIR, "metadata.json")

    if not os.path.exists(embeddings_path):
        raise Exception("Vector store not built yet. Run build_vectorstore() first.")

    matrix   = np.load(embeddings_path)
    metadata = json.load(open(metadata_path, encoding="utf-8"))

    query_vec = get_embeddings([query])[0]
    scores    = cosine_similarity(query_vec, matrix)
    top_idx   = np.argsort(scores)[::-1][:top_k]

    return [{**metadata[i], "relevance": round(float(scores[i]), 3)} for i in top_idx]

# ── Test retrieval ────────────────────────────────────────────────────────────

def test_retrieval():
    print("\n" + "="*50)
    print("TESTING RETRIEVAL")
    print("="*50)

    queries = [
        "punishment for murder",
        "fundamental rights of citizens",
        "arrest without warrant",
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        results = search(query, top_k=3)
        for r in results:
            print(f"    [{r['doc_key'].upper()} {r['section_number']}] "
                  f"{r['section_title'][:55]} "
                  f"(relevance: {r['relevance']})")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*50)
    print("PHASE 2 — BUILDING VECTOR STORE")
    print("="*50)
    print("\n Loading chunks...")
    build_vectorstore()
    print("\n Running test queries...")
    test_retrieval()
    print("\n Phase 2 Complete!")
    print(" Vector store ready at data/vectorstore/")