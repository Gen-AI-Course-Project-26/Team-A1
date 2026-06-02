import os
import json
import numpy as np
import requests
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN     = os.getenv("HF_TOKEN")

VECTOR_DIR  = "data/vectorstore"
MODEL       = "llama-3.3-70b-versatile"
HF_MODEL_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
MAX_RESULTS = 5

# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text):
    headers  = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload  = {"inputs": [text], "options": {"wait_for_model": True}}
    response = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"HF API error {response.status_code}: {response.text}")
    result = response.json()
    if isinstance(result, list) and isinstance(result[0], list):
        return np.array(result[0], dtype=np.float32)
    raise Exception(f"Unexpected HF response: {result}")

# ── Vector search ─────────────────────────────────────────────────────────────

def search(query, top_k=MAX_RESULTS):
    matrix   = np.load(os.path.join(VECTOR_DIR, "embeddings.npy"))
    metadata = json.load(open(os.path.join(VECTOR_DIR, "metadata.json"), encoding="utf-8"))

    query_vec   = get_embedding(query)
    query_norm  = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    scores      = matrix_norm @ query_norm
    top_idx     = np.argsort(scores)[::-1][:top_k]

    return [{**metadata[i], "relevance": round(float(scores[i]), 3)} for i in top_idx]

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(query, sections, simple_language=False):
    doc_names = {
        "constitution": "Constitution of India",
        "ipc":          "Indian Penal Code (IPC)",
        "crpc":         "Code of Criminal Procedure (CrPC)",
        "bnss":         "Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023",
    }

    context = ""
    for i, sec in enumerate(sections, 1):
        context += f"\n[Source {i}] {doc_names.get(sec['doc_key'], sec['doc_key'].upper())}"
        context += f" — Section/Article {sec['section_number']}\n"
        context += f"{sec['text'][:800]}\n"
        context += "-" * 40 + "\n"

    tone = (
        "Explain in simple plain English that anyone can understand. Avoid jargon."
        if simple_language else
        "Provide a precise legal answer. Reference specific sections and articles directly."
    )

    return f"""You are LexRAG, an expert Indian legal assistant.
Answer using ONLY the legal sections provided below.
Do NOT use any knowledge outside the provided sources.
Always cite which Section/Article your answer comes from.
If the sections do not contain enough information, say so clearly.

{tone}

--- LEGAL SOURCES ---
{context}
--- END OF SOURCES ---

User Question: {query}

Answer:"""

# ── Groq LLM ──────────────────────────────────────────────────────────────────

def call_groq(prompt):
    client   = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.1,
    )
    return response.choices[0].message.content

# ── Full RAG pipeline ─────────────────────────────────────────────────────────

def ask(query, simple_language=False):
    print(f"\n{'='*60}")
    print(f" Query: {query}")
    print(f"{'='*60}")

    print("\n Retrieving relevant sections...")
    sections = search(query)

    print(f"  Found {len(sections)} relevant sections:")
    for sec in sections:
        print(f"    [{sec['doc_key'].upper()} {sec['section_number']}] "
              f"{sec['section_title'][:55]} (relevance: {sec['relevance']})")

    prompt = build_prompt(query, sections, simple_language)

    print("\n Generating answer with Groq LLM...")
    answer = call_groq(prompt)

    return {"query": query, "answer": answer, "sources": sections}

def print_result(result):
    doc_names = {"constitution": "Constitution", "ipc": "IPC",
                 "crpc": "CrPC", "bnss": "BNSS 2023"}
    print(f"\n ANSWER:\n{'-'*60}")
    print(result["answer"])
    print(f"{'-'*60}")
    print(f"\n SOURCES USED:")
    for sec in result["sources"]:
        print(f"  * {doc_names.get(sec['doc_key'], sec['doc_key'].upper())} "
              f"Section {sec['section_number']} — "
              f"{sec['section_title'][:60]} (relevance: {sec['relevance']})")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print(" PHASE 3 — RAG PIPELINE")
    print("="*60)

    tests = [
        {"query": "What is the punishment for murder?",                 "simple_language": False},
        {"query": "Can police arrest someone without a warrant?",        "simple_language": False},
        {"query": "What are the fundamental rights of Indian citizens?", "simple_language": True},
    ]

    for t in tests:
        result = ask(t["query"], t["simple_language"])
        print_result(result)
        print()

    print("\n Phase 3 Complete!")