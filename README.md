# ⚖️ LawRAG

### A Hierarchy-Aware Legal RAG System for Indian Courts

> Retrieve · Verify · Detect · Explain

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange?style=flat-square)]()

---

## Overview

The Indian judicial system produces thousands of judgments each year across the Supreme Court, High Courts, and subordinate courts. Navigating this vast body of jurisprudence is challenging — not just because of volume, but because **court hierarchy governs precedence**, cited cases may be fictitious or misconstrued, and legal positions can be silently overruled over time.

**LawRAG** is a Retrieval-Augmented Generation (RAG) system purpose-built for Indian legal research. It ranks retrieved judgments by judicial hierarchy, generates grounded legal answers with verifiable citations, detects conflicting or overruled precedents, and produces transparent, explainable outputs.

---

## The Problem

| Gap | Impact |
|-----|--------|
| Existing tools ignore court hierarchy | Lower-court judgments ranked alongside Supreme Court precedents |
| No citation verification | Risk of hallucinated or contextually inaccurate case references |
| No conflict detection | Outdated or overruled judgments presented as valid precedent |
| Opaque outputs | No explanation of *why* a judgment is relevant or authoritative |

---

## Key Features

### 🏛️ Hierarchy-Aware Retrieval
Judgments are ranked and weighted by their position in the Indian judicial hierarchy. A Supreme Court constitutional bench ruling carries more weight than a single-judge High Court order — and LawRAG treats them accordingly.

```
Supreme Court (Constitutional Bench)
    └── Supreme Court (Division / Single Bench)
        └── High Courts
            └── District & Subordinate Courts
```

### ✅ Citation Verification
Every case cited in a generated answer is verified against the document corpus. LawRAG flags:
- Citations that do not exist in the knowledge base
- Citations used out of their original context or ratio decidendi

### ⚡ Conflict Detection
LawRAG identifies when two retrieved judgments take opposing legal positions on the same issue — including cases where a later judgment has **overruled**, **distinguished**, or **departed** from an earlier one.

### 🔍 Explainable Outputs
Each answer includes:
- The source judgment(s) it draws from
- The court and bench level of each source
- A relevance rationale explaining why the judgment applies
- Conflict or caution warnings where applicable

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Query Understanding                        │
│        (Legal domain classification + entity extraction)    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Hierarchy-Aware Retrieval Engine               │
│   Vector Search  →  Court-Level Reranking  →  Top-K Docs   │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
     ┌────────────┐ ┌─────────────┐ ┌──────────────┐
     │  Citation  │ │  Conflict   │ │  Relevance   │
     │  Verifier  │ │  Detector   │ │  Explainer   │
     └────────────┘ └─────────────┘ └──────────────┘
              │             │             │
              └─────────────┼─────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Answer Generation (LLM)                    │
│           Grounded · Cited · Conflict-Flagged               │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
lawRAG/
├── data/
│   ├── raw/                    # Raw judgment files (PDF / HTML)
│   ├── processed/              # Cleaned and chunked documents
│   └── metadata/               # Court level, date, bench info
│
├── ingestion/
│   ├── scraper.py              # Judgment scrapers (SC, HC sources)
│   ├── parser.py               # PDF/HTML → structured text
│   └── chunker.py              # Hierarchy-aware document chunking
│
├── indexing/
│   ├── embedder.py             # Legal domain embeddings
│   ├── vector_store.py         # Vector DB interface (Chroma / Qdrant)
│   └── metadata_index.py       # Court-level metadata index
│
├── retrieval/
│   ├── retriever.py            # Base retrieval logic
│   ├── reranker.py             # Hierarchy-aware reranking
│   └── filters.py              # Date, court, and bench filters
│
├── verification/
│   ├── citation_verifier.py    # Citation existence + context check
│   └── conflict_detector.py    # Overruling / contradiction detection
│
├── generation/
│   ├── prompt_builder.py       # Legal-domain prompt templates
│   ├── answer_generator.py     # LLM answer generation
│   └── explainer.py            # Rationale and citation explainer
│
├── api/
│   ├── main.py                 # FastAPI application
│   └── schemas.py              # Request/response models
│
├── evaluation/
│   ├── benchmarks/             # Legal QA benchmark datasets
│   └── eval.py                 # Faithfulness, accuracy, hallucination metrics
│
├── notebooks/                  # Exploratory and demo notebooks
├── tests/                      # Unit and integration tests
├── config.yaml                 # Model and system configuration
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A vector database (Chroma for local dev, Qdrant for production)
- An LLM API key (OpenAI / Anthropic / local model via Ollama)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/lawRAG.git
cd lawRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
cp config.yaml.example config.yaml
```

Edit `config.yaml` to set your:
- LLM provider and API key
- Vector store connection
- Court hierarchy weights
- Retrieval top-K and reranking parameters

### Ingesting Judgments

```bash
# Ingest from a directory of judgment PDFs
python ingestion/parser.py --input data/raw/ --output data/processed/

# Build the vector index
python indexing/embedder.py --docs data/processed/ --store ./vector_db
```

### Running the API

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`. Swagger docs at `/docs`.

### Example Query

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "question": "What is the test for determining anticipatory bail under Section 438 CrPC?",
    "filters": {
        "court_level": ["supreme_court", "high_court"],
        "date_from": "2010-01-01"
    }
})

print(response.json())
```

**Sample Output:**

```json
{
  "answer": "The Supreme Court in Siddharam Satlingappa Mhetre v. State of Maharashtra (2011) laid down...",
  "sources": [
    {
      "case_name": "Siddharam Satlingappa Mhetre v. State of Maharashtra",
      "citation": "(2011) 1 SCC 694",
      "court": "Supreme Court of India",
      "bench": "Division Bench",
      "relevance_score": 0.94,
      "rationale": "Directly addresses the multi-factor test for anticipatory bail."
    }
  ],
  "conflicts_detected": [],
  "citation_verification": {
    "all_verified": true,
    "flagged": []
  }
}
```

---

## Judicial Hierarchy Model

LawRAG uses a configurable hierarchy weight to rerank retrieved documents:

| Court | Hierarchy Level | Default Weight |
|-------|----------------|---------------|
| Supreme Court — Constitutional Bench (5+ judges) | 1 | 1.00 |
| Supreme Court — Division Bench | 2 | 0.90 |
| Supreme Court — Single Bench | 3 | 0.85 |
| High Court — Full Bench | 4 | 0.75 |
| High Court — Division Bench | 5 | 0.65 |
| High Court — Single Bench | 6 | 0.55 |
| District / Subordinate Courts | 7 | 0.30 |

Weights are combined with semantic similarity scores to produce a final **hierarchy-adjusted relevance score**.

---

## Evaluation

LawRAG is evaluated across four dimensions:

| Metric | Description |
|--------|-------------|
| **Retrieval Accuracy** | Whether the correct authoritative judgment was retrieved |
| **Answer Faithfulness** | Whether the answer stays grounded in retrieved documents |
| **Citation Precision** | Whether cited cases exist and are used in correct context |
| **Conflict Recall** | Whether genuine contradictions in the corpus are flagged |

```bash
python evaluation/eval.py --benchmark evaluation/benchmarks/sc_qa.json
```

---

## Roadmap

- [ ] Judgment scraper for Indian Kanoon and Supreme Court website
- [ ] Legal-domain fine-tuned embedding model
- [ ] Overruling relationship graph (Neo4j)
- [ ] Multi-hop reasoning for complex constitutional questions
- [ ] Support for legislative text (Acts, Amendments) alongside case law
- [ ] Hindi language query support
- [ ] Web UI for legal researchers

---

## Contributing

Contributions are welcome. Please open an issue before submitting a pull request so we can discuss the proposed change.

```bash
# Run tests before submitting
pytest tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for coding standards and branch conventions.

---

## Disclaimer

LawRAG is a research and assistive tool. It is **not a substitute for qualified legal advice**. Always consult a licensed legal professional for matters with legal consequences. The accuracy of retrieved judgments depends on the quality and coverage of the underlying corpus.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">
  Built for the Indian legal research community
</div>
