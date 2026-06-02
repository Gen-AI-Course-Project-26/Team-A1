import pdfplumber
import re
import json
import os

DOCUMENTS = {
    "constitution": {
        "path": "data/raw/constitution-of-india.pdf",
        "type": "constitution",
        "strategy": "constitution",
    },
    "ipc": {
        "path": "data/raw/ipc-bare-act.pdf",
        "type": "act",
        "strategy": "act",
    },
    "crpc": {
        "path": "data/raw/crpc-bare-act-1973.pdf",
        "type": "act",
        "strategy": "act",
    },
    "bnss": {
        "path": "data/raw/bharatiya-nagarik-suraksha-sanhita-2023.pdf",
        "type": "act",
        "strategy": "constitution",
    },
}

OUTPUT_DIR = "data/processed"

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += text + "\n"
            if (i + 1) % 50 == 0:
                print(f"    Read {i+1}/{total_pages} pages...")
    print(f"  Extracted {len(full_text):,} characters ({total_pages} pages)")
    return full_text

def clean_text(text):
    text = re.sub(r'(?m)^\d{1,3}(\d{3,}\.)', r'\1', text)
    text = re.sub(r'(?m)^\d+\.\s+(Subs\.|Ins\.|The words|Added|Omitted|Rep\.).*\n', '', text)
    text = re.sub(r'(?m)^\d(Subs\.|Ins\.|Art\.|The ).*\n', '', text)
    text = re.sub(r'(?m)^\s*\d{1,4}\s*$', '', text)
    text = re.sub(r'THE CONSTITUTION OF INDIA \d+', '', text)
    text = re.sub(r'\(Part [A-Z]+.*?\)', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()

def split_act(text, doc_key):
    content_start = find_content_start_act(text)
    content = text[content_start:]
    pattern = re.compile(r'(?m)^(\d{1,3}[A-Z]?\.\s+[A-Z][A-Za-z ,"\'\(\)\[\]\/\-]{5,})')
    matches = list(pattern.finditer(content))
    if not matches:
        print(f"  WARNING: No sections found, using fallback")
        return fallback_chunk(content, doc_key)
    print(f"  Found {len(matches)} candidate sections")
    return build_chunks(matches, content, doc_key, min_length=50)

def find_content_start_act(text):
    m = re.search(r'(?m)^\d+[A-Z]?\.\s+[A-Z][^\n]{3,}(\.--|--|\.—|—)', text)
    return m.start() if m else 0

def split_constitution(text, doc_key):
    lines = text.split('\n')
    normalised_lines = []
    for line in lines:
        m = re.search(r'(\d{1,3}[A-Z]?\.\s+[A-Z(1])', line)
        if m and m.start() > 2:
            normalised_lines.append(line[m.start():])
        else:
            normalised_lines.append(line)
    content = '\n'.join(normalised_lines)
    pattern = re.compile(r'(?m)^(\d{1,3}[A-Z]?\.\s+)')
    matches = list(pattern.finditer(content))
    if not matches:
        print(f"  WARNING: No articles found, using fallback")
        return fallback_chunk(content, doc_key)
    print(f"  Found {len(matches)} candidate articles")
    return build_chunks(matches, content, doc_key, min_length=50)

def build_chunks(matches, content, doc_key, min_length=50):
    sections = []
    seen = set()
    for i, match in enumerate(matches):
        start = match.start()
        end   = matches[i+1].start() if i+1 < len(matches) else len(content)
        body  = content[start:end].strip()
        if len(body) < min_length:
            continue
        header  = match.group(0).strip()
        sec_num = extract_section_number(header)
        if sec_num in seen:
            continue
        seen.add(sec_num)
        sections.append({
            "doc_key":        doc_key,
            "section_id":     f"{doc_key}_{sec_num}",
            "section_number": sec_num,
            "section_title":  body.split('\n')[0][:120],
            "text":           body,
            "char_count":     len(body),
        })
    print(f"  Kept {len(sections)} real sections after filtering")
    return sections

def extract_section_number(header):
    m = re.match(r'^(\d+[A-Z]?)', header.strip())
    return m.group(1) if m else header[:10].replace(" ", "_")

def fallback_chunk(text, doc_key, chunk_size=800, overlap=80):
    words = text.split()
    chunks, i, n = [], 0, 0
    while i < len(words):
        chunk_text = " ".join(words[i:i+chunk_size])
        chunks.append({
            "doc_key":        doc_key,
            "section_id":     f"{doc_key}_chunk_{n}",
            "section_number": str(n),
            "section_title":  f"Chunk {n}",
            "text":           chunk_text,
            "char_count":     len(chunk_text),
        })
        i += chunk_size - overlap
        n += 1
    return chunks

def save_chunks(chunks, doc_key):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{doc_key}_chunks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(chunks)} chunks -> {path}")
    return path

def process_all_documents():
    all_chunks = []
    summary    = {}

    for doc_key, config in DOCUMENTS.items():
        print(f"\n{'='*50}")
        print(f" Processing: {doc_key}  (strategy: {config['strategy']})")

        if not os.path.exists(config["path"]):
            print(f"  ERROR: File not found — {config['path']}")
            continue

        raw   = extract_text_from_pdf(config["path"])
        clean = clean_text(raw)

        if config["strategy"] == "constitution":
            chunks = split_constitution(clean, doc_key)
        else:
            chunks = split_act(clean, doc_key)

        path = save_chunks(chunks, doc_key)
        all_chunks.extend(chunks)
        summary[doc_key] = {
            "total_sections": len(chunks),
            "total_chars":    sum(c["char_count"] for c in chunks),
            "output_file":    path,
        }

        print(f"\n  Sample sections:")
        for c in chunks[:3]:
            print(f"    [{c['section_id']}] {c['section_title'][:70]}")
            print(f"    => {c['text'][:100].strip()}...")
            print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("PARSING COMPLETE")
    print('='*50)
    for doc_key, stats in summary.items():
        print(f"  {doc_key:15s}: {stats['total_sections']:4d} sections,  {stats['total_chars']:>10,} chars")
    print(f"\n  Total chunks : {len(all_chunks)}")
    print(f"  Output folder: {OUTPUT_DIR}/")
    print('='*50)

if __name__ == "__main__":
    process_all_documents()