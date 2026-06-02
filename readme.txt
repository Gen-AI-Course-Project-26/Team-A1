## Running LexRAG from Scratch

### Step 1 — Open VS Code
Open VS Code and open your folder:
```
File → Open Folder → D:\lexrag
```

---

### Step 2 — Open Terminal
```
Ctrl + `  (backtick key)
```

---

### Step 3 — Activate Virtual Environment
```bash
d:\lexrag\venv\Scripts\activate
```
You should see `(venv)` appear at the start of the line.

---

### Step 4 — Run the App
```bash
streamlit run app.py
```

That's it. Your browser will open automatically at `http://localhost:8501`

---

### Important — You Do NOT Need to Re-run

You only need to run these once ever — you've already done them:

| Script | Run again? |
|---|---|
| `python src/parser.py` | ❌ No — chunks already saved |
| `python src/embedder.py` | ❌ No — vector store already saved |
| `streamlit run app.py` | ✅ Yes — every time you want to use it |

Your data is permanently saved in:
- `data/processed/` — the 1,799 JSON chunks
- `data/vectorstore/` — the embeddings

---

### Every Time You Want to Use LexRAG

Just these 3 commands:
```bash
d:\lexrag\venv\Scripts\activate
streamlit run app.py
```

Then open `http://localhost:8501` in your browser if it doesn't open automatically.

---

### To Stop the App
Press `Ctrl + C` in the terminal.