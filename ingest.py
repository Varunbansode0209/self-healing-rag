# ingest.py
# ============================================================
# Cleans, chunks, embeds, and stores documents in ChromaDB
# Usage: python ingest.py --domain developer_docs
# ============================================================

import os
import re
import argparse
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────
DOMAIN_PATHS = {
    "developer_docs": "./documents/developer_docs",
    "legal":          "./documents/legal",
    "finance":        "./documents/finance",
    "health":         "./documents/health",
}

CHROMA_BASE  = "./chroma_db"
CHUNK_SIZE   = 1000
CHUNK_OVERLAP = 200

# Files to skip — archive/download pages with no real content
SKIP_PATTERNS = [
    "archives", "downloads", "epub", "pdf", "texinfo",
    "tar_bz2", "zip", "_downloads"
]


# ── STEP 1: CLEAN TEXT ───────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Fixes encoding artifacts and removes residual UI noise.
    Run on every document before chunking.
    """

    # Fix encoding artifacts — the â€™ style corruption
    encoding_fixes = {
        "â€™":  "'",
        "â€œ":  '"',
        "â€\x9d": '"',
        "â€”": "—",  # Correct EM dash character mapping
        "â€“": "–",  # Correct EN dash character mapping
        "Â¶":   "",
        "Â©":   "©",
        "Â":    "",
        "\xa0": " ",      # non-breaking space
        "\u2019": "'",    # right single quotation
        "\u2018": "'",    # left single quotation
        "\u201c": '"',    # left double quotation
        "\u201d": '"',    # right double quotation
        "\u2013": "-",    # en dash
        "\u2014": "--",   # em dash
        "developerâs":   "developer's",
        "Pythonâs":      "Python's",
        "developersâ":   "developers'",
    }
    for broken, fixed in encoding_fixes.items():
        text = text.replace(broken, fixed)

    # Remove residual UI noise patterns
    ui_noise = [
        r"^Table of Contents\s*$",
        r"^topic\s*$",
        r"^index\s*$",
        r"^modules\s*$",
        r"^previous\s*$",
        r"^next\s*$",
        r"^Navigation\s*$",
        r"^Search\s*$",
        r"Copyright.*?Python Software Foundation.*?$",
        r"^This Page\s*$",
        r"^Show Source\s*$",
        r"^Quick search\s*$",
        r"^\s*Â¶\s*$",
    ]
    for pattern in ui_noise:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    # Collapse 3+ blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lines that are just 1-2 characters (leftover noise)
    lines = text.split("\n")
    lines = [l for l in lines if len(l.strip()) > 2 or l.strip() == ""]
    text = "\n".join(lines)

    return text.strip()


def should_skip(filepath: str) -> bool:
    """Returns True for archive/download pages with no real content."""
    fname = filepath.lower()
    return any(pattern in fname for pattern in SKIP_PATTERNS)


def is_too_short(text: str, min_chars: int = 300) -> bool:
    """Skip files with almost no content."""
    # Exclude the SOURCE_URL line from length count
    content = re.sub(r"SOURCE_URL:.*\n", "", text).strip()
    return len(content) < min_chars


# ── STEP 2: LOAD + CLEAN DOCUMENTS ───────────────────────────

def load_documents(domain_path: str, domain: str) -> list[Document]:
    """
    Loads all .txt files, cleans them, filters noise files.
    Returns list of LangChain Document objects.
    """
    docs_path = Path(domain_path)
    all_files = list(docs_path.rglob("*.txt"))

    print(f"\n  Found {len(all_files)} .txt files")

    documents = []
    skipped_archive = 0
    skipped_short   = 0
    loaded           = 0

    for filepath in all_files:
        # Skip archive/download pages
        if should_skip(str(filepath)):
            skipped_archive += 1
            continue

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            # Extract source URL from first line
            source_url = ""
            if raw_text.startswith("SOURCE_URL:"):
                first_line = raw_text.split("\n")[0]
                source_url = first_line.replace("SOURCE_URL:", "").strip()
                raw_text   = "\n".join(raw_text.split("\n")[2:])

            # Clean the text
            clean = clean_text(raw_text)

            # Skip if too short after cleaning
            if is_too_short(clean):
                skipped_short += 1
                continue

            # Create LangChain Document with metadata
            doc = Document(
                page_content=clean,
                metadata={
                    "source":     source_url or str(filepath),
                    "filename":   filepath.name,
                    "domain":     domain,
                    "file_path":  str(filepath),
                }
            )
            documents.append(doc)
            loaded += 1

        except Exception as e:
            print(f"  ERROR loading {filepath.name}: {e}")

    print(f"  Loaded:          {loaded} files")
    print(f"  Skipped archive: {skipped_archive} files")
    print(f"  Skipped short:   {skipped_short} files")

    return documents


# ── STEP 3: CHUNK ────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Splits documents into chunks optimised for technical docs.
    Uses token-aware splitting with code-friendly separators.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # Order matters — try to split on these first
        # Keeps code blocks and paragraphs intact
        separators=[
            "\n\n\n",   # major section break
            "\n\n",     # paragraph break
            "\n",       # line break
            ". ",       # sentence
            ", ",       # clause
            " ",        # word
            "",         # character (last resort)
        ],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"\n  Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"  Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks


# ── STEP 4: EMBED + STORE ────────────────────────────────────

def embed_and_store(chunks: list[Document], domain: str):
    """
    Embeds chunks using OpenAI and stores in ChromaDB.
    Processes in batches to avoid rate limits.
    """
    db_path = str(Path(CHROMA_BASE) / domain)
    os.makedirs(db_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},   # uses your RTX 4050
    encode_kwargs={"normalize_embeddings": True}
    )

    print(f"\n  Embedding {len(chunks)} chunks...")
    print(f"  Storing in: {db_path}")
    print(f"  Estimated cost: ~${len(chunks) * 0.00002:.4f} USD")

    # Process in batches of 100 to avoid rate limits
    BATCH_SIZE = 100
    total_batches = (len(chunks) // BATCH_SIZE) + 1

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1

        print(f"  Batch {batch_num}/{total_batches} — {len(batch)} chunks...")

        if i == 0:
            # First batch — create the vector store
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=db_path,
                collection_name=domain
            )
        else:
            # Subsequent batches — add to existing store
            vectorstore = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings,
                collection_name=domain
            )
            vectorstore.add_documents(batch)

    print(f"\n  Done. {len(chunks)} chunks stored in ChromaDB.")
    return vectorstore


# ── STEP 5: QUICK VERIFICATION ───────────────────────────────

def verify(domain: str):
    """
    Quick sanity check — run 3 test queries after ingestion.
    """
    db_path = os.path.join(CHROMA_BASE, domain)

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=domain
    )

    test_queries = {
        "developer_docs": [
            "How do I use a for loop in Python?",
            "What is a Python decorator?",
            "How does exception handling work?",
        ],
        "legal":   ["What is a contract?", "What is bail?", "PIL procedure"],
        "finance": ["What is SEBI?", "Repo rate RBI", "Mutual fund regulations"],
        "health":  ["Diabetes management", "Vaccine schedule", "Dengue symptoms"],
    }

    queries = test_queries.get(domain, ["Tell me about this domain"])

    print(f"\n  Running verification queries...")
    for q in queries:
        results = vectorstore.similarity_search_with_score(q, k=2)
        if results:
            top_score = 1 - results[0][1]
            print(f"  Q: {q[:50]}")
            print(f"     Score: {top_score:.3f} | Source: {results[0][0].metadata.get('filename','?')[:40]}")
        else:
            print(f"  Q: {q} → NO RESULTS")


# ── MAIN ─────────────────────────────────────────────────────

def ingest(domain: str):
    domain_path = DOMAIN_PATHS.get(domain)
    if not domain_path:
        print(f"Unknown domain. Choose from: {list(DOMAIN_PATHS.keys())}")
        return

    if not os.path.exists(domain_path):
        print(f"Domain folder not found: {domain_path}")
        print(f"Create it and add documents first.")
        return

    print(f"\n{'='*55}")
    print(f" INGESTING DOMAIN: {domain.upper()}")
    print(f"{'='*55}")

    # Step 1 — Load and clean
    print("\n[1/4] Loading and cleaning documents...")
    documents = load_documents(domain_path, domain)
    if not documents:
        print("No documents found. Check your folder path.")
        return

    # Step 2 — Chunk
    print("\n[2/4] Chunking documents...")
    chunks = chunk_documents(documents)

    # Step 3 — Embed and store
    print("\n[3/4] Embedding and storing in ChromaDB...")
    embed_and_store(chunks, domain)

    # Step 4 — Verify
    print("\n[4/4] Verifying retrieval...")
    verify(domain)

    print(f"\n{'='*55}")
    print(f" INGESTION COMPLETE — {domain.upper()}")
    print(f" Next step: python search.py --domain {domain}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--domain",
        required=True,
        choices=list(DOMAIN_PATHS.keys()),
        help="Domain to ingest"
    )
    args = parser.parse_args()
    ingest(args.domain)