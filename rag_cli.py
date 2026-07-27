"""
Simple RAG CLI
--------------
Ingests PDFs, raw text, and URLs into a local ChromaDB vector store,
then answers questions using an OpenRouter-hosted LLM (default: NVIDIA
Nemotron 3 Ultra, free tier) with retrieved context.

Setup:
    pip install pypdf trafilatura sentence-transformers chromadb openai python-dotenv

    Get a free OpenRouter API key from https://openrouter.ai/keys
    Put it in a .env file in this folder:
        OPENROUTER_API_KEY=your-key-here

Usage:
    python rag_cli.py add-pdf path/to/file.pdf
    python rag_cli.py add-url https://example.com/article
    python rag_cli.py add-text path/to/notes.txt
    python rag_cli.py peek-pdf path/to/file.pdf     (debug: show raw extracted text)
    python rag_cli.py peek-url https://example.com   (debug: show raw extracted text)
    python rag_cli.py ask "What is the main finding of the paper?"
    python rag_cli.py list
    python rag_cli.py reset
"""

import os
import sys
import argparse

import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import trafilatura
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # reads .env in the current directory and loads it into os.environ

DB_PATH = "./rag_db"
COLLECTION_NAME = "docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4


# ---------- Setup ----------

def get_llm_client():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in your .env file first.")
        print("  Get a free key at https://openrouter.ai/keys")
        sys.exit(1)
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)


# ---------- Loaders ----------

def load_pdf(path):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        print("WARNING: No extractable text found. This may be a scanned/image PDF (would need OCR).")
    return text


def load_url(url):
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        print(f"ERROR: Could not fetch {url}")
        sys.exit(1)
    text = trafilatura.extract(downloaded)
    if not text:
        print("WARNING: Could not extract main content from this page.")
    return text or ""


def load_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------- Chunking ----------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------- Debugging ----------

def peek(source_type, path_or_url, preview_chars=1000):
    """Show what raw text was actually extracted, before chunking/embedding.
    Useful for diagnosing scanned PDFs, JS-heavy sites, etc."""
    if source_type == "pdf":
        text = load_pdf(path_or_url)
    elif source_type == "url":
        text = load_url(path_or_url)
    else:
        text = load_text_file(path_or_url)

    char_count = len(text)
    word_count = len(text.split())

    print(f"\n--- PEEK: {path_or_url} ---")
    print(f"Total extracted characters: {char_count}")
    print(f"Total extracted words: {word_count}")

    if char_count == 0:
        print("(Nothing extracted at all.)")
    else:
        print(f"\nFirst {min(preview_chars, char_count)} characters:\n")
        print(text[:preview_chars])
        if char_count > preview_chars:
            print(f"\n... ({char_count - preview_chars} more characters not shown)")
    print("--- END PEEK ---\n")


def add_document(text, source_name):
    if not text.strip():
        print(f"Nothing to add from {source_name} (empty text).")
        return

    chunks = chunk_text(text)
    if not chunks:
        print(f"No chunks produced from {source_name}.")
        return

    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": source_name} for _ in chunks],
    )
    print(f"Added {len(chunks)} chunks from '{source_name}'. "
          f"Total chunks in store: {collection.count()}")


# ---------- Retrieval + Generation ----------

def retrieve(query, k=TOP_K):
    q_emb = embedder.encode([query], show_progress_bar=False).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    return list(zip(docs, metas))


def answer(query):
    if collection.count() == 0:
        print("No documents in the store yet. Add some first with add-pdf / add-url / add-text.")
        return

    retrieved = retrieve(query)
    context_blocks = []
    for doc, meta in retrieved:
        context_blocks.append(f"[Source: {meta.get('source', 'unknown')}]\n{doc}")
    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't
contained in the context, say so clearly instead of guessing.

Context:
{context}

Question: {query}

Answer:"""

    client = get_llm_client()
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer_text = response.choices[0].message.content

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer_text)
    print("\n" + "-" * 60)
    print("Sources used:", ", ".join(sorted(set(m.get("source", "?") for _, m in retrieved))))


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Simple RAG CLI (ChromaDB + OpenRouter)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pdf = sub.add_parser("add-pdf", help="Ingest a PDF file")
    p_pdf.add_argument("path")

    p_url = sub.add_parser("add-url", help="Ingest a webpage by URL")
    p_url.add_argument("url")

    p_text = sub.add_parser("add-text", help="Ingest a plain text file")
    p_text.add_argument("path")

    p_ask = sub.add_parser("ask", help="Ask a question against the store")
    p_ask.add_argument("query")

    p_peek_pdf = sub.add_parser("peek-pdf", help="Show raw extracted text from a PDF (debug)")
    p_peek_pdf.add_argument("path")

    p_peek_url = sub.add_parser("peek-url", help="Show raw extracted text from a URL (debug)")
    p_peek_url.add_argument("url")

    sub.add_parser("list", help="List ingested sources")
    sub.add_parser("reset", help="Delete the vector store and start fresh")

    args = parser.parse_args()

    if args.command == "add-pdf":
        text = load_pdf(args.path)
        add_document(text, os.path.basename(args.path))

    elif args.command == "add-url":
        text = load_url(args.url)
        add_document(text, args.url)

    elif args.command == "add-text":
        text = load_text_file(args.path)
        add_document(text, os.path.basename(args.path))

    elif args.command == "ask":
        answer(args.query)

    elif args.command == "peek-pdf":
        peek("pdf", args.path)

    elif args.command == "peek-url":
        peek("url", args.url)

    elif args.command == "list":
        if collection.count() == 0:
            print("Store is empty.")
        else:
            all_docs = collection.get()
            sources = sorted(set(m.get("source", "?") for m in all_docs["metadatas"]))
            print(f"{collection.count()} chunks from {len(sources)} source(s):")
            for s in sources:
                print(f"  - {s}")

    elif args.command == "reset":
        client.delete_collection(COLLECTION_NAME)
        print("Store reset.")


if __name__ == "__main__":
    main()