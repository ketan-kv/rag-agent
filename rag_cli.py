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
import re
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

def normalize_pdf_text(text):
    """PDF extraction preserves the *visual* line breaks from the page layout
    (every time a line wrapped at the margin), which almost never line up
    with sentence or paragraph boundaries. Left as-is, these fake newlines
    confuse the recursive chunker into treating arbitrary word-wrap points
    as legitimate split points. Here we:
      1. Treat any run of 2+ newlines as a genuine paragraph break, and
         standardize it to exactly "\\n\\n".
      2. Collapse any remaining single "\\n" (a mid-paragraph line wrap)
         into a plain space, so sentences flow together as one line again
         and their real punctuation (periods, etc.) becomes visible to the
         chunker.
    """
    text = re.sub(r"\n\s*\n+", "\n\n", text)          # standardize paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)        # collapse mid-paragraph line wraps
    text = re.sub(r"[ \t]+", " ", text)                 # collapse repeated spaces/tabs
    return text


def load_pdf(path):
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        print("WARNING: No extractable text found. This may be a scanned/image PDF (would need OCR).")
        return text
    return normalize_pdf_text(text)


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

CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "recursive")  # "fixed" or "recursive"

# Separators tried in priority order: paragraph breaks first, then lines,
# then sentence endings, then plain spaces, then (as a last resort) raw characters.
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def fixed_window_chunk(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Naive baseline: split purely by character count, ignoring sentence/paragraph
    boundaries entirely. Kept around so we can compare against recursive splitting."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def _split_into_pieces(text, chunk_size, separators):
    """Recursively break text down into small boundary-respecting pieces —
    no merging, no overlap yet. Just keeps splitting on finer separators
    until every piece is chunk_size or smaller. Deliberately does NOT strip
    text here, since a trailing separator (e.g. "\\n\\n") often carries the
    paragraph boundary that the outer merge step needs to preserve."""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    # Pick the first separator (in priority order) that's actually present.
    # "" (empty string) always "matches" and is our last-resort character split.
    separator = ""
    remaining_separators = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    if separator == "":
        # Base case: no structure left to respect, split by raw character count.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    raw_pieces = text.split(separator)
    # Re-attach the separator to each piece (except the last) so we don't
    # lose the punctuation/whitespace that made this a meaningful boundary.
    pieces = [p + separator if idx < len(raw_pieces) - 1 else p
              for idx, p in enumerate(raw_pieces)]

    result = []
    for piece in pieces:
        if not piece.strip():  # skip genuinely empty/whitespace-only pieces
            continue
        if len(piece) > chunk_size:
            result.extend(_split_into_pieces(piece, chunk_size, remaining_separators))
        else:
            result.append(piece)  # keep piece as-is, including its trailing separator
    return result


def _merge_with_overlap(pieces, chunk_size, overlap):
    """Single merge pass (never called recursively) that glues small pieces
    together up to chunk_size, adding overlap by carrying the tail of the
    previous chunk into the start of the next one. Doing this exactly once,
    after all splitting is done, avoids double-counting overlap."""
    chunks = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if overlap > 0 and chunks:
                tail = chunks[-1][-overlap:]
                # A raw character slice can start mid-word (e.g. "i -structured"
                # from "semi -structured"). Trim up to the first space so the
                # overlap begins on a whole word instead.
                space_idx = tail.find(" ")
                if space_idx != -1:
                    tail = tail[space_idx + 1:]
                current = tail + piece
            else:
                current = piece
    if current:
        chunks.append(current)
    return chunks


def recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Structure-aware splitting: respects paragraph/sentence/word boundaries
    before ever falling back to a raw character cut. Fixes the mid-sentence
    cutting problem that fixed-window chunking has."""
    pieces = _split_into_pieces(text.strip(), chunk_size, RECURSIVE_SEPARATORS)
    chunks = _merge_with_overlap(pieces, chunk_size, overlap)
    return [c.strip() for c in chunks if c.strip()]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, strategy=None):
    strategy = strategy or CHUNK_STRATEGY
    if strategy == "fixed":
        return fixed_window_chunk(text, chunk_size, overlap)
    return recursive_chunk(text, chunk_size, overlap)


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


def compare_chunks(source_type, path_or_url, num_chunks_to_show=3):
    """Show fixed-window vs recursive chunking output side by side, so the
    mid-sentence-cutting difference can be seen directly, not just asserted."""
    if source_type == "pdf":
        text = load_pdf(path_or_url)
    else:
        text = load_url(path_or_url)

    if not text.strip():
        print("Nothing extracted from this source, nothing to compare.")
        return

    fixed_chunks = fixed_window_chunk(text)
    recursive_chunks = recursive_chunk(text)

    print(f"\n=== COMPARISON: {path_or_url} ===")
    print(f"Fixed-window chunks:  {len(fixed_chunks)}")
    print(f"Recursive chunks:     {len(recursive_chunks)}")

    print(f"\n--- First {num_chunks_to_show} FIXED-WINDOW chunks ---")
    for i, c in enumerate(fixed_chunks[:num_chunks_to_show]):
        print(f"\n[Chunk {i}] ({len(c)} chars)")
        print(c)

    print(f"\n--- First {num_chunks_to_show} RECURSIVE chunks ---")
    for i, c in enumerate(recursive_chunks[:num_chunks_to_show]):
        print(f"\n[Chunk {i}] ({len(c)} chars)")
        print(c)
    print("\n=== END COMPARISON ===\n")


def add_document(text, source_name, strategy=None):
    if not text.strip():
        print(f"Nothing to add from {source_name} (empty text).")
        return

    chunks = chunk_text(text, strategy=strategy)
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
    p_pdf.add_argument("--strategy", choices=["fixed", "recursive"], default=None,
                        help="Chunking strategy to use (default: recursive)")

    p_url = sub.add_parser("add-url", help="Ingest a webpage by URL")
    p_url.add_argument("url")
    p_url.add_argument("--strategy", choices=["fixed", "recursive"], default=None,
                        help="Chunking strategy to use (default: recursive)")

    p_text = sub.add_parser("add-text", help="Ingest a plain text file")
    p_text.add_argument("path")
    p_text.add_argument("--strategy", choices=["fixed", "recursive"], default=None,
                         help="Chunking strategy to use (default: recursive)")

    p_ask = sub.add_parser("ask", help="Ask a question against the store")
    p_ask.add_argument("query")

    p_peek_pdf = sub.add_parser("peek-pdf", help="Show raw extracted text from a PDF (debug)")
    p_peek_pdf.add_argument("path")

    p_peek_url = sub.add_parser("peek-url", help="Show raw extracted text from a URL (debug)")
    p_peek_url.add_argument("url")

    p_compare_pdf = sub.add_parser("compare-chunks-pdf",
                                    help="Compare fixed vs recursive chunking on a PDF (debug)")
    p_compare_pdf.add_argument("path")

    p_compare_url = sub.add_parser("compare-chunks-url",
                                    help="Compare fixed vs recursive chunking on a URL (debug)")
    p_compare_url.add_argument("url")

    sub.add_parser("list", help="List ingested sources")
    sub.add_parser("reset", help="Delete the vector store and start fresh")

    args = parser.parse_args()

    if args.command == "add-pdf":
        text = load_pdf(args.path)
        add_document(text, os.path.basename(args.path), strategy=args.strategy)

    elif args.command == "add-url":
        text = load_url(args.url)
        add_document(text, args.url, strategy=args.strategy)

    elif args.command == "add-text":
        text = load_text_file(args.path)
        add_document(text, os.path.basename(args.path), strategy=args.strategy)

    elif args.command == "ask":
        answer(args.query)

    elif args.command == "peek-pdf":
        peek("pdf", args.path)

    elif args.command == "peek-url":
        peek("url", args.url)

    elif args.command == "compare-chunks-pdf":
        compare_chunks("pdf", args.path)

    elif args.command == "compare-chunks-url":
        compare_chunks("url", args.url)

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