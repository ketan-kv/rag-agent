# Limitations

## Ingestion
- pypdf cannot read scanned/image-only PDFs and it fails to read any image that is present in a PDF. In some cases some text may be retrieved (e.g. a cover page or embedded metadata), even if the main body is a scanned image.
- Sites with client-side-rendering retrieve little to no text compared to server-side-rendered sites, since trafilatura only sees the initial HTML and does not execute JavaScript. Note that a site "using React" does not always mean it fails — react.dev itself is server-rendered and extracts cleanly, so the real distinction is CSR vs SSR/SSG, not which framework is used.
- Multi-column layout PDFs like academic papers' text may not be retrieved in reading order, as pypdf extracts based on the PDF's internal structure and not how it looks visually.
- Tables inside PDFs are flattened into plain text during extraction, so row/column structure is lost.

## Chunking
- Fixed-window chunking (kept in the code for comparison purposes) cuts purely by character count and can cut a word in half — confirmed directly during testing, where a chunk ended mid-word ("heightene" instead of "heightened").
- Recursive chunking splits sentences on ". " (period followed by a space), which cannot tell a real sentence ending apart from a numbered heading ("1. Introduction"), an abbreviation, or a decimal number. This was confirmed on a real PDF, where a numbered heading got separated from its title across two chunks. No information is actually lost, since chunk overlap recovers the full heading in the next chunk, but the split point itself is not ideal. A proper fix would need a trained sentence tokenizer such as NLTK's punkt or spaCy, which is out of scope for this from-scratch implementation.
- Overlap between chunks is based on raw characters, trimmed to the nearest word boundary — it recovers boundary text but has no understanding of meaning.
- The chunk size of 800 characters is based on general reasoning about the tradeoffs (too small loses context, too large blurs the embedding), not on any measured retrieval quality. Validating this properly needs the evaluation framework planned for a later phase.

## Embeddings
- all-MiniLM-L6-v2 is smaller and faster than cloud embedding models like OpenAI's or Cohere's, but produces less accurate embeddings for nuanced semantic distinctions.
- The model is trained mainly on general English text, so embedding quality may be weaker for code, non-English text, or highly technical/domain-specific content.

## Vector Storage
- ChromaDB runs locally on a single machine and is not built for multi-user access or massive scale, unlike a managed service like Pinecone or a self-hosted Weaviate cluster.

## Retrieval
- Retrieval always returns k results even if none of them are actually relevant to the query, since there is no similarity threshold filtering yet.
- There is no redundancy control, so near-duplicate chunks can appear together in the top-k results, wasting context space that could have covered more of the topic. This would be fixed by implementing MMR (Maximal Marginal Relevance), which has not been done yet.
- Metadata filtering (e.g. searching within only one source document) is supported by ChromaDB internally, but is not yet exposed as a CLI option.
- There is no re-ranking step. Retrieval is single-stage, based purely on embedding similarity, so it can miss subtler relevance that a cross-encoder re-ranker would catch.
- Retrieval is purely semantic (embedding-based), with no keyword/hybrid search, so exact names, IDs, or acronyms may not match well since embeddings represent meaning rather than exact tokens.

## Generation
- Sources are only listed after the full answer, not attributed to individual claims within it.
- There is no protection against prompt injection — retrieved context and the system's instructions currently share the same prompt with no separation.
- The output does not include any confidence signal indicating how well-grounded the answer actually is in the retrieved context.
- Generation depends on OpenRouter's free tier, which limits Nemotron 3 Ultra to 200 requests per day — not something a production system could rely on.

## Evaluation
- There is no formal evaluation yet — no test set, and no precision/recall/MRR metrics. Improvements so far (like the chunking upgrade) have only been judged by manually inspecting output, not by systematic measurement. This is planned for a later phase.