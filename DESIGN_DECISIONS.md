# Design Decisions

## Ingestion
- pypdf is used for Ingestion as it performs text extraction from pdf's. It understands text, metadata, annotations and embedded objects in pdf's. The limitations of pypdf is it cannot read text inside images or describe the images moreover pypdf cannot understand scanned images (It does not do OCR).
- trafilatura is utilized for web scraping, it retrieves only the main textual content in the page and removes any advertisements, navigation menus, sidebars and other clutter. BeautifulSoup is another package that is used for URL's but we have to manually determine which parts to ignore which can be a tedious process. The limitations of trafilatura is it cannot read JavaScript rendered websites and it cannot do OCR.

## Chunking
- Chunk size of 800 chars is chosen for the rag-agent because firstly, if the chunk size is too small (e.g. 100 chars) lose context, a topic might span several chunks and during retrieval all chunks may not be retrieved. Secondly, if the chunk size is too large (e.g. 3000 chars) a chunk covers multiple topics and it's vector  won't represent a single topic instead "an average meaning" of all the topics the chunk covers.
- Overlap of 100 chars is set as without overlap a sentence that falls at the boundary of 2 chunks would lose the meaning of both halves.
- Fixed-size chunking is simple to implement since it just counts characters, but this simplicity is exactly its weakness — it ignores sentence and paragraph boundaries entirely, which is what causes the mid-sentence cutting problem noted below.

## Embeddings
- all-MiniLM-L6-v2 does the job of embedding which is turning the sentences and paragraphs into a numerical vector that represents it's meaning. It uses a sentence-transformer model.
- MiniLM is small and runs locally. Cloud embedding models from OpenAI and Google exists which are bigger and better but are heavier and runs slowly and MiniLM makes up with speed and zero cost while it loses in retrieval accuracy to the bigger models.

## Vector Storage
- ChromaDB is the vector storage for this project as it runs locally and free to use with no server costs or API calls.
- ChromaDB is not built for multi-user or for massive scale operations as opposed to Pinecone (cloud-scale, managed) or a self hosted Weaviate cluster. But for out project ChromaDB satisfies the needs.

## Retrieval
- Top-k is set to 4 for retrieval, if k is too small (1-2) and the answer spans multiple chunks the LLM does not have enough information to work with and if k is too large (15-20) the information is diluted and "lost in the middle". 
- One of the limitations is it always returns k results even if nothing is relevant. Another limiations is linked with redundancy where the top-k can return near-duplicate chunks that all say almost the same thing. 

## Generation
- For Generation this project uses Nemotron 3 Ultra model via the OpenRouter API Key. This model has been selected is it can perform complex reasoning tasks and has a huge context window (Upto 1M tokens). This model is free and can be used via the OpenRouter API. Alternatively, Gemini also provides free to use models through the Google AI Studio.
- Open source models like Llama may offer better control but they hallucinate more and have lesser context windows.

## Known Limitations 
- Re-ranking is not performed by the RAG Agent which means that chunks which basically say the same thing may get retrieved and the whole information about the topic is not retrieved. So the same information chunks may appear in top-k retrieval.
- Even if the information queried is not present, top-k results are generated which do not have any relevant content to the query. Similarity threshold filtering can be implemented as a solution for this where the chunks are only retrieved only when it exceeds a certian similarity to the query.
- pypdf cannot perform OCR hence text in images, handwritten text and information in images are not recognized.
- Our fixed-window chunking splits purely by character count (800 chars with 100-char overlap), which means sentences and paragraphs can be cut mid-way across chunk boundaries. This produces incomplete semantic units that have weaker embeddings and can waste context space when both halves are retrieved as separate chunks.
