import os
import re
import json
import nltk
import faiss
import numpy as np
from news import articles_dump
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

def chunk_by_sentence(text: str, max_words: int = 150) -> list:
    """
    Split the input text into chunks without exceeding the specified max word count per chunk
    Sentence boundaries are preserved to maintain semantic structure
    """
    sentences = sent_tokenize(text) # Splits text into sentences using punctuation (e.g., '.', '?', '!')
    chunks, current = [], []

    for sentence in sentences:
        tentative = current + [sentence]
        if len(" ".join(tentative).split()) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = [sentence] # Start a new chunk w/ the current sentence
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))

    return chunks

def embedding(_q):
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:   
        nltk.download('punkt_tab')

    # Dump articles and return the saved file path
    path = articles_dump(_q) 

    articles = []
    with open(path, "r", encoding = "utf-8") as f:
        for line in f:
            data = json.loads(line)
            article = data["text"]

            article = article.strip()
            article = article.replace("\\n", "\n") # Convert escaped newlines to real line breaks
            article = re.sub(r'[ \t]+', ' ', article) # Remove spaces and tabs
            articles.append(article)


    results = []
    for article in articles:
        chunks = chunk_by_sentence(article)
        results.append(chunks)

    # Embedding model only accepts List[str]
    flattened_chunks = [chunk for article_chunks in results for chunk in article_chunks]

    save_dir = os.path.join(".", "vector_store")
    os.makedirs(save_dir, exist_ok = True)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    base_path = os.path.splitext(path)[0]
    file_name = os.path.basename(base_path)
    faiss_path = os.path.join(save_dir, f"{file_name}.faiss")

    if not os.path.exists(faiss_path):
        embeddings = model.encode(
            flattened_chunks,
            batch_size = 32,
            show_progress_bar = True
        )
        embedding_matrix = np.array(embeddings).astype("float32")
        if embedding_matrix.shape[0] == 0:
            raise ValueError(f"Embedding matrix is empty for query '{_q}'. Check your data pipeline.")
        dimension = embedding_matrix.shape[1]  # Usually 384 for MiniLM models
        # [0]: #vec
        # [1]: dim of the vec
        
        index = faiss.IndexFlatL2(dimension) # Euclidean distance (smaller is better); or use cosine similarity (larger is better) if needed
        # < 10,000: Use a flat index (IndexFlatL2) for exact results — no need for approximate methods.
        # 10K–1M+: Use approximate search to speed things up — like IVF or HNSW.
        # 100M+: Consider product quantization (PQ) to save memory.

        index.add(embedding_matrix)
        faiss.write_index(index, faiss_path)
    else:
        # Load existing FAISS index (need better logic to decide when to regenerate)
        index = faiss.read_index(faiss_path)

    return model, index, flattened_chunks