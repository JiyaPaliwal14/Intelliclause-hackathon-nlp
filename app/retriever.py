import asyncio
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
from app.embeddings import embed_chunks_async
from app.vector_store import search_chunks_async
from app.parser import extract_question_intent_async

# Load the cross-encoder model once and reuse it
# This model is small and fast, designed specifically for reranking
rerank_model = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')

async def retrieve_top_chunks_async(query: str, doc_filter: Optional[str] = None, top_k: int = 10) -> List[Dict]:
    """
    Enhanced async retrieval with a larger initial search and a powerful cross-encoder for reranking.
    """
    query_vectors = await embed_chunks_async([query])
    query_vector = query_vectors[0]
    
    if query_vector is None:
        return []
    
    # 1. Increase initial retrieval count (e.g., to 20 or 25)
    # This creates a larger pool of candidates for the reranker.
    initial_k = 25
    
    print(f"🔍 Performing initial vector search with top_k={initial_k}...")
    candidate_chunks = await search_chunks_async(
        query_vector, 
        filters={"document_id": doc_filter} if doc_filter else None, 
        top_k=initial_k
    )
    
    if not candidate_chunks:
        return []

    # 2. Use a Cross-Encoder for powerful reranking
    print(f"🧠 Reranking {len(candidate_chunks)} candidates with Cross-Encoder...")
    
    # The cross-encoder needs pairs of [query, chunk_text]
    model_inputs = [[query, chunk.get("chunk", "")] for chunk in candidate_chunks]
    
    # Calculate relevance scores
    scores = rerank_model.predict(model_inputs)
    
    # Add scores to the chunks
    for chunk, score in zip(candidate_chunks, scores):
        chunk["relevance_score"] = score
        
    # Sort chunks by the new relevance score in descending order
    reranked_chunks = sorted(candidate_chunks, key=lambda x: x["relevance_score"], reverse=True)
    
    # Return the top_k results after reranking
    return reranked_chunks[:top_k]