# app/embeddings.py (SMART - Dynamic rate limiting based on API keys)
"""
Smart async embedding generation with intelligent rate limiting.
"""
import asyncio
from typing import List, Optional
from app.gemini import GeminiClient
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Get API keys and calculate smart batch sizes
api_keys_str = os.getenv("GOOGLE_API_KEY")
if not api_keys_str:
    raise ValueError("GOOGLE_API_KEY environment variable not set or empty.")

# Split and clean API keys
api_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
client = GeminiClient(api_keys)

# Smart configuration based on number of API keys
NUM_API_KEYS = len(api_keys)
print(f"🔑 Embedding service initialized with {NUM_API_KEYS} API keys")

# Dynamic batch sizing: more keys = larger batches, but stay conservative
SMART_BATCH_SIZE = max(3, min(NUM_API_KEYS * 2, 10))  # Between 3-10
INTER_BATCH_DELAY = max(0.5, 3.0 / NUM_API_KEYS)      # Less delay with more keys

print(f"⚙️ Smart batch size: {SMART_BATCH_SIZE}, Inter-batch delay: {INTER_BATCH_DELAY}s")

async def embed_chunks_async(chunks: List[str]) -> List[Optional[List[float]]]:
    """
    Smart async embedding generation with dynamic rate limiting.
    """
    if not chunks:
        print("No chunks provided for embedding")
        return []
    
    print(f"🔄 Generating embeddings for {len(chunks)} chunks...")
    print(f"🔑 Using {NUM_API_KEYS} API keys with batch size {SMART_BATCH_SIZE}")
    
    embeddings = []
    
    # Process in smart batches
    for i in range(0, len(chunks), SMART_BATCH_SIZE):
        batch = chunks[i:i + SMART_BATCH_SIZE]
        batch_num = (i // SMART_BATCH_SIZE) + 1
        total_batches = (len(chunks) + SMART_BATCH_SIZE - 1) // SMART_BATCH_SIZE
        
        print(f"📦 Processing embedding batch {batch_num}/{total_batches} ({len(batch)} chunks)")
        
        batch_start_time = time.time()
        batch_tasks = [embed_single_chunk_async(chunk, idx + i) for idx, chunk in enumerate(batch)]
        batch_embeddings = await asyncio.gather(*batch_tasks, return_exceptions=True)
        batch_time = time.time() - batch_start_time
        
        print(f"⏱️ Batch {batch_num} completed in {batch_time:.1f}s")
        
        # Process results
        successful_embeddings = 0
        for j, embedding in enumerate(batch_embeddings):
            if isinstance(embedding, Exception):
                print(f"❌ Error in batch embedding {i+j}: {embedding}")
                embeddings.append(None)
            elif embedding is None:
                print(f"⚠️ No embedding returned for chunk {i+j}")
                embeddings.append(None)
            else:
                embeddings.append(embedding)
                successful_embeddings += 1
        
        print(f"✅ Batch {batch_num}: {successful_embeddings}/{len(batch)} successful")
        
        # Smart delay between batches (less delay with more API keys)
        if i + SMART_BATCH_SIZE < len(chunks):
            # Adaptive delay: longer if batch was slow (rate limiting detected)
            adaptive_delay = INTER_BATCH_DELAY
            if batch_time > 10:  # If batch took > 10s, likely hit rate limits
                adaptive_delay *= 2
                print(f"⚠️ Slow batch detected, increasing delay to {adaptive_delay}s")
            
            print(f"⏳ Rate limit delay: {adaptive_delay}s...")
            await asyncio.sleep(adaptive_delay)
    
    valid_embeddings = sum(1 for e in embeddings if e is not None)
    success_rate = (valid_embeddings / len(chunks)) * 100 if chunks else 0
    
    print(f"📊 Generated {valid_embeddings}/{len(chunks)} valid embeddings ({success_rate:.1f}% success rate)")
    
    return embeddings

async def embed_single_chunk_async(chunk: str, chunk_idx: int) -> Optional[List[float]]:
    """Embed a single chunk with smart retry logic"""
    if not chunk or len(chunk.strip()) < 5:
        print(f"⚠️ Chunk {chunk_idx} too short: '{chunk[:50]}...'")
        return None
    
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            loop = asyncio.get_event_loop()
            raw_embedding = await loop.run_in_executor(None, client.embed_text, chunk)
            
            # Normalize the embedding to a flat list of floats
            normalized = normalize_embedding(raw_embedding)
            
            if normalized is not None:
                return normalized
            else:
                print(f"❌ Failed to normalize embedding for chunk {chunk_idx}")
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            
            # Check for rate limiting
            if "quota" in error_str or "limit" in error_str or "rate" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"🚦 Rate limit hit for chunk {chunk_idx}, waiting {delay}s...")
                    await asyncio.sleep(delay)
                    continue
            
            # Check for other recoverable errors
            elif "timeout" in error_str or "connection" in error_str:
                if attempt < max_retries - 1:
                    delay = base_delay * (attempt + 1)
                    print(f"🔄 Connection issue for chunk {chunk_idx}, retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
            
            print(f"❌ Error embedding chunk {chunk_idx} (attempt {attempt + 1}): {e}")
            
            if attempt == max_retries - 1:
                return None

def normalize_embedding(raw_embedding) -> Optional[List[float]]:
    """Normalize embedding to a flat list of floats"""
    if raw_embedding is None:
        return None
    
    # Handle list format
    if isinstance(raw_embedding, list):
        if len(raw_embedding) == 0:
            return None
        
        # If it's a nested list, take the first element
        if isinstance(raw_embedding[0], list):
            if len(raw_embedding[0]) > 0:
                embedding = raw_embedding[0]
            else:
                return None
        else:
            embedding = raw_embedding
        
        # Convert to flat list of floats
        try:
            normalized = [float(x) for x in embedding]
            if len(normalized) > 0:
                return normalized
        except (ValueError, TypeError) as e:
            print(f"Error converting embedding to floats: {e}")
            return None
    
    # Handle other formats
    elif hasattr(raw_embedding, 'embedding'):
        return normalize_embedding(raw_embedding.embedding)
    
    elif isinstance(raw_embedding, dict) and 'embedding' in raw_embedding:
        return normalize_embedding(raw_embedding['embedding'])
    
    print(f"Unknown embedding format: {type(raw_embedding)}")
    return None

# Test function with rate limit awareness
async def test_embeddings():
    """Test function with realistic rate limit simulation"""
    test_texts = [
        "This is a test sentence for rate limiting.",
        "Another test sentence for embedding.",
        "Third test to check batch processing.",
        "Fourth test for smart rate limiting.",
        "Fifth test to verify API key rotation."
    ]
    
    print(f"🧪 Testing embeddings with {len(test_texts)} samples...")
    start_time = time.time()
    
    embeddings = await embed_chunks_async(test_texts)
    
    end_time = time.time()
    
    success_count = sum(1 for e in embeddings if e is not None)
    
    print(f"\n📊 Test Results:")
    print(f"   ✅ Success: {success_count}/{len(test_texts)}")
    print(f"   ⏱️ Total time: {end_time - start_time:.2f}s")
    print(f"   📈 Avg time per embedding: {(end_time - start_time) / len(test_texts):.2f}s")
    
    return success_count == len(test_texts)

if __name__ == "__main__":
    # Test embeddings when run directly
    asyncio.run(test_embeddings())