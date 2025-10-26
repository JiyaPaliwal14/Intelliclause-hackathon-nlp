# app/gemini.py (Fixed version)
"""
Enhanced Gemini client with better error handling and debugging.
"""
import os
from itertools import cycle
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

class GeminiClient:
    def __init__(self, keys):
        if not isinstance(keys, list) or not keys:
            raise ValueError("API keys must be provided as a non-empty list.")
        
        # Filter out empty keys
        self.keys = [key.strip() for key in keys if key.strip()]
        if not self.keys:
            raise ValueError("No valid API keys provided.")
        
        self.key_cycle = cycle(self.keys)
        print(f"✅ GeminiClient initialized with {len(self.keys)} API keys")

    def _get_next_key(self):
        return next(self.key_cycle)

    def generate_response(self, prompt, max_retries=3):
        """Generate response with retry logic"""
        for attempt in range(max_retries):
            try:
                key = self._get_next_key()
                genai.configure(api_key=key)
                
                # Use a more recent model
                model = genai.GenerativeModel("gemini-2.5-flash-lite")
                
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 1000,
                }
                
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                if response.text and len(response.text.strip()) > 5:
                    return response.text.strip()
                else:
                    raise Exception("Generated response too short or empty")
                    
            except Exception as e:
                print(f"❌ Response generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return f"I apologize, but I couldn't generate a proper response: {str(e)}"
    
    def embed_text(self, text, max_retries=3):
        """Generate embeddings with better error handling and debugging"""
        if not text or len(text.strip()) < 5:
            print(f"⚠️ Text too short for embedding: '{text[:50]}...'")
            return None
        
        for attempt in range(max_retries):
            try:
                key = self._get_next_key()
                genai.configure(api_key=key)
                
                print(f"🔄 Attempt {attempt + 1}: Generating embedding for text: {text[:100]}...")
                
                # Use the correct embedding model
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text if isinstance(text, str) else str(text),
                    task_type="RETRIEVAL_DOCUMENT"
                )
                
                print(f"📡 Raw response type: {type(response)}")
                
                # Handle different response formats
                if hasattr(response, 'embedding'):
                    embedding = response.embedding
                    print(f"✅ Found embedding in response.embedding, size: {len(embedding)}")
                    return [embedding]  # Return as list for consistency
                elif isinstance(response, dict) and 'embedding' in response:
                    embedding = response['embedding']
                    print(f"✅ Found embedding in response dict, size: {len(embedding)}")
                    return [embedding]
                elif isinstance(response, list) and len(response) > 0:
                    print(f"✅ Response is list, size: {len(response)}")
                    return response
                else:
                    print(f"❌ Unexpected response format: {type(response)}")
                    if hasattr(response, '__dict__'):
                        print(f"Response attributes: {list(response.__dict__.keys())}")
                    continue
                    
            except Exception as e:
                print(f"❌ Embedding attempt {attempt + 1} failed: {e}")
                if "quota" in str(e).lower() or "limit" in str(e).lower():
                    print("⏳ Rate limit hit, waiting before retry...")
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                
                if attempt == max_retries - 1:
                    print(f"❌ All embedding attempts failed for text: {text[:50]}...")
                    return None

# Test the client
if __name__ == "__main__":
    import asyncio
    
    async def test_client():
        api_keys = os.getenv("GOOGLE_API_KEY", "").split(",")
        client = GeminiClient(api_keys)
        
        # Test embedding
        result = client.embed_text("This is a test sentence.")
        print(f"Test embedding result: {result is not None}")
        
        if result:
            print(f"Embedding dimension: {len(result[0]) if isinstance(result, list) and result else 'Unknown'}")
    
    asyncio.run(test_client())