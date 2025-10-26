import asyncio
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app import (
    pdfToText,
    chunkCreator,
    embeddings,
    vector_store,
    retriever,
    response_builder
)
# Note: No need for database.py or models_db.py imports here for now,
# as vector_store handles the DB interaction.

# --- Pydantic Models for API ---

class ProcessResponse(BaseModel):
    document_id: str
    file_name: str
    message: str

class QueryRequest(BaseModel):
    document_id: str
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

# --- FastAPI App ---
app = FastAPI(
    title="Legal Document RAG API",
    description="API to process and query legal documents from an uploaded file.",
)

@app.on_event("startup")
async def startup_event():
    """Ensure Qdrant collection exists on startup."""
    print("Application startup: Ensuring vector collection exists...")
    try:
        await vector_store.ensure_collection_correct_async()
        print("✅ Vector collection is ready.")
    except Exception as e:
        print(f"❌ Failed to initialize vector collection: {e}")
        raise RuntimeError("Could not connect to or configure Qdrant.") from e

@app.post("/process-and-query/", response_model=ProcessResponse)
async def process_document(
    pdf_file: UploadFile = File(...),
    questions: List[str] = Form(...) # To handle multipart form
):
    """
    Main endpoint to receive a PDF, process it, store it, and answer initial questions.
    This is now primarily for INGESTING the document.
    """
    try:
        # 1. Read uploaded file content
        print(f"📄 Receiving file: {pdf_file.filename}")
        pdf_content = await pdf_file.read()
        
        # 2. Extract text from PDF content
        print("Extracting text...")
        full_text = ""
        # The generator works with URLs, so we need to adapt.
        # Let's create a temporary text extraction logic for BytesIO.
        from io import BytesIO
        import pdfplumber
        
        page_texts = []
        with pdfplumber.open(BytesIO(pdf_content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    page_texts.append(page_text)
        full_text = "\n\n".join(page_texts)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        # 3. Chunk the text
        print("🧩 Chunking text...")
        chunks_data = chunkCreator.chunk_pageText(full_text)
        chunk_texts = [c['text'] for c in chunks_data]

        if not chunk_texts:
            raise HTTPException(status_code=400, detail="Failed to create chunks from the document.")

        # 4. Generate embeddings
        print("🧠 Generating embeddings...")
        chunk_vectors = await embeddings.embed_chunks_async(chunk_texts)
        
        valid_indices = [i for i, v in enumerate(chunk_vectors) if v is not None]
        valid_chunks = [chunk_texts[i] for i in valid_indices]
        valid_vectors = [chunk_vectors[i] for i in valid_indices]
        
        if not valid_vectors:
            raise HTTPException(status_code=500, detail="Failed to generate any valid embeddings.")

        # 5. Upsert to Vector Store
        import uuid
        document_id = str(uuid.uuid4())
        
        metadata_list = [{
            "document_id": document_id,
            "file_name": pdf_file.filename,
            "chunk_id": i,
            "page_number": 1, # Simplified for example
        } for i in range(len(valid_chunks))]

        print(f"💾 Upserting {len(valid_chunks)} chunks for doc_id: {document_id}")
        await vector_store.upsert_chunks_async(document_id, valid_chunks, valid_vectors, metadata_list)
        
        return ProcessResponse(
            document_id=document_id,
            file_name=pdf_file.filename,
            message=f"Successfully processed and indexed '{pdf_file.filename}'. You can now use the document_id to ask questions."
        )

    except Exception as e:
        print(f"❌ An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Endpoint to ask a question about a previously processed document.
    """
    try:
        print(f"❓ Received query for doc_id: {request.document_id}")
        # 1. Retrieve relevant chunks
        top_chunks_data = await retriever.retrieve_top_chunks_async(
            request.question, 
            doc_filter=request.document_id, 
            top_k=5
        )
        top_chunk_texts = [c['chunk'] for c in top_chunks_data if 'chunk' in c]

        if not top_chunk_texts:
            answer = "I couldn't find any relevant information in the document to answer that question."
        else:
            # 2. Generate final answer
            answer = await response_builder.build_final_response_async(request.question, top_chunks_data)

        return QueryResponse(question=request.question, answer=answer)

    except Exception as e:
        print(f"❌ An error occurred during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))