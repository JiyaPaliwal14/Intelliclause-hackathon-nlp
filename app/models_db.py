# app/models_db.py (Fixed version with correct table creation)
"""
Database models for PostgreSQL with proper relationships and constraints.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

# Create base class
Base = declarative_base()

class Document(Base):
    """Document metadata table"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, index=True)  # UUID
    file_name = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    first_words = Column(String(500), nullable=False)  # First 20 words for deduplication
    content_hash = Column(String(32), unique=True, index=True, nullable=False)  # MD5 hash
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    total_chunks = Column(Integer, default=0, nullable=False)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id='{self.id}', file_name='{self.file_name}', chunks={self.total_chunks})>"

class DocumentChunk(Base):
    """Document chunks table with proper indexing"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    chunk_id = Column(Integer, nullable=False)
    page_number = Column(Integer)
    section_title = Column(String(500))
    doc_type = Column(String(50))
    text = Column(Text, nullable=False)
    
    # Relationship back to document
    document = relationship("Document", back_populates="chunks")
    
    # Create composite index for better query performance
    __table_args__ = (
        Index('idx_document_chunk', 'document_id', 'chunk_id'),
        Index('idx_document_page', 'document_id', 'page_number'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id='{self.document_id}', chunk_id={self.chunk_id})>"