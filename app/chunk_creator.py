import re
from typing import List, Dict

def advanced_chunking(full_text: str) -> List[Dict[str, str]]:
    """
    Creates smaller, more focused chunks by identifying a hierarchy of headers.
    This helps the retriever find more precise information.
    """
    if not full_text or not full_text.strip():
        return []

    chunks = []
    
    # More comprehensive patterns to capture different heading levels
    patterns = [
        r'\n\s*(\d+\.\s+[A-Za-z ]+)\n',          # 1. In Patient Treatment
        r'\n\s*([A-Z]\.\s+[A-Za-z ]+)\n',          # A. For Cashless Settlement
        r'\n\s*([a-z]\)\s+[A-Za-z ]+)',            # a) Asthma, bronchitis...
        r'\n\s*(ix|iv|v?i{0,3})\.\s+',             # i., ii., v., ix. roman numerals
    ]
    
    # Combine patterns for splitting
    # This pattern looks for a newline, optional whitespace, a header pattern, and then captures the text until the next header
    combined_pattern = '|'.join(f'(?={p})' for p in patterns)
    
    # Split the document by the identified headers
    potential_chunks = re.split(combined_pattern, full_text)

    current_header = "General"
    for part in potential_chunks:
        if not part or not part.strip():
            continue

        # Check if this part is a header
        is_header = False
        for p in patterns:
            match = re.match(p, '\n' + part) # Add newline to match pattern style
            if match:
                current_header = match.group(1).strip()
                # The rest of 'part' after the header is content
                part = re.sub(p, '', '\n' + part, 1).strip()
                is_header = True
                break
        
        # Simple fallback for paragraph splitting within sections
        paragraphs = re.split(r'\n\s*\n', part)
        for para in paragraphs:
            para = para.strip()
            # Clean up source tags and excessive whitespace
            para = re.sub(r'\s+', ' ', para).strip()
            
            if len(para) > 50: # Only add meaningful paragraphs
                chunks.append({
                    "section_number": current_header,
                    "text": para
                })
                
    return chunks

# You would replace your old chunk_pageText with this new logic
def chunk_pageText(full_text: str) -> List[Dict[str, str]]:
    return advanced_chunking(full_text)