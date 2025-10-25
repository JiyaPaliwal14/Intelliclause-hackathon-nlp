"""
Async PDF text extraction from remote URLs.
"""
import aiohttp
import pdfplumber
from io import BytesIO

async def extract_text_generator_async(url):
    """
    Async generator that yields plain text page by page from a PDF at the given URL.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                response.raise_for_status()
                content = await response.read()
                
        pdf_file = BytesIO(content)
        
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    yield page_text
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")