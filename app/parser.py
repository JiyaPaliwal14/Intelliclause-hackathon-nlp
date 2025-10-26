"""
Enhanced query parsing and intent extraction.
"""
import re
from typing import Dict, List
import asyncio
from app.gemini import GeminiClient
import os

client = GeminiClient(os.getenv("GOOGLE_API_KEY", "").split(","))

async def extract_question_intent_async(question: str) -> Dict[str, any]:
    """
    Enhanced async question intent extraction using LLM.
    """
    intent_prompt = f"""
    Analyze this insurance/policy question and extract the following information:
    
    Question: "{question}"
    
    Extract:
    1. Main topic/subject (e.g., "premium payment", "waiting period", "coverage")
    2. Question type (e.g., "what", "when", "how", "does", "is")
    3. Key entities (e.g., specific medical procedures, policy features)
    4. Urgency level (low/medium/high)
    5. Expected answer type (yes/no, duration, amount, definition, list)
    
    Return as JSON format:
    {{
        "main_topic": "topic here",
        "question_type": "type here", 
        "key_entities": ["entity1", "entity2"],
        "urgency": "level",
        "answer_type": "type"
    }}
    """
    
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(None, client.generate_response, intent_prompt)
        # Parse JSON response (simplified for demo)
        return {
            "main_topic": extract_main_topic(question),
            "question_type": extract_question_type(question),
            "key_entities": extract_key_entities(question),
            "urgency": "medium",
            "answer_type": "definition"
        }
    except Exception as e:
        print(f"Error extracting intent: {e}")
        return {
            "main_topic": question.lower()[:50],
            "question_type": "general",
            "key_entities": [],
            "urgency": "medium",
            "answer_type": "general"
        }

def extract_main_topic(question: str) -> str:
    """Extract main topic using keyword matching"""
    topics = {
        "premium": ["premium", "payment", "pay"],
        "waiting_period": ["waiting", "period", "wait"],
        "coverage": ["cover", "coverage", "covered", "benefit"],
        "exclusion": ["exclude", "exclusion", "not covered"],
        "claim": ["claim", "reimbursement"],
        "maternity": ["maternity", "pregnancy", "childbirth"],
        "surgery": ["surgery", "operation", "procedure"],
        "hospital": ["hospital", "hospitalization"],
        "pre_existing": ["pre-existing", "PED"],
    }
    
    question_lower = question.lower()
    for topic, keywords in topics.items():
        if any(keyword in question_lower for keyword in keywords):
            return topic
    
    return "general"

def extract_question_type(question: str) -> str:
    """Extract question type from question words"""
    question_lower = question.lower().strip()
    
    if question_lower.startswith(("what", "which")):
        return "what"
    elif question_lower.startswith(("when", "how long")):
        return "when"
    elif question_lower.startswith("how"):
        return "how"
    elif question_lower.startswith(("does", "is", "are", "can")):
        return "yes_no"
    elif question_lower.startswith("where"):
        return "where"
    elif question_lower.startswith("why"):
        return "why"
    else:
        return "general"

def extract_key_entities(question: str) -> List[str]:
    """Extract key entities using pattern matching"""
    entities = []
    
    # Medical procedures
    medical_terms = re.findall(r'\b(?:surgery|operation|procedure|treatment|therapy|scan|test)\b', question, re.IGNORECASE)
    entities.extend(medical_terms)
    
    # Policy features
    policy_terms = re.findall(r'\b(?:premium|deductible|copay|coverage|benefit|claim|exclusion)\b', question, re.IGNORECASE)
    entities.extend(policy_terms)
    
    # Time periods
    time_terms = re.findall(r'\b(?:\d+\s*(?:days?|months?|years?)|waiting period|grace period)\b', question, re.IGNORECASE)
    entities.extend(time_terms)
    
    # Amounts
    amounts = re.findall(r'\b\$?\d+(?:,\d{3})*(?:\.\d{2})?\b', question)
    entities.extend(amounts)
    
    return list(set(entities))  # Remove duplicates
