
# app/response_builder.py (Simple working version)
"""
Simple async response building that works.
"""
import asyncio
from typing import List
from app.gemini import GeminiClient
import os

# Initialize client
api_keys = os.getenv("GOOGLE_API_KEY", "").split(",")
client = GeminiClient([key.strip() for key in api_keys if key.strip()])

async def build_final_response_async(question: str, top_chunks: List[str]) -> str:
    """Generate response from question and relevant chunks"""
    if not top_chunks:
        return "I couldn't find relevant information to answer your question from the given document."
    
    try:
        # Create context from top chunks
        context = "\n\n".join([f"Section {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks[:5])])
        
        # Create prompt
        prompt = f"""Based on the following info, provide an accurate answer to the question. The sections may not be exactly relevant to the question, but use them to form your answer.

RELEVANT SECTIONS:
{context}

QUESTION: {question}

Please provide a direct answer based solely on the information in the relevant sections above. If specific conditions, amounts, time, or values are mentioned, include them in your response.
Don't return "/n" "*" or any other symbol unless needed. NO BOLD TEXT / HEADING IS NEEDED. Just give a small neat reply with all required details.
EXPLICITLY MENTION YES OR NO IF THE QUESTION REQUIRES A YES / NO RESPONSE. HAVE AT LEAST 1 STATEMENT FOR THE ANSWER.

Some sample answers are: (TRY TO MATCH THIS FORMAT WHILE ANSWERING)
        "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
        "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
        "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
        "The policy has a specific waiting period of two (2) years for cataract surgery.",
        "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
        "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
        "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
        "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
        "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
        "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."

ANSWER:"""

        # Generate response
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, client.generate_response, prompt)
        
        if response and len(response.strip()) > 10:
            return response.strip()
        else:
            return "I found relevant information but couldn't generate a complete answer. Please try rephrasing your question."
            
    except Exception as e:
        print(f"❌ Error in response generation: {e}")
        return "I apologize, but I encountered an error while generating the response."
