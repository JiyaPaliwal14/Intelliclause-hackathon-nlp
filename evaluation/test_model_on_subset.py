import sys
import os
import asyncio
import json
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import torch
import time
import re

# Add the 'utils' directory to the path so we can import our function
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import your custom modules
from app import chunkCreator, embeddings, response_builder, retriever, vector_store
from utils.data_loader import load_cuad_data

class SemanticEvaluator:
    def __init__(self):
        # Initialize sentence transformer model for semantic similarity
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = TfidfVectorizer()
        
    def calculate_semantic_similarity(self, predicted: str, expected: str) -> float:
        """
        Calculate semantic similarity between predicted and expected answers
        Returns a score between 0 and 1
        """
        if not predicted or not expected:
            return 0.0
            
        # Encode sentences to get their embeddings
        embeddings = self.model.encode([predicted, expected], convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_scores.item()

    def evaluate_answers(self, predictions: List[str], references: List[str], threshold: float = 0.7) -> Dict[str, float]:
        """
        Evaluate predictions against references using semantic similarity
        """
        if not predictions or not references:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # For each prediction, check if it semantically matches any reference
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for i, pred in enumerate(predictions):
            max_similarity = 0
            best_match = None
            
            for ref in references:
                similarity = self.calculate_semantic_similarity(pred, ref)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = ref
            
            # Count as correct if similarity exceeds threshold
            if max_similarity >= threshold:
                correct_predictions += 1
                true_positives += 1
            else:
                false_positives += 1
                false_negatives += 1  # Simplified approach
        
        # Calculate metrics
        accuracy = correct_predictions / len(predictions) if predictions else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "threshold": threshold
        }
    
    def hybrid_evaluation(self, predictions, references):
        """
        Comprehensive evaluation with multiple metrics
        """
        results = {
            "substring_accuracy": 0,
            "accuracy": 0,
        }
        
        exact_matches = 0
        total_similarity = 0
        substring_matches = 0
        presence_matches = 0
        
        for pred, ref in zip(predictions, references):
            
            # Substring match
            if ref != "NO_ANSWER" and ref.lower() in pred.lower():
                substring_matches += 1
            
            # Presence detection
            if ((ref != "NO_ANSWER" and ("yes" in pred.lower() or ref.lower() in pred.lower())) or
                (ref == "NO_ANSWER" and ("no" in pred.lower() or "couldn't find" in pred.lower()))):
                presence_matches += 1
            
            # Semantic similarity (for non-NO_ANSWER cases)
            if ref != "NO_ANSWER":
                similarity = self.calculate_semantic_similarity(pred, ref)
                total_similarity += similarity
        
        # results["exact_match"] = exact_matches / len(predictions)
        results["substring_accuracy"] = substring_matches / len(predictions)
        results["accuracy"] = presence_matches / len(predictions)
        # results["semantic_similarity_avg"] = total_similarity / len([r for r in references if r != "NO_ANSWER"])
        
        return results
    
    
class LegalRAGEvaluator:
    """
    Evaluates a RAG model for legal Q&A based on the presence of the correct
    answer in the generated response and calculates standard performance metrics.
    """
    def normalize_text(self, text: str) -> str:
        """
        Converts text to lowercase, removes punctuation, and strips whitespace.
        """
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def evaluate_single(self, predicted_answer: str, true_answers: List[str]) -> str:
        """
        Evaluates a single prediction and classifies it as TP, FP, TN, or FN.
        
        Returns:
            str: The evaluation outcome ('TP', 'FP', 'TN', 'FN').
        """
        normalized_prediction = self.normalize_text(predicted_answer)
        is_no_answer_true = "NO_ANSWER" in true_answers

        # Define keywords that indicate the model correctly found no answer
        negative_keywords = ["no", "not find", "does not contain", "is not mentioned", "no information"]

        if is_no_answer_true:
            # Ground truth is "NO_ANSWER"
            if any(keyword in normalized_prediction for keyword in negative_keywords):
                return "TN"  # True Negative: Correctly identified no answer.
            else:
                return "FP"  # False Positive: Hallucinated an answer.
        else:
            # Ground truth has a specific answer
            if any(self.normalize_text(true_ans) in normalized_prediction for true_ans in true_answers):
                 return "TP" # True Positive: Correctly found the answer.
            else:
                 # False Negative check: did the model say "no" when it should have found an answer?
                 if any(keyword in normalized_prediction for keyword in negative_keywords):
                    return "FN" # False Negative: Incorrectly said no answer.
                 else:
                    # This could also be classified as FN, as it failed to provide the correct answer string.
                    return "FN" # False Negative: Provided a wrong answer.


    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculates final metrics from a list of evaluated results.
        """
        counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        total_time = 0
        
        for res in results:
            outcome = res.get("eval_outcome")
            if outcome in counts:
                counts[outcome] += 1
            total_time += res.get("time_taken", 0)

        tp, fp, tn, fn = counts["TP"], counts["FP"], counts["TN"], counts["FN"]
        
        # --- METRIC CALCULATIONS ---
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_time = total_time / len(results) if results else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "average_inference_time_seconds": avg_time,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "total_questions": len(results)
        }


async def ingest_context(context_text: str, title: str) -> str:
    """
    Ingest context into the vector store for retrieval
    """
    try:
        # 1. Chunk text
        print("🧩 Chunking text...")
        chunks_data = chunkCreator.chunk_pageText(context_text)
        chunk_texts = [c['text'] for c in chunks_data]
        
        print(f"\n✅ Generated {len(chunk_texts)} chunks.\n")
        print("="*60)
        print("PREVIEW OF FIRST 3 CHUNKS")
        print("="*60)

        for i, chunk in enumerate(chunk_texts[:3]): # Only show the first 3
            print(f"\n--- Chunk {i+1} (Length: {len(chunk)} chars) ---")
            # Print a truncated version for clarity, e.g., first 200 characters
            print(chunk[:200] + "..." if len(chunk) > 200 else chunk)
            print("-" * 40)

        
        if not chunk_texts:
            raise Exception("Failed to create chunks from the document.")
        
        # 2. Embed text
        print("🧠 Generating embeddings...")
        chunk_vectors = await embeddings.embed_chunks_async(chunk_texts)
        
        valid_indices = [i for i, v in enumerate(chunk_vectors) if v is not None]
        valid_chunks = [chunk_texts[i] for i in valid_indices]
        valid_vectors = [chunk_vectors[i] for i in valid_indices]
        
        if not valid_vectors:
            raise Exception("Failed to generate any valid embeddings.")
        
        # 3. Insert into vector store
        import uuid
        document_id = str(uuid.uuid4())
        
        metadata_list = [{
            "document_id": title,
            "file_name": title,
            "chunk_id": i,
            "page_number": 1
        } for i in range(len(valid_chunks))]

        print(f"💾 Upserting {len(valid_chunks)} chunks for doc_id: {document_id}")
        await vector_store.upsert_chunks_async(title, valid_chunks, valid_vectors, metadata_list)
        
        return document_id
        
    except Exception as e:
        print(f"Error ingesting context: {e}")
        raise

async def run_model_inference(context: str, question: str, title: str) -> str:
    """
    Run model inference on a question
    """
    try:
        # 1. Retrieve relevant chunks
        print(f"🔍 Retrieving top chunks for query: '{question}'")
        top_chunks_data = await retriever.retrieve_top_chunks_async(
            question, 
            doc_filter=title, 
            top_k=5
        )
        top_chunk_texts = [c['chunk'] for c in top_chunks_data if 'chunk' in c]

        # --- Print retrieved chunks in readable format ---
        print(f"✅ Retrieved {len(top_chunks_data)} chunks:")
        print("=" * 80)

        for i, chunk_data in enumerate(top_chunks_data):
            print(f"\n📄 TOP CHUNK #{i+1} (Relevance Score: {chunk_data.get('relevance_score', 'N/A'):.4f})")
            print(f"📄 Source: {chunk_data.get('document_id', 'N/A')} - Page {chunk_data.get('page_number', 'N/A')}")
            print("-" * 60)
            
            # Display truncated chunk text for readability
            chunk_text = chunk_data.get('chunk', '')
            preview_length = 250  # Adjust this value as needed
            if len(chunk_text) > preview_length:
                print(chunk_text[:preview_length] + "...")
            else:
                print(chunk_text)
            print("-" * 60)

        if not top_chunk_texts:
            answer = "I couldn't find any relevant information in the document to answer that question."
        else:
            # 2. Generate final answer
            answer = await response_builder.build_final_response_async(question, top_chunks_data)
        
        return answer, top_chunk_texts
        
    except Exception as e:
        print(f"Error in model inference: {e}")
        return f"Error: {str(e)}"

async def main():
    """
    Main async function to run the evaluation.
    """
    # Initialize the new evaluator
    evaluator = LegalRAGEvaluator()
    
    # Load the contexts from the test set
    file_path = 'data/test.json'
    contexts_to_test = load_cuad_data(file_path, num_contexts=1) # Adjust num_contexts as needed

    # List to store detailed results
    results = []

    # Iterate through each context
    for context_data in contexts_to_test:
        title = context_data['title']
        context = context_data['context']
        qas = context_data['qas']
        
        # Ingest the context into the vector store
        try:
            document_id = await ingest_context(context, title)
            print(f"✅ Successfully ingested context: {title} with ID: {document_id}")
        except Exception as e:
            print(f"❌ Failed to ingest context: {title}. Error: {e}")
            continue

        print(f"\n--- Testing on Contract: {title} ---")
        
        # Iterate through each Question-Answer pair
        for qa in qas:
            question = qa['question']
            true_answers = [ans['text'] for ans in qa['answers']] if not qa['is_impossible'] else ["NO_ANSWER"]

            # --- Time the inference step ---
            start_time = time.time()
            predicted_answer, top_chunks_text = await run_model_inference(context, question, title)
            end_time = time.time()
            time_taken = end_time - start_time

            # --- Evaluate the single result ---
            eval_outcome = evaluator.evaluate_single(predicted_answer, true_answers)
            
            # Print a summary for this question
            print(f"\nQ: {question}")
            print(f"  - True Answer(s): {true_answers}")
            print(f"  - Model Predicted: {predicted_answer[:150]}...") # Truncate for readability
            print(f"  - Time Taken: {time_taken:.2f}s")
            print(f"  - Evaluation: {eval_outcome}")

            # Store the detailed result for this QA pair
            results.append({
                'context_title': title,
                'question_id': qa['id'],
                'question': question,
                'true_answers': true_answers,
                'predicted_answer': predicted_answer,
                'is_impossible': qa['is_impossible'],
                'time_taken': time_taken,
                'eval_outcome': eval_outcome,
                'chunks': top_chunks_text
            })

    # --- FINAL EVALUATION ---
    print(f"\n\n--- Evaluation Complete ---")
    
    if results:
        # Calculate final metrics across all results
        final_metrics = evaluator.calculate_metrics(results)
        
        print("\n📊 Final Performance Metrics:")
        print("-----------------------------")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():<30}: {value}")
        print("-----------------------------")

        # Save detailed results and final metrics to file
        output_file = "evaluation/evaluation_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "final_metrics": final_metrics,
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Detailed results saved to: {output_file}")
        
    else:
        print("No results were generated to evaluate.")

if __name__ == "__main__":
    asyncio.run(main())