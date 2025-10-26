import sys
import os
import asyncio
import json
import time
import re
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from openai import api_key

load_dotenv()

# Add the 'utils' directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import your data loader
from utils.data_loader import load_cuad_data

class GenericChatbotEvaluator:
    """
    Evaluates a generic chatbot for legal Q&A using the same metrics as the RAG evaluator.
    """
    def __init__(self, model_name: str = "gemma2-9b-it"):
        
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        
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
        negative_keywords = ["no", "not find", "does not contain", "is not mentioned", "couldn't"]

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
            # "true_positives": tp,
            # "false_positives": fp,
            # "true_negatives": tn,
            # "false_negatives": fn,
            # "total_questions": len(results)
        }

    def create_prompt(self, context: str, question: str) -> str:
        """
        Creates a prompt for the generic chatbot with the context and question.
        """
        prompt = f"""
        You are a legal assistant analyzing a contract. Based on the contract text provided below, 
        answer the question that follows. If the answer cannot be found in the contract text, 
        respond with "I couldn't find the answer in the contract."
        
        CONTRACT TEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """
        return prompt

    async def get_chatbot_response(self, context: str, question: str) -> str:
        """
        Gets a response from the generic chatbot.
        """
        try:
            prompt = self.create_prompt(context, question)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful legal assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic responses
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting chatbot response: {e}")
            return f"Error: {str(e)}"

async def evaluate_generic_chatbot():
    """
    Main async function to evaluate the generic chatbot.
    """
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    
    # Check if API key is set
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it using: export GROQ_API_KEY='your-api-key-here'")
        return
    
    print(f"Using API key: {api_key[:10]}...")  # Show first 10 chars for verification
    
    try:
        evaluator = GenericChatbotEvaluator(model_name="gemma2-9b-it")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Load the contexts from the test set
    file_path = 'data/test.json'
    try:
        contexts_to_test = load_cuad_data(file_path, num_contexts=1)  # Adjust num_contexts as needed
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return

    # List to store detailed results
    results = []

    # Iterate through each context
    for context_data in contexts_to_test:
        title = context_data['title']
        context = context_data['context']
        qas = context_data['qas']
        
        print(f"\n--- Testing Generic Chatbot on Contract: {title} ---")
        
        # Iterate through each Question-Answer pair
        for qa in qas:
            question = qa['question']
            true_answers = [ans['text'] for ans in qa['answers']] if not qa['is_impossible'] else ["NO_ANSWER"]

            # --- Time the inference step ---
            start_time = time.time()
            predicted_answer = await evaluator.get_chatbot_response(context, question)
            end_time = time.time()
            time_taken = end_time - start_time

            # --- Evaluate the single result ---
            eval_outcome = evaluator.evaluate_single(predicted_answer, true_answers)
            
            # Print a summary for this question
            print(f"\nQ: {question}")
            print(f"  - True Answer(s): {true_answers}")
            print(f"  - Chatbot Predicted: {predicted_answer[:150]}...")  # Truncate for readability
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
                'eval_outcome': eval_outcome
            })

    # --- FINAL EVALUATION ---
    print(f"\n\n--- Generic Chatbot Evaluation Complete ---")
    
    if results:
        # Calculate final metrics across all results
        final_metrics = evaluator.calculate_metrics(results)
        
        print("\n📊 Generic Chatbot Performance Metrics:")
        print("---------------------------------------")
        for metric, value in final_metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():<30}: {value}")
        print("---------------------------------------")

        # Save detailed results and final metrics to file
        output_file = "evaluation/generic_chatbot_results.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "final_metrics": final_metrics,
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Generic chatbot results saved to: {output_file}")
        
    else:
        print("No results were generated to evaluate.")

def compare_results(rag_file: str, chatbot_file: str):
    """
    Compare the results of the RAG model and the generic chatbot.
    """
    # Load RAG results
    with open(rag_file, 'r', encoding='utf-8') as f:
        rag_data = json.load(f)
    
    # Load chatbot results
    with open(chatbot_file, 'r', encoding='utf-8') as f:
        chatbot_data = json.load(f)
    
    rag_metrics = rag_data['final_metrics']
    chatbot_metrics = chatbot_data['final_metrics']
    
    print("\n" + "="*60)
    print("COMPARISON: RAG MODEL vs GENERIC CHATBOT")
    print("="*60)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'average_inference_time_seconds']
    
    for metric in metrics_to_compare:
        rag_value = rag_metrics.get(metric, 0)
        chatbot_value = chatbot_metrics.get(metric, 0)
        
        if isinstance(rag_value, float):
            print(f"{metric.replace('_', ' ').title():<35}: RAG: {rag_value:.4f} | Chatbot: {chatbot_value:.4f}")
            
            # Calculate improvement
            if rag_value > chatbot_value:
                improvement = ((rag_value - chatbot_value) / chatbot_value) * 100
                print(f"  → RAG improvement: +{improvement:.2f}%")
            elif chatbot_value > rag_value:
                decline = ((chatbot_value - rag_value) / rag_value) * 100
                print(f"  → RAG decline: -{decline:.2f}%")
            else:
                print(f"  → No difference")
        else:
            print(f"{metric.replace('_', ' ').title():<35}: RAG: {rag_value} | Chatbot: {chatbot_value}")
        
        print("-" * 60)

# Add this function to run evaluation for multiple models
async def evaluate_multiple_models():
    """
    Evaluate multiple models and compare their performance
    """
    # Get API key from environment variable
    api_key = os.getenv("GROQ_API_KEY")
    
    # Check if API key is set
    if not api_key:
        print("❌ Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it using: export GROQ_API_KEY='your-api-key-here'")
        return
    
    # Models to evaluate
    models_to_test = [
        "gemma2-9b-it",
        "llama-3.1-8b-instant", 
        "openai/gpt-oss-120b",
        "deepseek-r1-distill-llama-70b"
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL: {model_name}")
        print(f"{'='*60}")
        
        try:
            evaluator = GenericChatbotEvaluator(model_name=model_name)
            
            # Load the contexts from the test set
            file_path = 'data/test.json'
            contexts_to_test = load_cuad_data(file_path, num_contexts=1)  # Adjust num_contexts as needed

            # List to store detailed results
            results = []

            # Iterate through each context
            for context_data in contexts_to_test:
                title = context_data['title']
                context = context_data['context']
                qas = context_data['qas']
                
                print(f"\n--- Testing {model_name} on Contract: {title} ---")
                
                # Iterate through each Question-Answer pair
                for qa in qas:
                    question = qa['question']
                    true_answers = [ans['text'] for ans in qa['answers']] if not qa['is_impossible'] else ["NO_ANSWER"]

                    # --- Time the inference step ---
                    start_time = time.time()
                    predicted_answer = await evaluator.get_chatbot_response(context, question)
                    end_time = time.time()
                    time_taken = end_time - start_time

                    # --- Evaluate the single result ---
                    eval_outcome = evaluator.evaluate_single(predicted_answer, true_answers)
                    
                    # Print a summary for this question
                    print(f"\nQ: {question}")
                    print(f"  - True Answer(s): {true_answers}")
                    print(f"  - Model Predicted: {predicted_answer[:150]}...")  # Truncate for readability
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
                        'eval_outcome': eval_outcome
                    })

            # Calculate final metrics across all results
            if results:
                final_metrics = evaluator.calculate_metrics(results)
                all_results[model_name] = {
                    "final_metrics": final_metrics,
                    "detailed_results": results
                }
                
                print(f"\n📊 {model_name} Performance Metrics:")
                print("---------------------------------------")
                for metric, value in final_metrics.items():
                    if isinstance(value, float):
                        print(f"{metric.replace('_', ' ').title():<30}: {value:.4f}")
                    else:
                        print(f"{metric.replace('_', ' ').title():<30}: {value}")
                print("---------------------------------------")

                # Save detailed results for this model
                output_file = f"evaluation/{model_name.replace('/', '_')}_results.json"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results[model_name], f, indent=2, ensure_ascii=False)
                print(f"\n✅ {model_name} results saved to: {output_file}")
                
            else:
                print(f"No results were generated for {model_name}.")
                
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
            continue
    
    # Compare all models
    if all_results:
        compare_multiple_models(all_results)
        
        # Save comparison results
        comparison_file = "evaluation/model_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Model comparison saved to: {comparison_file}")

# Add this function to compare multiple models
def compare_multiple_models(all_results: Dict[str, Any]):
    """
    Compare the results of multiple models
    """
    print(f"\n{'='*80}")
    print("COMPARISON: MULTIPLE MODELS")
    print(f"{'='*80}")
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'average_inference_time_seconds']
    
    # Get RAG model results if available
    rag_results_file = "evaluation/evaluation_results.json"
    if os.path.exists(rag_results_file):
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        all_results["RAG_MODEL"] = rag_data
    
    # Print comparison table
    print(f"{'MODEL':<25}", end="")
    for metric in metrics_to_compare:
        print(f"{metric.upper():<15}", end="")
    print()
    
    print("-" * (25 + 15 * len(metrics_to_compare)))
    
    for model_name, results in all_results.items():
        metrics = results['final_metrics']
        print(f"{model_name:<25}", end="")
        for metric in metrics_to_compare:
            value = metrics.get(metric, 0)
            if isinstance(value, float):
                print(f"{value:.4f}{'':<11}", end="")
            else:
                print(f"{value}{'':<14}", end="")
        print()
    
    print("-" * (25 + 15 * len(metrics_to_compare)))
    
    # Find best model for each metric
    print("\n🏆 BEST PERFORMERS:")
    print("-" * 40)
    for metric in metrics_to_compare:
        best_model = None
        best_value = -1
        
        for model_name, results in all_results.items():
            value = results['final_metrics'].get(metric, 0)
            if (metric != 'average_inference_time_seconds' and value > best_value) or \
               (metric == 'average_inference_time_seconds' and (best_value == -1 or value < best_value)):
                best_value = value
                best_model = model_name
        
        if best_model:
            if metric != 'average_inference_time_seconds':
                print(f"{metric.replace('_', ' ').title():<25}: {best_model} ({best_value:.4f})")
            else:
                print(f"{metric.replace('_', ' ').title():<25}: {best_model} ({best_value:.2f}s)")

# Modify your main function to use the new multi-model evaluation
if __name__ == "__main__":
    # Run evaluation for multiple models
    asyncio.run(evaluate_multiple_models())
    
    # You can still run individual model evaluation if needed
    # asyncio.run(evaluate_generic_chatbot())
    
    # Compare results (after all evaluations have been run)
    rag_results_file = "evaluation/evaluation_results.json"
    if os.path.exists(rag_results_file):
        # Load all model results for comparison
        all_results = {}
        
        # Add RAG model results
        with open(rag_results_file, 'r', encoding='utf-8') as f:
            rag_data = json.load(f)
        all_results["RAG_MODEL"] = rag_data
        
        # Add other model results if they exist
        models = ["gemma2-9b-it", "llama-3.1-8b-instant", "openai_gpt-oss-120b", "deepseek-r1-distill-llama-70b"]
        for model in models:
            model_file = f"evaluation/{model}_results.json"
            if os.path.exists(model_file):
                with open(model_file, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                all_results[model] = model_data
        
        # Compare all models
        compare_multiple_models(all_results)