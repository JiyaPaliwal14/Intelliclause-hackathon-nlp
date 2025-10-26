import json

def load_cuad_data(file_path, num_contexts=10):
    """
    Loads the CUAD dataset and returns the first 'num_contexts' contexts
    along with all their associated Q&A pairs.

    Args:
        file_path (str): Path to the CUAD JSON file (e.g., 'data/test.json')
        num_contexts (int): Number of contexts to extract from the beginning.

    Returns:
        list: A list of dictionaries. Each dictionary represents a context and its Q&As.
              Format: [{'title': str, 'context': str, 'qas': list}, ...]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract the list of data items (each item is a title with multiple paragraphs/contexts)
    all_data_items = data['data']
    
    selected_contexts_list = []
    contexts_count = 0

    # Iterate through each document (each item in the 'data' list)
    for document in all_data_items:
        # Iterate through each context/paragraph in this document
        for paragraph in document['paragraphs']:
            # Stop if we have enough contexts
            if contexts_count >= num_contexts:
                break
            
            # Append the context and its Q&As to our list
            selected_contexts_list.append({
                'title': document['title'],
                'context': paragraph['context'],
                'qas': paragraph['qas']
            })
            contexts_count += 1
        
        # Break the outer loop if we have enough contexts
        if contexts_count >= num_contexts:
            break

    print(f"Successfully loaded {len(selected_contexts_list)} contexts.")
    return selected_contexts_list

# Optional: If you want to test this function directly
if __name__ == "__main__":
    sample_data = load_cuad_data('../data/test.json', num_contexts=2)
    print(f"First context title: {sample_data[0]['title']}")
    print(f"Number of Q&As in first context: {len(sample_data[0]['qas'])})")