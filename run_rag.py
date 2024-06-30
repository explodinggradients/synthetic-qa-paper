from datasets import load_dataset
from rag_utils import SimpleRAG
import argparse, yaml, json, torch

# Define a dictionary to map data types to PyTorch data types
DTYPE_2_TORCH_DTYPE = {
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32
}

# Function to load questions from Hugging Face datasets
def load_questions_from_hf(hf_id: str, split: str, column: str):
    """
    Load questions from a Hugging Face dataset.

    Args:
        hf_id (str): The identifier of the Hugging Face dataset.
        split (str): The split of the dataset to load.
        column (str): The column in the dataset that contains the questions.

    Returns:
        list: A list of questions.
    """
    ds = load_dataset(hf_id, split=split)
    ls_questions = ds[column]
    return ls_questions

# Function to load questions from a file
def load_questions_from_file_lines(file_path: str):
    """
    Load questions from a file. Each line in the file should contain a question.

    Args:
        file_path (str): The path to the file containing the questions.

    Returns:
        list: A list of questions.
    """
    with open(file_path, 'r') as file:
        ls_questions = file.readline()
    return ls_questions

# Function to save outputs to a JSON file
def save_outputs_to_json(data: dict, save_path: str):
    """
    Save data to a JSON file.

    Args:
        data (dict): The data to save.
        save_path (str): The path to the file where the data should be saved.
    """
    with open(save_path, 'w') as json_file:
        print(f'- Saving RAG Outputs: {save_path}')
        json.dump(data, json_file, indent=True)

# Main function
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="This script accepts two command-line arguments.")
    parser.add_argument('--yaml-config', 
                        dest='yaml_config', 
                        default='', 
                        help='Path to the YAML configuration file. Default is an empty string.')
    parser.add_argument('--run-mode', 
                        dest='run_mode', 
                        default='single', 
                        choices=['single', 'batch'], 
                        help='Mode in which to run the script. Default is "single". Options are "single" and "batch".')
    args = parser.parse_args()
    
    # Load configuration from a YAML file
    with open(args.yaml_config, 'r') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
        config['generator_model_torch_dtype'] = DTYPE_2_TORCH_DTYPE[config['generator_model_torch_dtype']]
    
    # Instantiate a SimpleRAG object
    rag = SimpleRAG.from_yaml_config(config)
    
    # Determine the run mode
    if args.run_mode == 'batch':
        # Load list of questions
        if config['questions_from_hf']:
            ls_questions = load_questions_from_hf(config['questions_dataset_id'], config['questions_dataset_split'], config['questions_dataset_column'])
        elif config['questions_from_file_lines']:
            ls_questions = load_questions_from_file_lines(config['questions_file_lines_path']) 
        elif config['questions_from_yaml']:
            ls_questions = config['questions_yaml_list']
        
        # Run RAG on each question
        data = rag.rag_pipeline(ls_questions)
        
        # Save results
        save_outputs_to_json(data, config['output_path'])
    
    elif args.run_mode == 'single':
        cont = True
        while cont:
            question = input('Enter a question:\n')
            rag_answer, retrieved_docs = rag.rag_invoke(question)
            print(rag_answer)
            
            cont = input('Continue? (y/n)') == 'y'
