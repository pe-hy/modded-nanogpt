import json
import random
import sys
from pathlib import Path
from typing import List
import os
sys.path.append(str(Path(__file__).parent.parent))
from utils.config_loader import load_config


def flip_digits(number_str: str) -> str:
    """Flip the digits of a number string."""
    return number_str[::-1]


def generate_arithmetic_example(config=None) -> str:
    """Generate a single arithmetic addition example with flipped result."""
    # Get parameters from config or use defaults
    if config:
        min_len = config.get('dataset.min_operand_length', 1)
        max_len = config.get('dataset.max_operand_length', 10)
        flip_result = config.get('dataset.flip_result', True)
    else:
        min_len, max_len, flip_result = 1, 10, True
    
    # Generate two random numbers with specified length range
    operand1_length = random.randint(min_len, max_len)
    operand2_length = random.randint(min_len, max_len)
    
    # Generate operands (avoiding leading zeros for multi-digit numbers)
    if operand1_length == 1:
        operand1 = random.randint(0, 9)
    else:
        operand1 = random.randint(10**(operand1_length-1), 10**operand1_length - 1)
    
    if operand2_length == 1:
        operand2 = random.randint(0, 9)
    else:
        operand2 = random.randint(10**(operand2_length-1), 10**operand2_length - 1)
    
    # Calculate the result
    result = operand1 + operand2
    
    # Convert to strings and add spaces between digits
    operand1_str = ' '.join(str(operand1))
    operand2_str = ' '.join(str(operand2))
    result_str = ' '.join(str(result))
    
    # Flip the result digits if configured to do so
    if flip_result:
        flipped_result = flip_digits(str(result))
        final_result_str = ' '.join(flipped_result)
    else:
        final_result_str = result_str
    
    # Create the example string: "1 0 + 1 0 = 0 2"
    example = f"{operand1_str} + {operand2_str} = {final_result_str}"
    
    return example


def generate_datasets(config=None) -> None:
    """
    Generate separate train and test datasets with non-overlapping examples.
    Uses configuration file for all parameters.
    """
    # Load configuration
    if config is None:
        config = load_config()
    
    # Get parameters from config
    train_examples = config.get('dataset.train_examples', 8000)
    test_examples = config.get('dataset.test_examples', 2000)
    
    # Use new path structure
    dataset_dir = config.get('paths.dataset_dir', 'data/datasets')
    train_filename = config.get('paths.train_file', 'train.json')
    test_filename = config.get('paths.test_file', 'test.json')
    
    # Create full paths
    train_file = os.path.join(dataset_dir, train_filename)
    test_file = os.path.join(dataset_dir, test_filename)
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    
    total_examples = train_examples + test_examples
    generated_examples = set()  # Track generated examples to avoid duplicates
    
    print(f"Generating {total_examples} arithmetic examples ({train_examples} train, {test_examples} test)...")
    
    train_data = []
    test_data = []
    
    # Generate training examples
    print("Generating training examples...")
    while len(train_data) < train_examples:
        example_text = generate_arithmetic_example(config)
        if example_text not in generated_examples:
            generated_examples.add(example_text)
            train_data.append({"text": example_text})
            
        if len(train_data) % 1000 == 0:
            print(f"Generated {len(train_data)}/{train_examples} training examples")
    
    # Generate test examples (non-overlapping)
    print("Generating test examples...")
    while len(test_data) < test_examples:
        example_text = generate_arithmetic_example(config)
        if example_text not in generated_examples:
            generated_examples.add(example_text)
            test_data.append({"text": example_text})
            
        if len(test_data) % 500 == 0:
            print(f"Generated {len(test_data)}/{test_examples} test examples")
    
    # Save training data
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    # Save test data
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDatasets saved:")
    print(f"  Training: {train_file} ({len(train_data)} examples)")
    print(f"  Test: {test_file} ({len(test_data)} examples)")
    
    # Print sample examples from both sets
    print("\nSample training examples:")
    for i in range(min(3, len(train_data))):
        print(f"  {train_data[i]['text']}")
    
    print("\nSample test examples:")
    for i in range(min(3, len(test_data))):
        print(f"  {test_data[i]['text']}")


if __name__ == "__main__":
    # Generate separate train and test datasets using configuration
    generate_datasets()