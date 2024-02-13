from datasets import load_dataset
import random

dataset_path = "data/corpus.jsonl"

# Load the dataset. Replace 'csv' with the appropriate format of your dataset if different.
dataset = load_dataset('json', data_files={'train': dataset_path})

# Print the names of the columns in the dataset
print("Column names:", dataset['train'].column_names)

# Print the first example in the dataset
# print("First example:", dataset['train'][10])s

corpus = dataset['train']['premises'] 
formal_data = [thm['code'] for lst in corpus for thm in lst if thm['code'].startswith("theorem")]
print(len(formal_data))
for _ in range(100):
    thm = random.choice(formal_data)
    print("true" if thm.startswith("theorem") else "false")
