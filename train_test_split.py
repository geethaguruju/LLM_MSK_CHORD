import json
import random
import os
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# from deepcopy import copy

# Load the JSON file
with open('cot_data_response.json', 'r') as file:
    cot_data_response = json.load(file)

# Convert each value of cot_data_response to JSON strings
keys_to_remove = []
for key, value in cot_data_response.items():
    try:
        cot_data_response[key] = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        keys_to_remove.append(key)

# Remove problematic keys
for key in keys_to_remove:
    del cot_data_response[key]

# Load the JSON file 'promt_data.json'
with open('prompt_data.json', 'r') as file:
    prompt_data = json.load(file)

# Shuffle the prompt_data randomly
random.shuffle(prompt_data)

# Extract all patient_ids from cot_data_response
patient_ids = set(cot_data_response.keys())

# Separate prompt_data into train and test sets
train_data = []
test_data = []

# Split prompt_data based on patient_ids
for item in prompt_data:
    if item.get('patient_id') in patient_ids:
        train_data.append(item)
    else:
        test_data.append(item)

cot_train_data = train_data.copy()

print(f"Length of train_data: {len(cot_train_data)}")

# Calculate the number of additional items needed for the train set
train_size = int(0.8 * len(prompt_data))
additional_needed = train_size - len(train_data)

# Add more items to train_data if needed
if additional_needed > 0:
    remaining_data = [item for item in test_data if item not in train_data]
    random.shuffle(remaining_data)
    train_data.extend(remaining_data[:additional_needed])
    test_data = remaining_data[additional_needed:]

# Print the sizes of the splits (optional)
print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

# Create a Hugging Face dataset
def create_hf_dataset(data):
    return Dataset.from_dict({
        "patient_id": [item.get("patient_id") for item in data],
        "patient_data": [item.get("patient_data") for item in data],
        "survival_status": [item.get("survival_status") for item in data],
        "survival_months": [item.get("survival_months") for item in data],
    })

def create_json_dataset(data):
    json_data = []
    for item in data:
        json_data.append({
            "patient_id": item.get("patient_id"),
            "patient_data": item.get("patient_data"),
            "survival_status": item.get("survival_status"),
            "survival_months": item.get("survival_months"),
        })
    return json_data

# Create train and test datasets
train_dataset = create_hf_dataset(train_data)
test_dataset = create_hf_dataset(test_data)

train_json_data = create_json_dataset(train_data)
test_json_data = create_json_dataset(test_data)

# Print the lengths of the JSON data
print(f"Length of train_json_data: {len(train_json_data)}")
print(f"Length of test_json_data: {len(test_json_data)}")

msk_chord_json_dataset = {
    "train": train_json_data,
    "test": test_json_data
}
# Save the datasets as JSON files
# with open('msk_chord_dataset.json', 'w') as json_file:
#     json.dump(msk_chord_json_dataset, json_file)


# Save as separate JSON files
train_dataset.to_json("msk_chord_dataset_train.json", orient="records", lines=True)
test_dataset.to_json("msk_chord_dataset_test.json", orient="records", lines=True)

def cot_json_dataset(cot_train_data):
    print(f"Length of cot_train_data: {len(cot_train_data)}")
    json_data = []
    for item in cot_train_data:
        json_data.append({
            "patient_id": item.get("patient_id"),
            "patient_data": item.get("patient_data"),
            "survival_status": item.get("survival_status"),
            "survival_months": item.get("survival_months"),
            "chain_of_thought": cot_data_response.get(item.get("patient_id"), {}).get("chain_of_thought", ""),
            "comments": cot_data_response.get(item.get("patient_id"), {}).get("comments", ""),
        })

    # Split the cot_train_data into 80% train and 20% test
    split_index = int(0.8 * len(cot_train_data))
    random.shuffle(cot_train_data)
    cot_train_split = cot_train_data[:split_index]
    cot_test_split = cot_train_data[split_index:]

    print(f"Length of cot_train_split: {len(cot_train_split)}")
    print(f"Length of cot_test_split: {len(cot_test_split)}")

    return {"train": cot_train_split, "test": cot_test_split}

# Add 'chain_of_thought' and 'comments' columns to cot_train_data

def cot_dataset(cot_train_data):
    print(f"Length of cot_train_data: {len(cot_train_data)}")
    dataset = Dataset.from_dict({
    "patient_id": [item.get("patient_id") for item in cot_train_data],
    "patient_data": [item.get("patient_data") for item in cot_train_data],
    "survival_status": [item.get("survival_status") for item in cot_train_data],
    "survival_months": [item.get("survival_months") for item in cot_train_data],
    "chain_of_thought": [cot_data_response.get(item.get("patient_id"), {}).get("chain_of_thought", "") for item in cot_train_data],
    "comments": [cot_data_response.get(item.get("patient_id"), {}).get("comments", "") for item in cot_train_data],
    })

    return dataset

# Add the 'split' column with value 'train'
# cot_train_dataset = cot_train_dataset.add_column("split", ["train"] * len(cot_train_dataset))

# Split the cot_train_data into 80% train and 20% test
split_index = int(0.8 * len(cot_train_data))
random.shuffle(cot_train_data)
cot_train_split = cot_train_data[:split_index]
cot_test_split = cot_train_data[split_index:]

cot_train_dataset = cot_dataset(cot_train_split)
cot_test_dataset = cot_dataset(cot_test_split)

# Save as separate JSON files
cot_train_dataset.to_json("msk_chord_cot_dataset_train.json", orient="records", lines=True)
cot_test_dataset.to_json("msk_chord_cot_dataset_test.json", orient="records", lines=True)

# cot_json_data = cot_json_dataset(cot_train_data)
# # Save the cot_train_dataset as a JSON file
# with open('msk_chord_cot_dataset.json', 'w') as json_file:
#     json.dump(cot_json_data, json_file)

# # Save the cot_train_dataset as a Hugging Face dataset
# # Save the cot_train_dataset locally
# cot_train_dataset.save_to_disk("msk_chord_cot_dataset")

# # Upload the cot_train_dataset to Hugging Face Hub as a private dataset
# api.upload_folder(
#     folder_path="msk_chord_cot_dataset",
#     repo_id="RaghuHemadri/msk_chord_cot_dataset",  # Replace 'your-username' with your Hugging Face username
#     repo_type="dataset",
#     commit_message="Upload msk_chord_cot_dataset"
# )
