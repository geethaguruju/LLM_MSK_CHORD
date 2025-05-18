from datasets import load_dataset

# Load the dataset
ds = load_dataset("lavita/MedQuAD", split="train")

# # Function to count null or empty values in a column
# def count_nulls(dataset, column_name):
#     return sum(1 for row in dataset if not row[column_name] or str(row[column_name]).strip() == "")

# # Count nulls in each split
# for split in ds.keys():
#     dataset = ds[split]
#     null_questions = count_nulls(dataset, "question")
#     null_answers = count_nulls(dataset, "answer")
    
#     print(f"Split: {split}")
#     print(f"  Null 'question' values: {null_questions}")
#     print(f"  Null 'answer' values: {null_answers}")

print("\n🔹 First 10 Training Samples:")
for i in range(10):
    print(ds[i])
