from datasets import load_dataset

# Load JSON files
train_ds = load_dataset("json", data_files="./data/medquad-split/training.json", split="train")
val_ds   = load_dataset("json", data_files="./data/medquad-split/validation.json", split="train")
test_ds  = load_dataset("json", data_files="./data/medquad-split/testing.json", split="train")

# Print first 10 samples from each
print("\n🔹 First 10 Training Samples:")
for i in range(10):
    print(train_ds[i])

print("\n🔹 First 10 Validation Samples:")
for i in range(10):
    print(val_ds[i])

print("\n🔹 First 10 Testing Samples:")
for i in range(10):
    print(test_ds[i])
