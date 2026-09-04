from datasets import Dataset

base = "/home/hyang150/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866"

train = Dataset.from_file(f"{base}/gsm8k-train.arrow")
test = Dataset.from_file(f"{base}/gsm8k-test.arrow")

print("train:", len(train))
print("test :", len(test))

print("\ntrain example:")
print(train[0])

assert len(train) == 7473
assert len(test) == 1319

print("\n✅ GSM8K 完整")
PY