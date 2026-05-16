import numpy as np
from cs336_basics.tokenizer import tokenizer
from itertools import islice
import os

train_data = np.memmap(
    "train_path/TinyStoriesV2-GPT4-train.bin",
    dtype=np.uint16,
    mode="r",
)

valid_data = np.memmap(
    "valid_path/TinyStoriesV2-GPT4-valid.bin",
    dtype=np.uint16,
    mode="r",
)

print("train len:", len(train_data))
print("valid len:", len(valid_data))

print("train min/max:", train_data.min(), train_data.max())
print("valid min/max:", valid_data.min(), valid_data.max())

print("train unique:", len(np.unique(train_data[:1_000_000])))
print("valid unique:", len(np.unique(valid_data[:1_000_000])))


VOCAB_PATH = os.path.join(os.path.dirname(__file__), "test_experiments/bpe_tinystories_model/vocab.json")
MERGES_PATH = os.path.join(os.path.dirname(__file__), "test_experiments/bpe_tinystories_model/merges.txt")

bpe_tokenizer = tokenizer.from_files(VOCAB_PATH, MERGES_PATH, ["<|endoftext|>"])

# TRAIN_TXT_PATH = "train_path/TinyStoriesV2-GPT4-train.txt"

# eos = "<|endoftext|>"
# eos_id = bpe_tokenizer.encode(eos)[0]

# with open(TRAIN_TXT_PATH, "r", encoding="utf-8") as f:
#     # 取前 20 行，里面应该已经有 eos
#     text = "".join(islice(f, 20))

# print("contains eos string:", eos in text)
# print("text snippet:")
# print(repr(text[:500]))

# ids_direct = bpe_tokenizer.encode(text)
# print("direct encode eos count:", ids_direct.count(eos_id))
# print("direct contains eos:", eos_id in ids_direct)

# ids_iter = list(bpe_tokenizer.encode_iterable([text]))
# print("iter encode eos count:", ids_iter.count(eos_id))
# print("iter contains eos:", eos_id in ids_iter)

# print("direct decode snippet:")
# print(bpe_tokenizer.decode(ids_direct[:300]))

eos = "<|endoftext|>"
ids = bpe_tokenizer.encode(eos)

print(ids)
print(len(ids))
print([bpe_tokenizer.decode([i]) for i in ids])

eos_id = bpe_tokenizer.encode("<|endoftext|>")[0]

train_eos_count = int((train_data == eos_id).sum())
valid_eos_count = int((valid_data == eos_id).sum())

print("eos_id:", eos_id)
print("train eos count:", train_eos_count)
print("valid eos count:", valid_eos_count)

print("train eos ratio:", train_eos_count / len(train_data))
print("valid eos ratio:", valid_eos_count / len(valid_data))

ids = train_data[:200].tolist()
print(ids)
print(bpe_tokenizer.decode(ids))

sample = train_data[:5_000_000]
values, counts = np.unique(sample, return_counts=True)

top_idx = np.argsort(counts)[-30:][::-1]

for idx in top_idx:
    token_id = int(values[idx])
    count = int(counts[idx])
    token_text = bpe_tokenizer.decode([token_id])
    print(token_id, count, repr(token_text))

