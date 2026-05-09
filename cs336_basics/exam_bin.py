import numpy as np

train_data = np.memmap(
    "cs336_basics/train_path/TinyStoriesV2-GPT4-train.bin",
    dtype=np.uint16,
    mode="r",
)

valid_data = np.memmap(
    "cs336_basics/valid_path/TinyStoriesV2-GPT4-valid.bin",
    dtype=np.uint16,
    mode="r",
)

print("train min:", train_data.min())
print("train max:", train_data.max())
print("valid min:", valid_data.min())
print("valid max:", valid_data.max())