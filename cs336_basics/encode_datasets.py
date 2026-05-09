import os
import numpy as np
from cs336_basics.tokenizer import tokenizer as Tokenizer


BASE_DIR = os.path.dirname(__file__)

TRAIN_TXT_PATH = os.path.join(
    BASE_DIR,
    "train_path/TinyStoriesV2-GPT4-train.txt",
)

VALID_TXT_PATH = os.path.join(
    BASE_DIR,
    "valid_path/TinyStoriesV2-GPT4-valid.txt",
)

VOCAB_PATH = os.path.join(
    BASE_DIR,
    "test_experiments/bpe_tinystories_model/vocab.json",
)

MERGES_PATH = os.path.join(
    BASE_DIR,
    "test_experiments/bpe_tinystories_model/merges.txt",
)


def make_output_path(input_txt_path: str) -> str:
    """
    把 xxx.txt 转成同目录下的 xxx.bin
    """
    folder = os.path.dirname(input_txt_path)
    filename = os.path.basename(input_txt_path)
    stem, _ = os.path.splitext(filename)
    return os.path.join(folder, f"{stem}.bin")


def write_token_ids_to_bin(
    input_txt_path: str,
    output_bin_path: str,
    tok,
    dtype=np.uint16,
    write_buffer_tokens: int = 1024 * 1024,
):
    """
    将 txt 文件分段 encode 成 token ids，然后以二进制形式写入 .bin 文件。

    output_bin_path 之后可以用：
        np.memmap(output_bin_path, dtype=np.uint16, mode="r")
    读取。
    """
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)

    total_tokens = 0
    max_token_id = -1
    buffer = []

    print(f"Encoding: {input_txt_path}")
    print(f"Output  : {output_bin_path}")

    with open(output_bin_path, "wb") as out_f:

        output_buffer = []

        with open(input_txt_path, "r") as f:
            for id in tok.encode_iterable(f):
                id = int(id)
                output_buffer.append(id)

                if id > max_token_id:
                    max_token_id = id

                if len(output_buffer) >= write_buffer_tokens:
                    arr = np.asarray(output_buffer, dtype=dtype)
                    arr.tofile(out_f)
                    output_buffer.clear()

            if output_buffer:
                arr = np.asarray(output_buffer, dtype=dtype)
                arr.tofile(out_f)
        
        if dtype == np.uint16 and max_token_id >= 65536:
            raise ValueError(
                f"max_token_id={max_token_id} is too large for uint16. "
                f"Use np.uint32 instead."
            )


def main():
    tok = Tokenizer.from_files(
        vocab_filepath=VOCAB_PATH,
        merges_filepath=MERGES_PATH,
        special_tokens=None,
    )

    train_bin_path = make_output_path(TRAIN_TXT_PATH)
    valid_bin_path = make_output_path(VALID_TXT_PATH)

    write_token_ids_to_bin(
        input_txt_path=TRAIN_TXT_PATH,
        output_bin_path=train_bin_path,
        tok=tok,
        dtype=np.uint16,
    )

    write_token_ids_to_bin(
        input_txt_path=VALID_TXT_PATH,
        output_bin_path=valid_bin_path,
        tok=tok,
        dtype=np.uint16,
    )

    # 简单验证能否 memmap 读取
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    valid_data = np.memmap(valid_bin_path, dtype=np.uint16, mode="r")

    print("Memmap check:")
    print(f"train_data.shape = {train_data.shape}")
    print(f"valid_data.shape = {valid_data.shape}")
    print(f"train first 10 tokens = {train_data[:10].tolist()}")
    print(f"valid first 10 tokens = {valid_data[:10].tolist()}")


if __name__ == "__main__":
    main()