from __future__ import annotations

import json
import os
import resource
import sys

import psutil
import pytest
import tiktoken
import pathlib

from tests.adapters import get_tokenizer
from tests.common import gpt2_bytes_to_unicode
from cs336_basics.tokenizer import tokenizer


VOCAB_PATH = (pathlib.Path(__file__).resolve().parent) / "bpe_tinystories_model/vocab.json"
MERGES_PATH = (pathlib.Path(__file__).resolve().parent) / "bpe_tinystories_model/merges.txt"
INPUT_PATH = (pathlib.Path(__file__).resolve().parent) / "sample_owt.txt"


'''
__file__                          # 当前这个 .py 文件的路径（字符串）
pathlib.Path(__file__)            # 转成 Path 对象，方便操作
.resolve()                        # 转成绝对路径（解析掉 .. 和软链接）
.parent                           # 取上一级目录（即当前文件所在的文件夹）
/ "fixtures"                      # 拼接子目录 fixtures
'''



def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


tokenizer = tokenizer.from_files(vocab_filepath=VOCAB_PATH,
                                merges_filepath=MERGES_PATH,
                                special_tokens=["<|endoftext|>"])

with open(INPUT_PATH) as f:
    corpus_contents = f.read()

ids = tokenizer.encode(corpus_contents)
print(f"原始字符数量：{len(corpus_contents)}")
print(f"token 数量：{len(ids)}")
content = tokenizer.decode(ids)