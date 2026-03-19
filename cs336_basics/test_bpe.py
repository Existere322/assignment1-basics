import time
from bpe_tokenizer import vocab_init, pre_tokenization, merge
import os
import json
from pathlib import Path



def main():
    print(f"当前主进程 PID: {os.getpid()}")
    start_time = time.time()
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    output_dir = "bpe_tinystories_model"

    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")  # 用于拼接文件目录和文件名的函数

    print("开始初始化...")
    vocab = vocab_init(special_tokens)

    print("开始预分词 (Pre-tokenization)...")
    word_counts = pre_tokenization(input_path, special_tokens)
    pre_tok_time = time.time()
    print(f"预分词耗时: {pre_tok_time - start_time:.2f} 秒")

    print("开始 BPE 合并 (Merge)...")
    vocab_result, merge_process = merge(word_counts, 10000, vocab)
    
    end_time = time.time()
    print(f"BPE 合并耗时: {end_time - pre_tok_time:.2f} 秒")
    print(f"总计耗时: {(end_time - start_time) / 60:.2f} 分钟")
    
    # 1. 修复最长 Token 的查找逻辑
    # 注意：vocab_result 是 {id: bytes}，所以我们要对 values() 进行长度比较
    longest_token_bytes = max(vocab_result.values(), key=len)
    longest_token_str = longest_token_bytes.decode("utf-8", errors="backslashreplace")
    print(f"最长的 Token 是: {longest_token_str}")
    print(f"长度: {len(longest_token_bytes)} 字节")
    # 思考：这个词是否有意义？通常它会是 TinyStories 中高频出现的固定短语片段。

    # 2. 修复 JSON 保存逻辑 (确保键和值都是 JSON 可序列化的类型)
    # 我们将保存为 { "token_string": token_id } 这种标准格式
    vocab_to_save = {}
    for token_id, token_bytes in vocab_result.items():
        if isinstance(token_bytes, bytes):
            # 将字节解码为字符串，非 UTF-8 字符用反斜杠转义，确保 JSON 安全
            token_str = token_bytes.decode("utf-8", errors="backslashreplace")
        else:
            token_str = str(token_bytes)
        vocab_to_save[token_str] = token_id

    with open(vocab_path, "w", encoding="utf-8") as f:
        # indent=2 使文件易于人类阅读检查
        json.dump(vocab_to_save, f, ensure_ascii=False, indent=2)

    # 3. 修复并保存 merges.txt
    merge_path = os.path.join(output_dir, "merges.txt")
    with open(merge_path, "w", encoding="utf-8") as f:
        for pair in merge_process:
            # pair 通常是 (bytes, bytes)
            p1 = pair[0].decode("utf-8", errors="backslashreplace")
            p2 = pair[1].decode("utf-8", errors="backslashreplace")
            # 按照标准 BPE 格式，每行保存合并的两个部分，中间空格隔开
            f.write(f"{p1} {p2}\n")

    # 4. 保存耗时统计
    time_path = os.path.join(output_dir, "time_consum.txt")
    with open(time_path, "w", encoding="utf-8") as f:
        f.write(f"预分词耗时: {pre_tok_time - start_time:.2f} 秒\n")
        f.write(f"BPE 合并耗时: {end_time - pre_tok_time:.2f} 秒\n")
        f.write(f"总计耗时: {(end_time - start_time) / 60:.2f} 分钟\n")

if __name__ == "__main__":
    main()