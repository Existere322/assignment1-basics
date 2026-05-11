from cs336_basics.module import Transformer_LM, cross_entropy, AdamW
from cs336_basics.module import learning_rate_schedule, gradient_clipping, data_loading
from cs336_basics.module import save_checkpoint, load_checkpoint, top_p_sampling, temperature_scaling_softmax
from cs336_basics.tokenizer import tokenizer
import torch
import numpy as np
from datetime import datetime
import argparse
import json
import os
import time

def parse_args():
    p = argparse.ArgumentParser()
    g_model = p.add_argument_group("model")
    g_model.add_argument("--vocab_size", type=int, default=10000)
    g_model.add_argument("--context_length", type=int, default=256)
    g_model.add_argument("--batch_size", type=int, default=64)
    # batch_size * total_step_count * context_length = 327680000
    g_model.add_argument("--num_layers", type=int, default=4)
    g_model.add_argument("--num_heads", type=int, default=16)
    g_model.add_argument("--d_ff", type=int, default=1344)
    g_model.add_argument("--d_model", type=int, default=512)
    g_model.add_argument("--rope_theta", type=float, default=10000.0)
    
    
    g_optimizer = p.add_argument_group("optimizer")
    g_optimizer.add_argument("--weight_decay", type=float, default=0.1)
    g_optimizer.add_argument("--betas", type=float, default=[0.9, 0.95], nargs=2)
    g_optimizer.add_argument("--eps", type=float, default=1e-8)


    g_training = p.add_argument_group("training")
    g_training.add_argument("--max_learning_rate", type=float, default=1e-3)
    g_training.add_argument("--min_learning_rate", type=float, default=1e-4)
    g_training.add_argument("--warmup_iters", type=int, default=1000)
    g_training.add_argument("--cosine_cycle_iters", type=int, default=19500)
    g_training.add_argument("--end_iter", type=int, default=20000)
    g_training.add_argument("--grad_clip_norm", type=float, default=1.0)
    g_training.add_argument("--val_interval", type=int, default=500)
    g_training.add_argument("--val_batches", type=int, default=50)
    g_training.add_argument("--device", type=str, default="cuda")
    g_training.add_argument("--seed", type=int, default=336)
    g_training.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])


    g_IO = p.add_argument_group("IO")
    g_IO.add_argument("--step_to_save", type=int, default=1000)
    g_IO.add_argument("--save_N", type=int, default=10)
    g_IO.add_argument("--train_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "train_path/TinyStoriesV2-GPT4-train.bin"))
    g_IO.add_argument("--val_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "valid_path/TinyStoriesV2-GPT4-valid.bin"))
    g_IO.add_argument("--save_position", type=str,
                      default=os.path.join(os.path.dirname(__file__), "save_position"))
    g_IO.add_argument("--log_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "log_position"))
    g_IO.add_argument("--log_interval", type=int, default=100)
    g_IO.add_argument("--max_token_num", type=int, default=100)
    g_IO.add_argument("--user_prompt", type=str, default=None)
    g_IO.add_argument("--temperature", type=float, default=0.5)
    g_IO.add_argument("--sampling_prob", type=float, default=0.8)
    
    return p.parse_args()

VOCAB_PATH = os.path.join(os.path.dirname(__file__), "test_experiments/bpe_tinystories_model/vocab.json")
MERGES_PATH = os.path.join(os.path.dirname(__file__), "test_experiments/bpe_tinystories_model/merges.txt")

def generating_text(model: Transformer_LM, user_prompt, max_token_num, temperature, sampling_prob, context_length, device):
    bpe_tokenizer = tokenizer.from_files(vocab_filepath=VOCAB_PATH, merges_filepath=MERGES_PATH)

    model.eval()

    encode_prompt = bpe_tokenizer.encode(user_prompt)

    input_ids = torch.tensor([encode_prompt], dtype=torch.long, device=device)
    # input_ids shape: [1, prompt_len]

    generated_ids = encode_prompt.copy()

    with torch.no_grad():
        for _ in range(max_token_num):

            input_context = input_ids[:, -context_length:]
            # 上下文窗口沿着生成的全部 token 移动，每次只取与上下文窗口一样数量的 token

            logits = model(input_context)  
            # output shape : [batch_size, context_len, vocab_size]

            next_token_logits = logits[:, -1, :]
            # next_token_logits shape: [1, vocab_size]

            softmax_result = temperature_scaling_softmax(next_token_logits, -1, temperature)
            top_p_result = top_p_sampling(softmax_result, -1, sampling_prob)
            next_token_id = torch.multinomial(top_p_result, num_samples=1)
            # Returns the indices of the maximum value of all elements in the :attr:`input` tensor.

            new_id = next_token_id.item()
            generated_ids.append(new_id)

            next_token = bpe_tokenizer.decode([new_id])
            if next_token == 'The':
                print("a new line !")
            if next_token == "<|endoftext|>" or new_id == 256: 
                break

            input_ids = torch.cat([input_ids, next_token_id], dim=1, )
            # Concatenates the given sequence of tensors in tensors in the given dimension. 


        generated_text = bpe_tokenizer.decode(generated_ids)
        print(f"Generate result: {generated_text}")



def evaluate(model, val_data, batch_size, context_length, num_batches, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = data_loading(val_data, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main(args):
    """
    TODO: 需要用到的函数与模块：
    √ 1. Transformer LM 模型本身  
    √ 2. cross_entropy 交叉熵损失函数——模型输出的概率分布与真实分布的差别
    √ 3. AdamW 优化器用来更新参数状态
    √ 4. learning_rate_schedule 用来获得当前的学习率
    √ 5. gradient_clipping 避免过大的损失影响训练
    √ 6. data_loading 加载数据并且格式化
    √ 7. save_checkpoint 和 load_checkpoint 保存断点和加载断点，可以设置多久保存/加载一次
    √ 8. 使用 argparse 传参
    √ 9. 统计训练信息 还剩 tokens/sec 或 step_time(看训练速度,后面优化时有用) 没有统计
    √ 10. 实现评估训练结果
    """

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    start_iter = 0

    dtype_dict = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_dict[args.dtype]

    transformer_model = Transformer_LM(args.vocab_size, args.context_length,
                                    args.d_model, args.num_layers, 
                                    args.num_heads, args.d_ff, 
                                    args.rope_theta, device, dtype)

    optimizer = AdamW(transformer_model.parameters(), args.max_learning_rate, args.weight_decay, args.betas, args.eps)

    train_data = np.memmap(args.train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode="r")

    # 在有检查点的情况下加载检查点
    if os.path.exists(args.save_position) and os.listdir(args.save_position):
        latest = os.path.join(args.save_position, "model_latest.pt")
        if os.path.exists(latest):
            start_iter = load_checkpoint(latest, transformer_model, optimizer) + 1

    if args.user_prompt is not None:
        generating_text(transformer_model, args.user_prompt, args.max_token_num, args.temperature, args.sampling_prob, args.context_length, args.device)

        return

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(args.log_path, f"train_{run_time}.jsonl")
    log_file = open(log_file_path, "a")


    # 按照步数迭代
    t = 0
    try:
        last_log_time = time.time()
        for t in range(start_iter, args.end_iter):
            step = t + 1
            
            lr_t = learning_rate_schedule(t, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr_t

            x, y = data_loading(train_data, args.batch_size, args.context_length, device)

            logits = transformer_model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))


            loss.backward()

            total_grad = gradient_clipping(transformer_model.parameters(), args.grad_clip_norm)

            if step > 0 and step % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                avg_step_time = (now - last_log_time) / args.log_interval if t > start_iter else 0.0
                last_log_time = now
                
                print(f"step:{step}, train_loss:{loss.item()}, lr:{lr_t}, grad_norm:{total_grad:.4f}, step_time:{avg_step_time:.3f}s")
                log_file.write(json.dumps({"step": step, "loss": loss.item(), "lr": lr_t, "grad_norm": total_grad, "step_time": avg_step_time}) + "\n")
                log_file.flush()

            if step > 0 and step % args.val_interval == 0:
                val_loss = evaluate(transformer_model, val_data, args.batch_size,
                                    args.context_length, args.val_batches, device)
                print(f"step:{step}, val_loss:{val_loss:.4f}")
                log_file.write(json.dumps({"step": step, "val_loss": val_loss}) + "\n")
                log_file.flush()
                last_log_time = time.time()  # 重置,避免下次 step_time 包含 val 时间

            optimizer.step()
            optimizer.zero_grad()

            # 每隔特定步数保存检查点
            if t > start_iter and t % args.step_to_save == 0:
                save_position = os.path.join(args.save_position, f"model_saved_{(t // args.step_to_save) % args.save_N}.pt")
                save_checkpoint(transformer_model, optimizer, t, save_position)

                latest_position = os.path.join(args.save_position, "model_latest.pt")
                save_checkpoint(transformer_model, optimizer, t, latest_position)
    finally:
        log_file.close()
        latest_position = os.path.join(args.save_position, "model_latest.pt")
        save_checkpoint(transformer_model, optimizer, t, latest_position)


if __name__ == "__main__": main(parse_args())