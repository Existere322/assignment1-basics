from cs336_basics.module import Transformer_LM, cross_entropy, AdamW
from cs336_basics.module import learning_rate_schedule, gradient_clipping, data_loading
from cs336_basics.module import save_checkpoint, load_checkpoint
import torch
import numpy as np
import argparse
import json
import os
import time

def parse_args():
    p = argparse.ArgumentParser()
    g_model = p.add_argument_group("model")
    g_model.add_argument("--vocab_size", type=int, default=32000)
    g_model.add_argument("--context_length", type=int, default=512)
    g_model.add_argument("--batch_size", type=int, default=32)
    g_model.add_argument("--num_layers", type=int, default=4)
    g_model.add_argument("--num_heads", type=int, default=8)
    g_model.add_argument("--d_ff", type=int, default=1344)
    g_model.add_argument("--d_model", type=int, default=512)
    g_model.add_argument("--rope_theta", type=float, default=10000.0)
    
    
    g_optimizer = p.add_argument_group("optimizer")
    g_optimizer.add_argument("--weight_decay", type=float, default=0.1)
    g_optimizer.add_argument("--betas", type=float, default=[0.9, 0.95], nargs=2)
    g_optimizer.add_argument("--eps", type=float, default=1e-8)


    g_training = p.add_argument_group("training")
    g_training.add_argument("--max_learning_rate", type=float, default=3e-4)
    g_training.add_argument("--min_learning_rate", type=float, default=3e-5)
    g_training.add_argument("--warmup_iters", type=int, default=500)
    g_training.add_argument("--cosine_cycle_iters", type=int, default=50000)
    g_training.add_argument("--end_iter", type=int, default=50000)
    g_training.add_argument("--grad_clip_norm", type=float, default=1.0)
    g_training.add_argument("--val_interval", type=int, default=500)
    g_training.add_argument("--val_batches", type=int, default=20)
    g_training.add_argument("--device", type=str, default="cuda")
    g_training.add_argument("--seed", type=int, default=42)
    g_training.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])


    g_IO = p.add_argument_group("IO")
    g_IO.add_argument("--step_to_save", type=int, default=1000)
    g_IO.add_argument("--save_N", type=int, default=10)
    g_IO.add_argument("--train_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "train_path"))
    g_IO.add_argument("--val_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "valid_path"))
    g_IO.add_argument("--save_position", type=str,
                      default=os.path.join(os.path.dirname(__file__), "save_position"))
    g_IO.add_argument("--log_path", type=str,
                      default=os.path.join(os.path.dirname(__file__), "log_position"))
    g_IO.add_argument("--log_interval", type=int, default=100)
    
    return p.parse_args()


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
    # TODO: tokenid 所需要的数据类型待定

    # 在有检查点的情况下加载检查点
    if os.path.exists(args.save_position) and os.listdir(args.save_position):
        latest = os.path.join(args.save_position, "model_latest.pt")
        if os.path.exists(latest):
            start_iter = load_checkpoint(latest, transformer_model, optimizer) + 1

    log_file = open(os.path.join(args.log_path, "train.jsonl"), "a")


    # 按照步数迭代
    t = start_iter - 1
    try:
        last_log_time = time.time()
        for t in range(start_iter, args.end_iter):
            
            lr_t = learning_rate_schedule(t, args.max_learning_rate, args.min_learning_rate, args.warmup_iters, args.cosine_cycle_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr_t

            x, y = data_loading(train_data, args.batch_size, args.context_length, device)

            logits = transformer_model(x)
            loss = cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()

            total_grad = gradient_clipping(transformer_model.parameters(), args.grad_clip_norm)

            if t > 0 and t % args.log_interval == 0:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                now = time.time()
                avg_step_time = (now - last_log_time) / args.log_interval if t > start_iter else 0.0
                last_log_time = now
                
                print(f"step:{t}, train_loss:{loss.item()}, lr:{lr_t}, grad_norm:{total_grad:.4f}, step_time:{avg_step_time:.3f}s")
                log_file.write(json.dumps({"step": t, "loss": loss.item(), "lr": lr_t, "grad_norm": total_grad, "step_time": avg_step_time}) + "\n")
                log_file.flush()

            if t > 0 and t % args.val_interval == 0:
                val_loss = evaluate(transformer_model, val_data, args.batch_size,
                                    args.context_length, args.val_batches, device)
                print(f"step:{t}, val_loss:{val_loss:.4f}")
                log_file.write(json.dumps({"step": t, "val_loss": val_loss}) + "\n")
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