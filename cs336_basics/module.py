import torch.nn as nn
import torch
from einops import rearrange, einsum
from collections.abc import Callable, Iterable 
import numpy.typing as npt
from typing import IO, Any, BinaryIO
import numpy as np
from typing import Optional 
import os
import math


class Linear(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int, 
        device: torch.device | None=None, 
        dtype: torch.dtype | None=None):

        # 调用父类进行初始化
        super().__init__()

        # 创建化权重矩阵 W 为 d_in, d_out 并初始化
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = (2 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W, x, "d_out d_in, ... d_in -> ... d_out")
    

class Embedding(nn.Module):
    def __init__(self, 
                 num_embeddings: int, 
                 embedding_dim: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.embedding_matrix, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # python 的高级索引，会把每个内容都当成行号，返回对应的内容
        return self.embedding_matrix[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 eps: float = 1e-5, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input size: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        x_nrom = x / rms_x
        result = einsum(x_nrom, self.g, "... d_model, d_model -> ... d_model")

        return result.to(in_dtype)


def SiLU(in_features: torch.Tensor) -> torch.Tensor:
    result = in_features * torch.sigmoid(in_features)
    return result


class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):    
        super().__init__()   
        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        silu = SiLU(self.W1(x))
        gate = self.W3(x)
        result = self.W2(silu * gate)
        return result


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None=None):
        super().__init__()


        k = torch.arange(1, d_k // 2 + 1, device=device) # 1, 2, ... , d/2
        freqs= 1.0 / (theta**((2*k-2)/d_k))  # d/2,
        position = torch.arange(max_seq_len, device=device)
        angles = einsum(freqs, position, "f, p -> p f")

        self.register_buffer("cos_buffer", torch.cos(angles), persistent=False)
        self.register_buffer("sin_buffer", torch.sin(angles), persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # the size of x is ... sequence_len d_m
        x_even = x[..., 0::2] 
        x_odd = x[..., 1::2]

        cos = self.cos_buffer[token_positions]
        sin = self.sin_buffer[token_positions]

        x_even_new = x_even * cos - x_odd * sin
        x_odd_new  = x_even * sin + x_odd * cos

        result = torch.stack([x_even_new, x_odd_new], dim=-1)
        result = result.reshape(x.shape)
        return result


def softmax(v: torch.Tensor, dim: int) -> torch.Tensor:

    max_val = torch.max(v, dim=dim, keepdim=True)
    v_submax = v - max_val.values
    v_exp = torch.exp(v_submax)
    v_sum = torch.sum(v_exp, dim=dim, keepdim=True)
    result = v_exp / v_sum
    return result


def dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    # Q: ... n d_k
    # K: ... m d_k
    # 其中 n 和 m 代表了 sequence length，当进行 self attention 机制的时候二者的大小就是一样的
    softmax_content = einsum(Q, K, "... n dk, ... m dk -> ... n m")/(Q.shape[-1] ** 0.5)
    if mask is not None:
        softmax_content = softmax_content.masked_fill(~mask, -torch.inf)
    softmax_result = softmax(softmax_content, -1)
    result = einsum(softmax_result, V, "... n m, ... m d -> ... n d")
    return result
    

class multihead_self_attention(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int, 
                 theta: float | None = None, 
                 max_seq_len: int = 0, 
                 token_positions: torch.Tensor | None = None, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None
                ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.device = device
        self.dtype = dtype
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.token_positions = token_positions
        self.q_proj_weight = Linear(d_model, d_model, device, dtype)
        self.k_proj_weight = Linear(d_model, d_model, device, dtype)
        self.v_proj_weight = Linear(d_model, d_model, device, dtype)
        self.o_proj_weight = Linear(d_model, d_model, device, dtype)
        if self.theta is not None:
            self.position_embedding = RoPE(self.theta, d_model//num_heads, self.max_seq_len, self.device)
        # 因为对于输入的矩阵按照了注意力头进行拆分，因此在位置编码时每个矩阵的向量大小也发生了改变，变为 d_model/num_heads


    def forward(self,
                in_features: torch.Tensor) -> torch.Tensor:
        Q_in = self.q_proj_weight(in_features)
        K_in = self.k_proj_weight(in_features)
        V_in = self.v_proj_weight(in_features)

        d_k = Q_in.shape[-1]
        Q_in = rearrange(Q_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)
        K_in = rearrange(K_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)
        V_in = rearrange(V_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)

        if self.theta is not None:
            seq_len = Q_in.shape[-2]
            if self.token_positions is None:
                self.token_positions = torch.arange(seq_len, device=in_features.device)
            Q_in = self.position_embedding(Q_in, self.token_positions)
            K_in = self.position_embedding(K_in, self.token_positions)

        seq_len = Q_in.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q_in.device))
        attention_result = dot_product_attention(Q_in, K_in, V_in, mask)
        attention_result = rearrange(attention_result, "... head seqlen dv -> ... seqlen (head dv)")
        result = self.o_proj_weight(attention_result)

        return result


class Transformer_Block(nn.Module):
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 max_seq_len: int,
                 theta: float,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.preNorm_block_one = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.preNorm_block_two = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.self_attention_block = multihead_self_attention(d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype)
        self.feedforward_block = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        preNorm_one = self.preNorm_block_one(in_features)
        self_attention_result = self.self_attention_block(preNorm_one)
        residual_one = self_attention_result + in_features

        preNorm_two = self.preNorm_block_two(residual_one)
        ffn_result = self.feedforward_block(preNorm_two)
        result = ffn_result + residual_one
        return result
    

class Transformer_LM(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float,
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_layer = Embedding(vocab_size, d_model, device, dtype)
        self.layers = nn.ModuleList([
            Transformer_Block(d_model, num_heads, d_ff, context_length, rope_theta, device, dtype)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.linear = Linear(d_model, vocab_size, device, dtype)

    def forward(self, in_indices: torch.Tensor) -> torch.Tensor:
        block_result = self.embedding_layer(in_indices)
        for layer in self.layers:
            block_result = layer(block_result)
        norm_result = self.norm(block_result)
        result = self.linear(norm_result)

        return result
        

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    batch_size = inputs.shape[0]
    max_entry = torch.max(inputs, dim=-1, keepdim=True)
    inputs_submax = inputs - max_entry.values
    inputs_exp = torch.exp(inputs_submax)
    inputs_sum = torch.sum(inputs_exp, dim=-1, keepdim=True)
    result = torch.log(inputs_sum) - inputs_submax[torch.arange(batch_size), targets]
    # -log(exp(target) / sum_exp(i)) = log(sum_exp(i)) - target
    result = result.mean()

    return result


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps=1e-8): 
        if lr < 0: raise ValueError(f"Invalid learning rate: {lr}") 
        # defaults 仅仅存储人为配置，不随着训练改变的参数
        defaults = {"lr": lr, 
                    "beta1": betas[0], 
                    "beta2": betas[1], 
                    "epsilon":eps,
                    "weight_decay": weight_decay
                    } 
        super().__init__(params, defaults)  

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # 参数可以分组存储，每一组都可以用不同的学习率参数去训练
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            epsilon = group["epsilon"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: continue
                state = self.state[p]
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 1
                t = state.get("t")
                m = state.get("m")
                v = state.get("v") 

                # 由于这一步涉及了原有的数据，因此要先计算这一步，然后再计算其他的
                p.data -= lr * weight_decay * p.data

                grad = p.grad.data 
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                alpha_t = lr * (math.sqrt(1-beta2**t)/(1-beta1**t))
                p.data -= alpha_t * m / (torch.sqrt(v) + epsilon)
                

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


def learning_rate_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        alpha_t = it/warmup_iters*max_learning_rate
    if it >= warmup_iters and it <= cosine_cycle_iters:
        alpha_t = min_learning_rate + 1/2*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    if it > cosine_cycle_iters:
        alpha_t = min_learning_rate

    return alpha_t


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total = 0
    for p in parameters:
        if p.grad is None: continue
        total += (p.grad ** 2).sum()

    total = math.sqrt(total)

    if total > max_l2_norm:
        scale_ratio = max_l2_norm / (total + 1e-6)
        for p in parameters:
            if p.grad is None: continue
            p.grad *= scale_ratio
    

def data_loading(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = len(dataset) - context_length

    ix = torch.randint(0, max_len, size=(batch_size,))
    ix.tolist()

    x_list = [dataset[i: i+context_length] for i in ix]
    y_list = [dataset[i+1 : i+context_length+1] for i in ix]

    x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)   
    y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)    

    return x, y                   


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration

