import torch.nn as nn
import torch
from einops import rearrange, einsum

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


class SwiGLU(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int, 
                 device: torch.device | None = None, 
                 dtype: torch.dtype | None = None):    
        super().__init__()   
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, device=device, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, device=device, dtype=dtype))
        std = (2 / (d_ff + d_model)) ** 0.5
        nn.init.trunc_normal_(self.W1, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.W2, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.W3, mean=0, std=std, a=-3*std, b=3*std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mid_1 = einsum(self.W1, x, "d_ff d_model, ... d_model -> ... d_ff")
        mid_2 = torch.sigmoid(mid_1)
        SiLU = mid_1 * mid_2
        gate = einsum(self.W3, x, "d_ff d_model, ... d_model -> ... d_ff")
        result = einsum(self.W2, SiLU * gate, "d_model d_ff, ... d_ff -> ... d_model")

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
                 theta: float = 0.0, 
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


    def forward(self,
                 q_proj_weight: torch.Tensor,
                 k_proj_weight: torch.Tensor,
                 v_proj_weight: torch.Tensor,
                 o_proj_weight: torch.Tensor,
                 in_features: torch.Tensor) -> torch.Tensor:
        Q_in = einsum(q_proj_weight, in_features, "dk din, ... seqlen din -> ... seqlen dk")
        K_in = einsum(k_proj_weight, in_features, "dk din, ... seqlen din -> ... seqlen dk")
        V_in = einsum(v_proj_weight, in_features, "dk din, ... seqlen din -> ... seqlen dk")

        d_k = Q_in.shape[-1]
        Q_in = rearrange(Q_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)
        K_in = rearrange(K_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)
        V_in = rearrange(V_in, "... seqlen (head dv) -> ... head seqlen dv", head=self.num_heads, dv=d_k//self.num_heads)

        if self.token_positions is not None:
            position_embedding = RoPE(self.theta, Q_in.shape[-1], self.max_seq_len, self.device)
            Q_in = position_embedding.forward(Q_in, self.token_positions)
            K_in = position_embedding.forward(K_in, self.token_positions)

        seq_len = Q_in.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q_in.device))
        attention_result = dot_product_attention(Q_in, K_in, V_in, mask)
        attention_result = rearrange(attention_result, "... head seqlen dv -> ... seqlen (head dv)")
        result = einsum(attention_result, o_proj_weight, "... seqlen dk, dm dk -> ... seqlen dm")

        return result


    
