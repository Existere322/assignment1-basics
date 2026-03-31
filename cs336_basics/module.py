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


