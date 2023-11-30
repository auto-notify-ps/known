
__all__ = [
     'LearnableWeightedScoring', 'LearnableScoring', 'LearnableNoDotScoring', 'DotProdScoring',
     'causal_mask', 'Attention', 'SelfAttention', 'CrossAttention',
]

import torch as tt
import torch.nn as nn
#import matplotlib.pyplot as plt
#from torch.nn import functional as F


class LearnableWeightedScoring(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True,  dtype=None, device=None) -> None:
        super().__init__()
        assert embed_dim%num_heads==0, f'embed_dim should be divisible by num_heads'
        self.factory = dict(dtype=dtype, device=device)
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.head_dim = self.embed_dim//self.num_heads
        self.scale =  (embed_dim**0.5) #if scale is None else scale
        self.use_bias = bias
        #self.block_size = block_size
        self.setup_weights()

    def setup_weights(self):
        self.Ws = nn.ModuleList( [nn.Bilinear(in1_features=self.head_dim, in2_features=self.head_dim, out_features=1, bias=self.use_bias, **self.factory) for _ in range(self.num_heads)] )
        self.Wd = nn.ParameterList( [nn.Parameter(tt.tensor(0.5, **self.factory), ) for _ in range(self.num_heads)] )
        
    def forward(self, q, k): #  (B, nh, T, hs) 
        
        d = (q @ k.transpose(-2, -1)) * self.scale
        assert d.shape[-1]==d.shape[-2]
        block_size = d.shape[-1]
        for h in range(self.num_heads):
            for i in range(block_size):
                for j in range(block_size): 
                    ws = self.Ws[h].forward(q[:, h, i, :], k[:, h, j, :]) 
                    d[:, h, i, j] += ws.squeeze(-1) * self.Wd[h]
        return d

class LearnableScoring(LearnableWeightedScoring):
    def setup_weights(self): 
        self.Ws = nn.ModuleList( [nn.Bilinear(in1_features=self.head_dim, in2_features=self.head_dim, out_features=1, bias=self.use_bias, **self.factory) for _ in range(self.num_heads)] )
        self.Wd = tt.tensor(1.0, **self.factory)

class LearnableNoDotScoring(LearnableWeightedScoring):
    def setup_weights(self): 
        self.Ws = nn.ModuleList( [nn.Bilinear(in1_features=self.head_dim, in2_features=self.head_dim, out_features=1, bias=self.use_bias, **self.factory) for _ in range(self.num_heads)] )
    
    def forward(self, q, k): #  (B, nh, T, hs) 
        batch_size, _, block_size, _ = q.shape
        d = tt.zeros((batch_size,self.num_heads,block_size,block_size), **self.factory)
        for h in range(self.num_heads):
            for i in range(block_size):
                for j in range(block_size): 
                    ws = self.Ws[h].forward(q[:, h, i, :], k[:, h, j, :]) 
                    d[:, h, i, j] += ws.squeeze(-1)
        return d
    
class DotProdScoring(LearnableWeightedScoring):
    def setup_weights(self): pass
    def forward(self, q, k): #  (B, nh, T, hs)
        return (q @ k.transpose(-2, -1)) * self.scale
        
    
@staticmethod
def causal_mask(block_size, device=None): 
    return tt.triu(tt.ones((block_size, block_size), dtype=tt.bool, device=device), diagonal=1).view(1, 1, block_size, block_size)

class Attention(nn.Module):

    def _init_residual_connection_parameters(self, num_layers): 
        tt.nn.init.normal_(self.P.weight, mean=0.0, std=0.02/(2 * num_layers)**0.5)
        tt.nn.init.normal_(self.P.bias, mean=0.0, std=0.02/(2 * num_layers)**0.5)

    def __init__(self, embed_dim, num_heads, score2F, dropout=0.0, bias=True, num_layers=None, dtype=None, device=None):
        super().__init__()
        assert embed_dim % num_heads == 0, f'Requires {num_heads=} to be divisible by {embed_dim=}'
        self.factory = dict(dtype=dtype, device=device)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        #self.block_size = block_size
        self.head_dim = self.embed_dim // self.num_heads
        
        self.Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.K = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.V = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # output projection
        self.P = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        # regularization
        self.ADropout = nn.Dropout(dropout) # attention dropout
        self.RDropout = nn.Dropout(dropout) # residual dropout
        self.scoreF = score2F[0](embed_dim=self.embed_dim, num_heads=self.num_heads, bias=bias, **score2F[1])

        if num_layers: self._init_residual_connection_parameters(num_layers)
        #self.do_store_attention(False)

    #def do_store_attention(self, store): self.store_attention = store

    def forward(self,             
            Q,
            K,
            V,
            key_padding_mask = None,
            need_weights = True,
            attn_mask = None,
            average_attn_weights = False,
            is_causal  = False):
        is_batched = Q.ndim>2
        if not is_batched: Q, K, V = Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
        B, T, C = Q.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q, k, v = \
            self.Q(Q).view(B, T, self.num_heads, self.head_dim).transpose(1, 2), \
            self.K(K).view(B, T, self.num_heads, self.head_dim).transpose(1, 2), \
            self.V(V).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v  (B, nh, T, hs)

        a = self.scoreF(q, k)
        if attn_mask is not None: a = a.masked_fill(attn_mask, -tt.inf)
        a = tt.softmax(a, dim=-1)
        #print(f'{a.shape=}')
        #if self.store_attention: self.attention_weights = a
        a = self.ADropout(a)
        y = a @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.RDropout(self.P(y))
        if not is_batched: y = y.squeeze(0)
        return y, a

class SelfAttention(Attention):
    def forward(self, S, mask=None): return super().forward(S, S, S, attn_mask=mask)

class CrossAttention(Attention):
    def forward(self, S, E, mask=None): return super().forward(S, E, E, attn_mask=mask)




# class SHA:

#     class DotProdScoreing(nn.Module):
#         def __init__(self, scale) -> None: 
#             super().__init__()
#             self.scale = scale
#         def forward(self, q, k): return (q @ k.transpose(-2, -1)) * self.scale

#     class LearnableScoring(nn.Module):
#         def __init__(self, embed_dim, block_size, num_heads, scale, bias=True, dtype=None, device=None) -> None:
#             super().__init__()
            
#             self.factory = dict(dtype=dtype, device=device)
#             self.embed_dim=embed_dim
#             self.scale =  (embed_dim**0.5) if scale is None else scale
#             self.block_size = block_size
#             self.Ws = nn.Bilinear(in1_features=self.embed_dim, in2_features=self.embed_dim, out_features=1, bias=bias, **self.factory) 
#             #self.Wd = nn.ParameterList( [nn.Parameter(tt.tensor(0.5, **self.factory)) for _ in range(num_heads)] )

#         def forward(self, q, k):

#             # # (B, nh, T, hs) # B T C
#             d = (q @ k.transpose(-2, -1)) * self.scale
#             for i in range(self.block_size):
#                 for j in range(self.block_size): 
#                     ws = self.Ws.forward(q[:, i, :], k[:, j, :]) 
#                     d[:, i, j] += ws.squeeze(-1)
#             return d

#     @staticmethod
#     def causal_mask(block_size, device=None): 
#         return tt.triu(tt.ones((block_size, block_size), dtype=tt.bool, device=device), diagonal=1).view(1, block_size, block_size)

#     class Attention(nn.Module):

#         def _init_residual_connection_parameters(self, num_layers): 
#                 tt.nn.init.normal_(self.P.weight, mean=0.0, std=0.02/(2 * num_layers)**0.5)
#                 tt.nn.init.normal_(self.P.bias, mean=0.0, std=0.02/(2 * num_layers)**0.5)

#         def __init__(self, embed_dim, block_size, num_heads, scoreF, dropout=0.0, bias=True, num_layers=None, dtype=None, device=None):
#             super().__init__()
#             self.factory = dict(dtype=dtype, device=device)

#             self.embed_dim = embed_dim
#             self.block_size = block_size

#             self.Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
#             self.K = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
#             self.V = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
#             # output projection
#             self.P = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
#             # regularization
#             self.ADropout = nn.Dropout(dropout) # attention dropout
#             self.RDropout = nn.Dropout(dropout) # residual dropout
#             self.scoreF = scoreF

#             if num_layers: self._init_residual_connection_parameters(num_layers)
#             self.do_store_attention(False)

#         def do_store_attention(self, store): self.store_attention = store

#         def forward(self, Q, K, V, mask=None):
#             is_batched = Q.ndim>2
#             if not is_batched: Q, K, V = Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
#             #B, T, C = Q.size() # batch size, sequence length, embedding dimensionality (n_embd)
#             q, k, v = self.Q(Q), self.K(K), self.V(V)
#             #print(f'{q.shape=}, {k.shape=}, {v.shape}')
#             #print(f'{q=}, {k=}, {v}')
#             # k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
#             # q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
#             # v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)

#             a = self.scoreF(q, k)
#             if mask is not None: a = a.masked_fill(mask, -tt.inf)
#             a = tt.softmax(a, dim=-1)
#             #print(f'{a.shape=}')
#             if self.store_attention: self.attention_weights = a
#             a = self.ADropout(a)
#             y =  a @ v # (B, T, T) x (B, T, C) -> (B, T, C)
#             y = y.contiguous()
#             #y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#             # output projection
#             y = self.RDropout(self.P(y))
#             if not is_batched: y = y.squeeze(0)
#             return y

#     class SelfAttention(Attention):
#         def forward(self, S, mask=None): return super().forward(S, S, S, mask=mask)

#     class CrossAttention(Attention):
#         def forward(self, S, E, mask=None): return super().forward(S, E, E, mask=mask)
