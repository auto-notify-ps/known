
import numpy as np
import torch as tt
import torch.nn as nn
from torch.nn import functional as ff
from math import log

__all__ = ['Pose', 'Norm','Vocab',]


class Pose:

    class FixedSinusoidal(nn.Module):

        def __init__(self, input_size: int, block_size: int, dim_constant:float=1e4, dtype=None, device=None):
            super().__init__()
            self.factory = dict(dtype=dtype, device=device) 
            self.input_size, self.block_size = input_size, block_size
            # self.dropout = nn.Dropout(p=dropout)  # we do not need dropout in embedding
            position = tt.arange(block_size).unsqueeze(1)
            div_term = tt.exp(tt.arange(0, input_size, 2) * (-log(dim_constant)/ input_size))
            embedding = tt.zeros((block_size, 1, input_size), dtype=dtype)
            embedding[:, 0, 0::2] = tt.sin(position * div_term)
            embedding[:, 0, 1::2] = tt.cos(position * div_term)
            embedding.swapaxes_(1,0)
            embedding = embedding.to(**self.factory)
            self.register_buffer('embedding', embedding) #<--- optinal

        def forward(self, x): return x + self.embedding
        
    class TrainableLinear(nn.Module):

        def __init__(self, input_size: int, block_size: int, dtype=None, device=None ):
            super().__init__()
            self.factory = dict(dtype=dtype, device=device)
            self.input_size, self.block_size = input_size, block_size
            # self.dropout = nn.Dropout(p=dropout)  # we do not need dropout in embedding
            self.embedding = nn.Embedding(block_size, input_size, **self.factory)
            self.position = tt.arange(0, block_size, 1, dtype=tt.int32, device=device) #<-- no need to unsqueeze, will broadcast
            #NOTE: call self.embedding(self.position) every time because the embedding weights get trained

        def forward(self, x): return x +  self.embedding(self.position)


class Norm:

    class _Norm(nn.Module):

        def __init__(self, embed_dim, bias=True, eps=1e-6, dtype=None, device=None):
            super().__init__()
            self.factory = dict(dtype=dtype, device=device)
            self.embed_dim=embed_dim
            self.eps = eps
            self.weight = nn.Parameter(tt.ones(self.embed_dim, **self.factory))
            self.bias = nn.Parameter(tt.zeros(self.embed_dim, **self.factory)) if bias else None 

    class rmsN(_Norm):

        def __init__(self, embed_dim, bias=True, eps=1e-6, dtype=None, device=None):
            super().__init__(embed_dim, bias, eps, dtype, device)
            if self.bias is None: self.bias = tt.tensor(0, requires_grad=False, **self.factory)
        def _norm(self, x): return x * tt.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        def forward(self, x): return self._norm(x.float()).type_as(x) * self.weight + self.bias

    class layerN(_Norm):

        def forward(self, x):  return ff.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class Vocab:

    @staticmethod
    def random_string(vocab, length): return vocab.backward(tt.randint(0, vocab.size, size=(length,)))

    @staticmethod
    def random_string_excluding(vocab, length, exclude): 
        selector = np.arange(vocab.size)
        mask = np.zeros((len(selector,)), dtype=np.int8)
        mask[exclude]=1
        choice = np.random.choice(np.where(mask==0)[0], size=length, replace=True)
        return vocab.backward(choice)


    @staticmethod
    def random_string_including(vocab, length, include): 
        selector = np.arange(vocab.size)
        mask = np.ones((len(selector,)), dtype=np.int8)
        mask[include]=0
        choice = np.random.choice(np.where(mask==0)[0], size=length, replace=True)
        return vocab.backward(choice)

    @staticmethod
    def random_string_including_starting(vocab, length, include, start): 
        selector = np.arange(vocab.size)
        mask = np.ones((len(selector,)), dtype=np.int8)
        mask[include]=0
        choice = np.hstack(( np.array([start]) , np.random.choice(np.where(mask==0)[0], size=length-1, replace=True)))
        return vocab.backward(choice)

    @staticmethod
    def random_string_excluding_starting(vocab, length, exclude, start): 
        selector = np.arange(vocab.size)
        mask = np.zeros((len(selector,)), dtype=np.int8)
        mask[exclude]=1
        choice = np.hstack(( np.array([start]) , np.random.choice(np.where(mask==0)[0], size=length-1, replace=True)))
        return vocab.backward(choice)
    
    class IntVocab:
        r"""
        maps symbols to tokens
            symbols can be any datatype <--- will be converted to int
            tokens are tt.long or integer type
        """

        # special symbols
        UKN, BOS, EOS, PAD  = -1, -2, -3, -4

        def __init__(self, symbols) -> None:
            self.vocab = {} # value v/s symbol
            self.vocab[self.UKN] =  0
            self.vocab[self.BOS] =  1
            self.vocab[self.EOS] =  2
            self.vocab[self.PAD] =  3
            for i,symbol in enumerate(symbols): self.vocab[int(symbol)] = i + 4 #<-- offset
            self.rvocab = list(self.vocab.keys())
            self.size = len(self.rvocab)

        def __len__(self): return self.size

        
        # inplcae forward
        def forward_(self, symbols, dest):
            for i,symbol in enumerate(symbols): dest[i] = self.vocab[int(symbol)]
        def forward1_(self, symbol, dest, i): dest[i] = self.vocab[int(symbol)]
        
        # forward converts symbol to token
        def forward(self, symbols): return [self.vocab[int(symbol)] for symbol in symbols]
        def forward1(self, symbol):  return self.vocab[int(symbol)]

        # backward converts token to symbol
        def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
        def backward1(self, token): return self.rvocab[int(token)]

    class FloatVocab:
        r"""
        maps symbols to tokens
            symbols can be any datatype <--- will be converted to float
            tokens are tt.long or integer type
        """

        # special symbols
        UKN, BOS, EOS, PAD  = -1., -2., -3., -4.

        def __init__(self, symbols) -> None:
            self.vocab = {} # value v/s symbol
            self.vocab[self.UKN] =  0
            self.vocab[self.BOS] =  1
            self.vocab[self.EOS] =  2
            self.vocab[self.PAD] =  3
            for i,symbol in enumerate(symbols): self.vocab[float(symbol)] = i + 4 #<-- offset
            self.rvocab = list(self.vocab.keys())
            self.size = len(self.rvocab)

        def __len__(self): return self.size

        
        # inplcae forward
        def forward_(self, symbols, dest):
            for i,symbol in enumerate(symbols): dest[i] = self.vocab[float(symbol)]
        def forward1_(self, symbol, dest, i): dest[i] = self.vocab[float(symbol)]
        
        # forward converts symbol to token
        def forward(self, symbols): return [self.vocab[float(symbol)] for symbol in symbols]
        def forward1(self, symbol):  return self.vocab[float(symbol)]

        # backward converts token to symbol
        def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
        def backward1(self, token): return self.rvocab[int(token)]

    class StrVocab:
        r"""
        maps symbols to tokens
            symbols can be any datatype <--- will be converted to str
            tokens are tt.long or integer type
        """

        # special symbols
        UKN, BOS, EOS, PAD  = "<UKN>", "<BOS>", "<EOS>", "<PAD>"

        def __init__(self, symbols) -> None:
            self.vocab = {} # value v/s symbol
            self.vocab[self.UKN] =  0
            self.vocab[self.BOS] =  1
            self.vocab[self.EOS] =  2
            self.vocab[self.PAD] =  3
            
            for i,symbol in enumerate( symbols ): self.vocab[symbol] = i + 4 #<-- offset
            self.rvocab = list(self.vocab.keys())
            self.size = len(self.rvocab)

        def __len__(self): return self.size

        
        # inplcae forward
        def forward_(self, symbols, dest):
            for i,symbol in enumerate(symbols): dest[i] = self.vocab[symbol]
        def forward1_(self, symbol, dest, i): dest[i] = self.vocab[symbol]
        
        # forward converts symbol to token
        def forward(self, symbols): return [self.vocab[symbol] for symbol in symbols]
        def forward1(self, symbol):  return self.vocab[symbol]

        # backward converts token to symbol
        def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
        def backward1(self, token): return self.rvocab[int(token)]

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

