
"""NOTE:

M. Gers et al. (2000) proposed initializing the forget gate biases to positive values and 
    Jozefowicz et al. (2015) showed that an initial bias of 1 for the LSTM forget gate makes the LSTM 
    as strong as the best of the explored architectural variants (including the GRU) 
    (Goodfellow et al., 2016, §10.10.2)

Recurrent neural networks (RNNs) typically create a lossy summary hT of a sequence. It is lossy
    because it maps an arbitrarily long sequence into a fixed length vector. As mentioned before,
    recent work has shown that this forgetting property of LSTMs is one of the most important (Greff
    et al., 2015; Jozefowicz et al., 2015).

The standard procedure is to initialize the weights U∗ and W∗ to be distributed as
    U ~ +/- [ (6 / (n(l) + n(l+1)) )**0.5 ] 
    where n(l) is the size of each layer l (He et al., 2015b; Glorot and Bengio, 2010), 
    and to initialize all biases to zero except for the forget gate bias bf , which is initialized to one
    (Jozefowicz et al., 2015).
>>> This implementation uses uniform initialization, to add custom initialization use torch.nn.init
>>>     Weight and Bias may be initialized used seperate methonds
>>>     Can implement custom initializers like chrono as required by JANET




NOTE: [RNNCell vs RNN vs RNNStack vs StackedRNNCell vs StackedRNN]:

    RNNCell : Implements RNN Cell
        ~ The `forward` methods implements forward-pass through one timestep only.
        ~ Excpected input of shape (batch_size, input_size)

    RNN : Implements RNN Cell over a sequence 
        ~ The `forward` methods implements forward-pass through multiple timesteps.
        ~ Excpected input of shape (batch_size, seq_len, input_size) or (seq_len, batch_size, input_size)
        ~ Internally, inherited from RNNCell (implements `batch_first` and `forward` only)

    RNNStack : Implements Layers of RNN over a sequence
        ~ The `forward` methods implements forward-pass through multiple timesteps.
        ~ Excpected input of shape (batch_size, seq_len, input_size) or (seq_len, batch_size, input_size)
        ~ Implments forward pass 'horizontally' i.e. computes all timesteps at a layer before going to next layer
        ~ Internally, implements a `nn.ModuleList` of RNN

    (depreciated) StackedRNNCell : Implements Layers of RNN Cells
        ~ The `forward` methods implements forward-pass through one timestep only.
        ~ Excpected input of shape (batch_size, input_size)

    StackedRNN : Implements Layers of RNN Cells over a sequence    
        ~ The `forward` methods implements forward-pass through multiple timesteps.
        ~ Excpected input of shape (batch_size, seq_len, input_size) or (seq_len, batch_size, input_size)
        ~ Implments forward pass 'vertically' i.e. computes each timestep at all layers before going to next timestep
        ~ Internally, inherited from StackedRNNCell (implements `batch_first` and `forward` only)


NOTE: Arguments to __init__ 

    RNNCell and RNN have same args, only one extra arg is added to RNN i.e.
        batch_first:    
            if True,expects input of shape (batch_size, seq_len, input_size) 
            if False, expects input of shape (seq_len, batch_size, input_size)

    RNN and RNNStack have same args, except `hidden_size` v/s `hidden_sizes`
        `hidden_sizes` expects a `list/tuple of ints` for `hidden_size` at each layer
        this also determines the number of layers in the stack

    (depeciated) RNNCell and StackedRNNCell have same args, except `hidden_size` v/s `hidden_sizes`
        `hidden_sizes` expects a `list/tuple of ints` for `hidden_size` at each layer
        this also determines the number of layers in the stack


    StackedRNN and RNNStack have same args


NOTE: Bias in Linear layers: 
    In the orignal RNN, bias is not required for hidden state, 
    `bias` args enables bias for hidden state only

NOTE: Dropout is applied to output of each hidden layer except the last layer in a stack
"""

# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =
import torch as tt
import torch.nn as nn
import math
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Elman or Vanilla RNN
# Ref :ref:`https://pytorch.org/docs/stable/generated/torch.nn.RNN.html`
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

class ELMANCell(nn.Module):

    def __init__(self, input_size, hidden_size, nlF, nlFA={}, bias=True, device=None, dtype=None, do_init=True):
        r"""
        Args:
            input_size      `integer`       : in_features or input_size
            hidden_size     `integer`       : hidden_features or hidden_size
            nlF             `nn.Module`     : non-linear activation  - usually `nn.Tanh`
            nlFA            `dict`          : args while initializing nlF - usually {}
            bias            `bool`          : if True, uses Bias at linear layers for hidden state
            do_init         `bool`          : if True, calls `reset_parameters()`
        """
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.has_bias = bias
        self.ih = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hh = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.nl = nlF(**nlFA)
        if do_init: self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for w in self.parameters():
           nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def forward(self, x, h=None):
        if h is None: h=self.init_hidden(x.shape[0], x.dtype, x.device)
        return self.nl(self.ih(x) + self.hh(h))

class ELMAN(ELMANCell):

    def __init__(self, input_size, hidden_size, nlF, nlFA={}, bias=True, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__(input_size, hidden_size, nlF, nlFA, bias, device, dtype, do_init)
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def forward(self, X, h=None):
        if h is None: h=self.init_hidden(X.shape[self.batch_dim], X.dtype, X.device)
        seq = [h]
        for x in tt.split(X, 1, dim=self.seq_dim): seq.append(self.nl(self.ih(x.squeeze(dim=self.seq_dim)) + self.hh(seq[-1])))
        out = tt.stack(seq[1:], dim=self.seq_dim)
        return out, seq[-1]

class ELMANStack(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size, self.hidden_sizes = input_size, tuple(hidden_sizes)
        self.dropout = dropout

        L = (self.input_size,) + self.hidden_sizes
        self.model = nn.ModuleList([ ELMAN(
                input_size=L[i-1],
                hidden_size=L[i],
                nlF=nlF,
                nlFA=nlFA,
                bias=bias,
                device=device,
                dtype=dtype,
                do_init=do_init,
                batch_first=batch_first
            ) for i in range(1, len(L)) ] )
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] )  
        self.no_hidden = [ None for _ in range(len(self.model)) ]
        
    def forward(self, x, h=None):
        H = []
        if h is None: h=self.no_hidden
        for i,layer in enumerate(self.model):
            x, lh = layer(x, h[i])
            x = self.dropouts[i](x) #<--- dropout only output
            H.append(lh)
        return x, tt.stack(H)

class StackedELMAN(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size=input_size
        self.hidden_sizes=tuple(hidden_sizes)
        self.has_bias = bias
        self.dropout = dropout

        self.layer_sizes = (self.input_size,) + self.hidden_sizes
        self.ihL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hhL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.nlL = [ nlF(**nlFA)  for i in range(1, len(self.layer_sizes))  ]
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        if do_init: self.reset_parameters()
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def reset_parameters(self):
        for i,hs in enumerate(self.hidden_sizes):
            stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
            for ww in zip(self.ihL[i].parameters(), self.hhL[i].parameters()):
                for w in ww: nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return [ tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes  ]

    def forward(self, Xt, h=None):
        if h is None: h=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        seq=[h]
        seq_last = []
        for xt in tt.split(Xt, 1, dim=self.seq_dim): 
            H = []
            x = xt.squeeze(dim=self.seq_dim)
            for i in range(len(self.hidden_sizes)):
                x = self.nlL[i](self.ihL[i](x) + self.hhL[i](seq[-1][i]))
                H.append(x)
                x = self.dropouts[i](x) #<--- dropout only output
            seq_last.append(x)
            seq.append(H)
            
        out = tt.stack(seq_last, dim=self.seq_dim)
        return out, tt.stack(seq[-1])


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Gated Recuurent Unit GRU
# Ref :ref:`https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html`
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, nlF, nlFA={}, bias=True, device=None, dtype=None, do_init=True):
        r"""
        Returns a GRU RNNCell Module.

        Args:
            input_size      `integer`       : in_features or input_size
            hidden_size     `integer`       : hidden_features or hidden_size
            nlF             `nn.Module`     : non-linear activation  - usually `nn.Tanh`
            nlFA            `dict`          : args while initializing nlF - usually {}
            bias            `bool`          : if True, uses Bias at linear layers for hidden state
            do_init         `bool`          : if True, calls `reset_parameters()`
        """
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.has_bias = bias

        self.ir = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hr = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iz = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hz = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iN = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hN = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.nl = nlF(**nlFA)
        if do_init: self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for w in self.parameters():
           nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def forward(self, x, h=None):
        if h is None: h=self.init_hidden(x.shape[0], x.dtype, x.device)
        r = tt.sigmoid(self.ir(x) + self.hr(h))
        z = tt.sigmoid(self.iz(x) + self.hz(h))
        n = self.nl(self.iN(x) + r*self.hN(h))
        h_ = (1-z) * n + (z * h)
        return h_

class GRU(GRUCell):

    def __init__(self, input_size, hidden_size, nlF, nlFA={}, bias=True, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__(input_size, hidden_size, nlF, nlFA, bias, device, dtype, do_init)
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def forward(self, X, h=None):
        if h is None: h=self.init_hidden(X.shape[self.batch_dim], X.dtype, X.device)
        seq = [h]
        for xx in tt.split(X, 1, dim=self.seq_dim): 
            x, h = xx.squeeze(dim=self.seq_dim), seq[-1]
            r = tt.sigmoid(self.ir(x) + self.hr(h))
            z = tt.sigmoid(self.iz(x) + self.hz(h))
            n = self.nl(self.iN(x) + r*self.hN(h))
            h_ = (1-z) * n + (z * h)
            seq.append(h_)

        out = tt.stack(seq[1:], dim=self.seq_dim)
        return out, seq[-1]

class GRUStack(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size, self.hidden_sizes = input_size, tuple(hidden_sizes)
        self.dropout = dropout
        L = (self.input_size,) + self.hidden_sizes
        self.model = nn.ModuleList([ GRU(
                input_size=L[i-1],
                hidden_size=L[i],
                nlF=nlF,
                nlFA=nlFA,
                bias=bias,
                device=device,
                dtype=dtype,
                do_init=do_init,
                batch_first=batch_first
            ) for i in range(1, len(L)) ] )
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] )  
        self.no_hidden = [ None for _ in range(len(self.model)) ]
        
    def forward(self, x, h=None):
        H = []
        if h is None: h=self.no_hidden
        for i,layer in enumerate(self.model):
            x, lh = layer(x, h[i])
            x = self.dropouts[i](x) #<--- dropout only output
            H.append(lh)
        return x, tt.stack(H)

class StackedGRU(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size=input_size
        self.hidden_sizes=tuple(hidden_sizes)
        self.has_bias = bias
        self.dropout = dropout
        self.layer_sizes = (self.input_size,) + self.hidden_sizes

        self.irL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hrL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.izL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hzL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.iNL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hNL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.nlL = [ nlF(**nlFA)  for i in range(1, len(self.layer_sizes))  ]
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        if do_init: self.reset_parameters()
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)
    
    def reset_parameters(self):
        for i,hs in enumerate(self.hidden_sizes):
            stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
            for ww in zip(self.irL[i].parameters(), self.hrL[i].parameters(),self.izL[i].parameters(), self.hzL[i].parameters(),self.iNL[i].parameters(), self.hNL[i].parameters()):
                for w in ww: nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return [ tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes  ]

    def forward(self, Xt, h=None): 
        if h is None: h=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        seq=[h]
        seq_last = []
        for xt in tt.split(Xt, 1, dim=self.seq_dim): 
            H = [] 
            x,h = xt.squeeze(dim=self.seq_dim), seq[-1]
            for i in range(len(self.hidden_sizes)):
                r = tt.sigmoid(self.irL[i](x) + self.hrL[i](h[i]))
                z = tt.sigmoid(self.izL[i](x) + self.hzL[i](h[i]))
                n = self.nlL[i](self.iNL[i](x) + r*self.hNL[i](h[i]))
                x = (1-z) * n + (z * h[i])
                H.append(x)
                x = self.dropouts[i](x) #<--- dropout only output
            seq_last.append(x)
            seq.append(H)
            
        out = tt.stack(seq_last, dim=self.seq_dim)
        return out, tt.stack(seq[-1])


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Long Short Term Memory  LSTM
# Ref :ref:`https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html`
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, nlF, nlF2 ,nlFA={}, nlFA2={}, bias=True, device=None, dtype=None, do_init=True):
        r"""
        Args:
            input_size      `integer`       : in_features or input_size
            hidden_size     `integer`       : hidden_features or hidden_size
            nlF             `nn.Module`     : non-linear activation  - usually `nn.Tanh`
            nlFA            `dict`          : args while initializing nlF - usually {}
            nlF2            `nn.Module`     : non-linear activation at output  - usually `nn.Tanh`
            nlFA2           `dict`          : args while initializing nlF2 - usually {}
            bias            `bool`          : if True, uses Bias at linear layers for hidden state
            do_init         `bool`          : if True, calls `reset_parameters()`
        """
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.has_bias = bias

        self.iI = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hI = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iF = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hF = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iG = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hG = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iO = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hO = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.nl = nlF(**nlFA)
        self.nl2 = nlF2(**nlFA2)

        if do_init: self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for w in self.parameters():
           nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def init_cell(self, batch_size, dtype=None, device=None):
        return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def forward(self, x, h=None, c=None):
        if h is None: h=self.init_hidden(x.shape[0], x.dtype, x.device)
        if c is None: c=self.init_cell(x.shape[0], x.dtype, x.device)
        I = tt.sigmoid(self.iI(x) + self.hI(h))
        F = tt.sigmoid(self.iF(x) + self.hF(h))
        G = self.nl(self.iG(x) + self.hG(h))
        O = tt.sigmoid(self.iO(x) + self.hO(h))
        c_ = F*c + I*G
        h_ = O * self.nl2(c_)
        return h_, c_

class LSTM(LSTMCell):

    def __init__(self, input_size, hidden_size, nlF, nlF2 ,nlFA={}, nlFA2={}, bias=True, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__(input_size, hidden_size, nlF, nlF2, nlFA, nlFA2, bias, device, dtype, do_init)
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def forward(self, X, h=None, c=None): 
        if h is None: h=self.init_hidden(X.shape[self.batch_dim], X.dtype, X.device)
        if c is None: c=self.init_cell(X.shape[self.batch_dim], X.dtype, X.device)

        seq = [h]
        ceq = [c]
        for xx in tt.split(X, 1, dim=self.seq_dim): 
            x, h, c = xx.squeeze(dim=self.seq_dim), seq[-1], ceq[-1]
            I = tt.sigmoid(self.iI(x) + self.hI(h))
            F = tt.sigmoid(self.iF(x) + self.hF(h))
            G = self.nl(self.iG(x) + self.hG(h))
            O = tt.sigmoid(self.iO(x) + self.hO(h))
            c_ = F*c + I*G
            h_ = O * self.nl2(c_)
            seq.append(h_)
            ceq.append(c_)

        out = tt.stack(seq[1:], dim=self.seq_dim)
        #cst = tt.stack(ceq[1:], dim=self.seq_dim)
        return out, seq[-1], ceq[-1]

class LSTMStack(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlF2 ,nlFA={}, nlFA2={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size, self.hidden_sizes = input_size, tuple(hidden_sizes)
        self.dropout = dropout
        L = (self.input_size,) + self.hidden_sizes
        self.model = nn.ModuleList([ LSTM(
                input_size=L[i-1],
                hidden_size=L[i],
                nlF=nlF,
                nlF2=nlF2,
                nlFA=nlFA,
                nlFA2=nlFA2,
                bias=bias,
                device=device,
                dtype=dtype,
                do_init=do_init,
                batch_first=batch_first
            ) for i in range(1, len(L)) ] )
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        self.no_hidden = [ None for _ in range(len(self.model)) ]
        self.no_cell = [ None for _ in range(len(self.model)) ]
        
    def forward(self, x, h=None, c=None):
        H,C = [], []
        if h is None: h=self.no_hidden
        if c is None: c=self.no_cell
        for i,layer in enumerate(self.model):
            x, lh, lc = layer(x, h[i], c[i])
            x = self.dropouts[i](x) #<--- dropout only output
            H.append(lh)
            C.append(lc)
        return x, tt.stack(H), tt.stack(C)

class StackedLSTM(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlF2, nlFA={}, nlFA2={}, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size=input_size
        self.hidden_sizes=tuple(hidden_sizes)
        self.has_bias = bias
        self.dropout = dropout
        self.layer_sizes = (self.input_size,) + self.hidden_sizes

        self.iIL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hIL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.iFL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hFL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.iGL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hGL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.iOL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hOL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.nlL = [ nlF(**nlFA)  for i in range(1, len(self.layer_sizes))  ]
        self.nl2L = [ nlF2(**nlFA2)  for i in range(1, len(self.layer_sizes))  ]
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        if do_init: self.reset_parameters()
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def reset_parameters(self):
        for i,hs in enumerate(self.hidden_sizes):
            stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
            for ww in zip(self.iIL[i].parameters(), self.hIL[i].parameters(), self.iFL[i].parameters(), self.hFL[i].parameters(),
                            self.iGL[i].parameters(), self.hGL[i].parameters(), self.iOL[i].parameters(), self.hOL[i].parameters()):
                for w in ww: nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return [ tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes  ]

    def init_cell(self, batch_size, dtype=None, device=None):
        return [ tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes  ]

    def forward(self, Xt, h=None, c=None): 
        if h is None: h=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        if c is None: c=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        seq=[h]
        ceq=[c]
        seq_last = []
        ceq_last = []
        for xt in tt.split(Xt, 1, dim=self.seq_dim): 
            H,C = [],[]
            x,h,c = xt.squeeze(dim=self.seq_dim), seq[-1], ceq[-1]
            for i in range(len(self.hidden_sizes)):
                I = tt.sigmoid(self.iIL[i](x) + self.hIL[i](h[i]))
                F = tt.sigmoid(self.iFL[i](x) + self.hFL[i](h[i]))
                G = self.nlL[i](self.iGL[i](x) + self.hGL[i](h[i]))
                O = tt.sigmoid(self.iOL[i](x) + self.hOL[i](h[i]))
                c_ = F*c[i] + I*G
                x = O * self.nl2L[i](c_)
                H.append(x)
                C.append(c_)
                x = self.dropouts[i](x) #<--- dropout only output
            seq_last.append(x)
            ceq_last.append(c_)
            seq.append(H)
            ceq.append(C)
            
        out = tt.stack(seq_last, dim=self.seq_dim)
        return out, tt.stack(seq[-1]), tt.stack(ceq[-1])



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Just Another NETwork  JANET : The unreasonable effectiveness of the forget gate
# Ref :ref:`https://arxiv.org/abs/1804.04849` 
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

"""NOTE:
Quote Authors
    " We speculate that the value for `beta` is dataset dependent, however, we found that setting `beta` = 1 provides
    the best results for the datasets analysed in this study, which have sequence lengths varying from 200 to 784. "

"""

class JANETCell(nn.Module):

    def __init__(self, input_size, hidden_size, nlF, nlFA={}, beta=0.0, bias=True, device=None, dtype=None, do_init=True):
        r"""
        Args:
            input_size      `integer`       : in_features or input_size
            hidden_size     `integer`       : hidden_features or hidden_size
            nlF             `nn.Module`     : non-linear activation  - usually `nn.Tanh`
            nlFA            `dict`          : args while initializing nlF - usually {}
            beta            `float`         : hyperparameter
            bias            `bool`          : if True, uses Bias at linear layers for hidden state
            do_init         `bool`          : if True, calls `reset_parameters()`
        """
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.has_bias = bias
        self.beta = beta

        self.iF = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hF = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.iG = nn.Linear(input_size, hidden_size, True, device=device, dtype=dtype )
        self.hG = nn.Linear(hidden_size, hidden_size, bias, device=device, dtype=dtype )
        self.nl = nlF(**nlFA)

        if do_init: self.reset_parameters()
    
    def reset_parameters(self) :
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for w in self.parameters():
           nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def forward(self, x, h=None): 
        if h is None: h=self.init_hidden(x.shape[0], x.dtype, x.device)

        F = tt.sigmoid(self.iF(x) + self.hF(h) - self.beta)
        G = self.nl(self.iG(x) + self.hG(h))
        h_ = F*h + (1-F)*G

        return h_

class JANET(JANETCell):
    
    def __init__(self, input_size, hidden_size, nlF, nlFA={}, beta=0.0, bias=True, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__(input_size, hidden_size, nlF, nlFA, beta, bias, device, dtype, do_init)
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def forward(self, X, h=None): 
        if h is None: h=self.init_hidden(X.shape[self.batch_dim], X.dtype, X.device)
        seq = [h]
        for xx in tt.split(X, 1, dim=self.seq_dim): 
            x, h = xx.squeeze(dim=self.seq_dim), seq[-1]

            F = tt.sigmoid(self.iF(x) + self.hF(h) - self.beta)
            G = self.nl(self.iG(x) + self.hG(h))
            h_ = F*h + (1-F)*G

            seq.append(h_)

        out = tt.stack(seq[1:], dim=self.seq_dim)
        return out, seq[-1]
            
class JANETStack(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, beta=0.0, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size, self.hidden_sizes = input_size, tuple(hidden_sizes)
        self.dropout = dropout
        L = (self.input_size,) + self.hidden_sizes
        self.model = nn.ModuleList([ JANET(
                input_size=L[i-1],
                hidden_size=L[i],
                nlF=nlF,
                nlFA=nlFA,
                beta=beta,
                bias=bias,
                device=device,
                dtype=dtype,
                do_init=do_init,
                batch_first=batch_first
            ) for i in range(1, len(L)) ] )
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        self.no_hidden = [ None for _ in range(len(self.model)) ]
        
    def forward(self, x, h=None):
        H = []
        if h is None: h=self.no_hidden
        for i,layer in enumerate(self.model):
            x, lh = layer(x, h[i])
            x = self.dropouts[i](x) #<--- dropout only output
            H.append(lh)
        return x, tt.stack(H)

class StackedJANET(nn.Module):

    def __init__(self, input_size, hidden_sizes, nlF, nlFA={}, beta=0.0, bias=True, dropout=0.0, device=None, dtype=None, do_init=True, batch_first=True):
        super().__init__()
        self.input_size=input_size
        self.hidden_sizes=tuple(hidden_sizes)
        self.has_bias = bias
        self.beta = beta
        self.dropout = dropout
        self.layer_sizes = (self.input_size,) + self.hidden_sizes

        self.iFL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hFL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.iGL = nn.ModuleList([ nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], True, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.hGL = nn.ModuleList([ nn.Linear(self.layer_sizes[i], self.layer_sizes[i], bias, device=device, dtype=dtype ) for i in range(1, len(self.layer_sizes))  ])
        self.nlL = [ nlF(**nlFA)  for i in range(1, len(self.layer_sizes))  ]
        self.dropouts = nn.ModuleList( [ nn.Dropout(dropout) for _ in range(len(self.hidden_sizes)-1) ] + [nn.Dropout(0.0)] ) 
        if do_init: self.reset_parameters()
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

    def reset_parameters(self):
        for i,hs in enumerate(self.hidden_sizes):
            stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
            for ww in zip(self.iFL[i].parameters(), self.hFL[i].parameters(), self.iGL[i].parameters(), self.hGL[i].parameters()):
                for w in ww: nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return [ tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes  ]


    def forward(self, Xt, h=None): 
        if h is None: h=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        seq=[h]
        seq_last = []
        for xt in tt.split(Xt, 1, dim=self.seq_dim): 
            H = []
            x,h = xt.squeeze(dim=self.seq_dim), seq[-1]
            for i in range(len(self.hidden_sizes)):
                #print(f'{x.shape=}, h[i].shape={seq[-1][i].shape}')
                F = tt.sigmoid(self.iFL[i](x) + self.hFL[i](h[i]) - self.beta)
                G = self.nlL[i](self.iGL[i](x) + self.hGL[i](h[i]))
                x = F*h[i] + (1-F)*G
                H.append(x)
                x = self.dropouts[i](x) #<--- dropout only output
            seq_last.append(x)
            seq.append(H)
            
        out = tt.stack(seq_last, dim=self.seq_dim)
        return out, tt.stack(seq[-1])


# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =