# ktorch
__doc__=r"""
:py:mod:`known/ktorch/rnnc.py`
"""

import torch as tt
import torch.nn as nn
from copy import deepcopy
from .common import build_activation  

__all__ = [
    'ELMANCell', 'GRUCell', 'MGUCell', 'LSTMCell', 'PLSTMCell', 'JANETCell', 'MGRUCell',
    'RNNCell', 'RNNStack', 'RNNModule',
]

#----------------------------
""" Pre-Defined Cells """
#----------------------------

class ELMANCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_bias=True, hidden_activation=None, 
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #---------- GATES ---------------------
        self.hidden_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = hidden_bias, 
            device=device, dtype=dtype )
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.has_cell_state = False

    def forward(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        h_ = self.hidden_act(self.hidden_gate( xh ))
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input
    
class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_bias=True, hidden_activation=None,
                update_bias=True, update_activation=None, 
                reset_bias=True, reset_activation=None, 
                cell_type=0,
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        self.cell_type=cell_type

        if self.cell_type==0: 
            in_features = self.input_size + self.hidden_size
            self.forward = self.forward_0
        else:
            in_features = self.input_size 
            self.forward = self.forward_1
        #---------- GATES ---------------------
        self.hidden_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = hidden_bias, 
            device=device, dtype=dtype )
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.update_gate = nn.Linear(
            in_features = in_features, 
            out_features = self.hidden_size, 
            bias = update_bias, 
            device=device, dtype=dtype )
        self.update_act = build_activation(update_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.reset_gate = nn.Linear(
            in_features = in_features, 
            out_features = self.hidden_size, 
            bias = reset_bias, 
            device=device, dtype=dtype )
        self.reset_act = build_activation(reset_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.has_cell_state = False

    def forward_0(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        u = self.update_act(self.update_gate( xh ))
        r = self.reset_act(self.reset_gate( xh ))
        xr = tt.concat( (x, r*h), dim=-1)
        d = self.hidden_act(self.hidden_gate( xr )) # candidate
        h_ = (u * h) + ((1-u) * d)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input

    def forward_1(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        u = self.update_act(self.update_gate( h ))
        r = self.reset_act(self.reset_gate( h ))
        xr = tt.concat( (x, r*h), dim=-1)
        d = self.hidden_act(self.hidden_gate( xr )) # candidate
        h_ = (u * h) + ((1-u) * d)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input
    
class MGUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_bias=True, hidden_activation=None,
                forget_bias=True, forget_activation=None, 
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size

        #---------- GATES ---------------------
        self.hidden_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size,
            out_features = self.hidden_size, 
            bias = hidden_bias, 
            device=device, dtype=dtype )
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.forget_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size,
            out_features = self.hidden_size, 
            bias = forget_bias, 
            device=device, dtype=dtype )
        self.forget_act = build_activation(forget_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.has_cell_state = False

    def forward(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        f = self.forget_act(self.forget_gate( xh ))
        xf = tt.concat( (x, f*h), dim=-1)
        d = self.hidden_act(self.hidden_gate( xf )) # candidate
        h_ = (f * d) + ((1-f) * h)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_activation=None,
                input_bias=True, input_activation=None, 
                forget_bias=True, forget_activation=None, 
                output_bias=True, output_activation=None, 
                cell_bias=True, cell_activation=None, 
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #---------- GATES ---------------------
        self.input_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = input_bias, 
            device=device, dtype=dtype )
        self.input_act = build_activation(input_activation, tt.tanh)
        #---------- ----- ---------------------
        self.forget_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = forget_bias, 
            device=device, dtype=dtype )
        self.forget_act = build_activation(forget_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.output_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = output_bias, 
            device=device, dtype=dtype )
        self.output_act = build_activation(output_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.cell_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = cell_bias, 
            device=device, dtype=dtype )
        self.cell_act = build_activation(cell_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.has_cell_state = True

    def forward(self, x, s): #<-- input x and state s
        h, c = s
        xh = tt.concat( (x, h), dim=-1)
        f = self.forget_act(self.forget_gate( xh ))
        i = self.input_act(self.input_gate( xh ))
        o = self.output_act(self.output_gate( xh ))
        g = self.cell_act(self.cell_gate( xh ))
        c_ = (f*c) + (i*g)
        h_ = o * self.hidden_act(c_)
        s_ = (h_, c_)
        return s_, xh #<--- next state s_ and conatenated input

class PLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_activation=None,
                input_bias=True, input_activation=None, 
                forget_bias=True, forget_activation=None, 
                output_bias=True, output_activation=None, 
                cell_bias=True, cell_activation=None, 
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #---------- GATES ---------------------
        self.input_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = input_bias, 
            device=device, dtype=dtype )
        self.input_act = build_activation(input_activation, tt.tanh)
        #---------- ----- ---------------------
        self.forget_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = forget_bias, 
            device=device, dtype=dtype )
        self.forget_act = build_activation(forget_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.output_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = output_bias, 
            device=device, dtype=dtype )
        self.output_act = build_activation(output_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.cell_gate = nn.Linear(
            in_features = self.input_size, 
            out_features = self.hidden_size, 
            bias = cell_bias, 
            device=device, dtype=dtype )
        self.cell_act = build_activation(cell_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.has_cell_state = True

    def forward(self, x, s): #<-- input x and state s
        h, c = s
        xh = tt.concat( (x, h), dim=-1)
        xc = tt.concat( (x, c), dim=-1)
        f = self.forget_act(self.forget_gate( xc ))
        i = self.input_act(self.input_gate( xc ))
        o = self.output_act(self.output_gate( xc ))
        g = self.cell_act(self.cell_gate( x ))
        c_ = (f*c) + (i*g)
        h_ = o * self.hidden_act(c_)
        s_ = (h_, c_)
        return s_, xh #<--- next state s_ and conatenated input
    
class JANETCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_bias=True, hidden_activation=None,
                forget_bias=True, forget_activation=None, 
                beta=1.0,
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #---------- GATES ---------------------
        self.hidden_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = hidden_bias, 
            device=device, dtype=dtype )
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.forget_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = forget_bias, 
            device=device, dtype=dtype )
        self.forget_act = build_activation(forget_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.beta = beta
        self.forward = self.forward_with_beta if self.beta!=0 else self.forward_without_beta
        #---------- ----- ---------------------
        self.has_cell_state = True

    def forward_without_beta(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        f = self.forget_act(self.forget_gate( xh ))
        d = self.hidden_act(self.hidden_gate( xh ))
        h_ = (f * h) + ((1-f) * d)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input

    def forward_with_beta(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        ff = self.forget_gate( xh )
        f = self.forget_act(ff)
        f_ = self.forget_act(ff-self.beta)
        d = self.hidden_act(self.hidden_gate( xh ))
        h_ = (f * h) + ((1-f_) * d)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input

class MGRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                hidden_bias=True, hidden_activation=None,
                update_bias=True, update_activation=None, 
                reset_bias=True, reset_activation=None, 
                dtype=None, device=None) -> None:
        super().__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #---------- GATES ---------------------
        self.hidden_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = hidden_bias, 
            device=device, dtype=dtype )
        self.hidden_act = build_activation(hidden_activation, tt.tanh)
        #---------- ----- ---------------------
        self.update_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = update_bias, 
            device=device, dtype=dtype )
        self.update_act = build_activation(update_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.reset_gate = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.hidden_size, 
            bias = reset_bias, 
            device=device, dtype=dtype )
        self.reset_act = build_activation(reset_activation, tt.sigmoid)
        #---------- ----- ---------------------
        self.has_cell_state = False

    def forward(self, x, s): #<-- input x and state s
        h, _ = s
        xh = tt.concat( (x, h), dim=-1)
        u = self.update_act(self.update_gate( xh ))
        r = self.reset_act(self.reset_gate( xh ))
        d = self.hidden_act(self.hidden_gate( xh ))

        h_ = (u * h) + (r * d)
        s_ = (h_, None)
        return s_, xh #<--- next state s_ and conatenated input

#----------------------------
""" Abstract Cells """
#----------------------------

class RNNCell(nn.Module):
    r""" Abstract RNN-Cell with a core that is one of pre-defined Cell"""
    
    @property
    def input_size(self):
        return self.cell.input_size
    
    @property
    def hidden_size(self):
        return self.cell.hidden_size
    
    @property
    def has_cell_state(self):
        return self.cell.has_cell_state

    def __init__(self, cell) -> None:
        super().__init__()
        self.cell = cell

        self.hasi2o=False
        self.haso2o=False
        self.buffed=0 # 0=no buff, 1=i2o only, 2=o2o only, 3=both
        self.forward = self.forward_i2h
        self.output_size = self.hidden_size

    def build_i2o(self, size, bias, activation, dtype, device ):
        if self.hasi2o: raise Exception(f'[!] only one i2o layer can be added')
        if self.haso2o: raise Exception(f'[!] i2o layer must be added before o2o layer')
        self.buffed=1
        self.forward = self.forward_i2o
        self.hasi2o=True
        
        self.i2o_size = size
        self.i2o = nn.Linear(
            in_features = self.input_size + self.hidden_size, 
            out_features = self.i2o_size, 
            bias = bias, 
            device=device, dtype=dtype )
        self.i2o_act = build_activation(activation, tt.relu)
        self.output_size = self.i2o_size
        
        return self

    def build_o2o(self, size, bias, activation, dtype, device ):
        if self.haso2o: raise Exception(f'[!] only one o2o layer can be added')
        self.buffed = 3 if self.hasi2o else 2
        self.forward = self.forward_i2o_o2o if self.hasi2o else self.forward_o2o
        self.haso2o = True

        self.o2o_size = size
        self.o2o = nn.Linear(
            in_features = self.hidden_size + (self.i2o_size if self.hasi2o else self.input_size), 
            out_features = self.o2o_size, 
            bias = bias, 
            device=device, dtype=dtype )
        self.o2o_act = build_activation(activation, tt.relu)
        self.output_size = self.o2o_size
        return self

    def get_state_tensor(self, batch_size, dtype, device):
       return tt.zeros(size=(batch_size, self.hidden_size), dtype=dtype, device=device)

    def get_init_states(self, batch_size, dtype, device):
        return (
            (self.get_state_tensor(batch_size, dtype, device)),
            (self.get_state_tensor(batch_size, dtype, device) if self.has_cell_state else None)
        )

    def forward_i2h(self, x, s):
        s_, xh = self.cell(x, s)
        y = s_[0]
        return y, s_
    
    def forward_i2o(self, x, s):
        s_, xh = self.cell(x, s)
        y = self.i2o_act(self.i2o( xh ))
        return y, s_

    def forward_o2o(self, x, s):
        s_, xh = self.cell(x, s)
        y = self.o2o_act(self.o2o( tt.concat((x, s_[0]), dim=-1) ))
        return y, s_
    
    def forward_i2o_o2o(self, x, s):
        s_, xh = self.cell(x, s)
        y = self.i2o_act(self.i2o( xh ))
        y = self.o2o_act(self.o2o( tt.concat((y, s_[0]), dim=-1) ))
        return y, s_

class RNNStack(nn.Module):
    r""" Stack of Abstract RNN-Cells """

    def __init__(self, layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.n_layers = len(self.layers)
    
    def init_states(self, batch_size, dtype, device):
        return [ l.get_init_states(batch_size, dtype, device)  for l in self.layers ]

    def forward(self, x, S):
        S_=[]
        for i,l in enumerate(self.layers):
            x, s_ = l.forward(x, S[i])
            S_.append(s_)
        return x, S_

class RNNModule(nn.Module):
    r""" RNN Module - with a core of stacked abstract cells, applying forward through the sequence """

    def __init__(self, coreF, bi=False, return_sequences=False, stack_output=False, batch_first=False) -> None:
        super().__init__()
        
        self.coreForward = coreF
        self.bi = bi

        self.return_sequences = return_sequences
        self.stack_output = stack_output

        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)

        #self.input_size = self.coreForward.input_size
        #self.hidden_size = self.coreForward.hidden_size
        if return_sequences and not stack_output:
            print(f'{__class__}:: {return_sequences=} and {stack_output=} :: forward method will return list')
        if self.bi:
            self.coreBackward = deepcopy(coreF)
            self.forward=self.forward_bi
            #self.output_size = self.coreForward.output_size*2
        else:
            self.forward=self.forward_uni
            #self.output_size = self.coreForward.output_size

    def forward_uni(self, X, H=None, future=0):
        return self.forward_core(self.coreForward, X, H, future=future, reverse=False)
    
    def forward_bi(self, X, H=(None, None), future=0):
        if self.return_sequences and not self.stack_output:
            outF = self.forward_core(self.coreForward, X, H[0], future=future, reverse=False)
            outB = self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)
            out = [ tt.cat(( f,b ), dim=-1) for f,b in zip(outF, outB) ]
        else:
            out = tt.cat((self.forward_core(self.coreForward, X, H[0], future=future, reverse=False),
            self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)), dim=-1)

        return out

    def forward_core(self, core, Xt, Ht, future=0, reverse=False):
        r""" Applies forward pass through the entire input sequence 
        
        :param Xt:  input sequence
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        :param reverse: It True, processes sequence in reverse order
        """
        if Ht is None: Ht = core.init_states(Xt.shape[0], Xt.dtype, Xt.device)
        #if Ht is None: Ht=self.init_states(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        #Ht=[h] #<------ no need to store all the hidden states
        Yt = [] #<---- outputs at each timestep
        timesteps = reversed(tt.split(Xt, 1, dim=self.seq_dim)) \
                    if reverse else tt.split(Xt, 1, dim=self.seq_dim)
        for xt in timesteps: 
            x = xt.squeeze(dim=self.seq_dim)
            y, Ht = core.forward(x, Ht)
            Yt.append(y)
        
        #<--- IMP: future arg will work only when (input_size == hidden_size of the last layer)
        for _ in range(future):
            x = Yt[-1]
            y, Ht = core.forward(x, Ht)
            Yt.append(y)
            
        if self.return_sequences:
            out= (tt.stack(Yt, dim=self.seq_dim) if self.stack_output else Yt)
        else:
            out= (y.unsqueeze(dim=self.seq_dim) if self.stack_output else y)
            
        return  out



# ARCHIVE

# if self.return_sequences:
#     if self.stack_output:
#         out = tt.cat((self.forward_core(self.coreForward, X, H[0], future=future, reverse=False),
#         self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)), dim=-1)
#     else:
#         outF = self.forward_core(self.coreForward, X, H[0], future=future, reverse=False)
#         outB = self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)
#         out = [ tt.cat(( f,b ), dim=-1) for f,b in zip(outF, outB) ]
#     #out= (tt.stack(Yt, dim=self.seq_dim) if self.stack_output else Yt)
# else:
#     if self.stack_output:
#         out = tt.cat((self.forward_core(self.coreForward, X, H[0], future=future, reverse=False),
#         self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)), dim=-1)
#     else:
#         outF = self.forward_core(self.coreForward, X, H[0], future=future, reverse=False)
#         outB = self.forward_core(self.coreBackward, X, H[1], future=future, reverse=True)
#         out = tt.cat(( outF,outB ), dim=-1)