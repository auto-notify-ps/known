#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/ktorch/rnn.py`

Implements base RNN class along with other RNN variants.
The variants have some common initialization arguments which can be refered from the base RNN class.

The RNN have base cell architecture for producing hidden states (`i2h`) and can be extended to have additional weights for

    * producing output with only `(input + previous_hidden)` called (`i2o`) 
        .. image:: rnn1.png
    * producing output with `(input + previous_hidden + current_hidden)` called (`o2o`) 
        .. image:: rnn2.png

The base cell architecture can be implemented by not specifying any output (dont define ``output_sizes`` and ``output_sizes2``).
The first architecture can be implemented by specifying only the first output (define ``output_sizes`` and dont define ``output_sizes2``).
The second architecture can be implemented by specifying both the outputs (define ``output_sizes`` and ``output_sizes2``).
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'RNN', 'ELMAN', 'GRU', 'LSTM', 'JANET', 'MGU'
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
import torch.nn as nn
#import torch.nn.functional as ff
import math
from typing import Any, Union, Iterable, Callable, Dict, Tuple, List
from ..common import LinearActivated
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class RNN(nn.Module):

    """ Recurrent Neural Network base class
        * additional parameters defined for 2 output -  `i2o` and `o2o`
        * can choose custom activation at each gate and each output (including seperate for last layer)
        * if output_sizes is None, no additional weights are defined for `i2o`
        * if output_sizes2 is None, no additional weights are defined for `o2o`

    :param input_size:      no of features in the input vector
    :param hidden_sizes:    no of features in the hidden vector (`i2h`)
    :param output_sizes:    optional, no of features in the first output vector (`i2o`)
    :param output_sizes2:   optional, no of features in the second output vector (`o2o`)
    :param dropout:         probability of dropout, dropout is not applied at the last layer
    :param batch_first:     if True, `batch_size` is assumed as the first dimension in the input
    :param stack_output:    if True, stacks the output of all timesteps into a single tensor, otherwise keeps them in a list
    :param cell_bias:       if True, uses bias at the cell level gates (`i2h`)
    :param out_bias:        if True, uses bias at first output (`i2o`)
    :param out_bias2:       if True, uses bias at second output (`o2o`)
    
    .. note:: 
        * Do not use this class directly, it is meant to provide a base class from which other RNN modules are inherited
        * Activation arguments can be a tuple like ``(nn.Tanh, {})`` or a callable like ``torch.tanh``
        * if ``batch_first`` is True, accepts input of the form ``(batch_size, seq_len, input_size)``, otherwise ``(seq_len, batch_size, input_size)``

    """
    
    def __init__(self,
                input_size,         # input features
                hidden_sizes,       # hidden features at each layer
                output_sizes=None,  # output features at each layer (if None, same as hidden)
                output_sizes2=None,  # output features at each layer (if None, same as hidden)
                dropout=0.0,        # dropout after each layer, only if hidden_sizes > 1
                batch_first=False,  # if true, excepts input as (batch_size, seq_len, input_size) else (seq_len, batch_size, input_size)
                stack_output=False, # if true, stack output from all timesteps, else returns a list of outputs
                cell_bias = True, 
                out_bias = True,
                out_bias2 = True,
                bidir = False,
                dtype=None,
                device=None,
                ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_sizes = tuple(hidden_sizes)
        self.n_hidden = len(self.hidden_sizes)
        self.n_last = self.n_hidden-1
        if output_sizes is not None:
            self.output_sizes = tuple(output_sizes)
            self.n_output = len(self.output_sizes)
            assert self.n_hidden==self.n_output, f'hidden_sizes should be equal to output_sizes, {self.n_hidden}!={self.n_output}'


            if output_sizes2 is not None:
                self.output_sizes2 = tuple(output_sizes2)
                self.n_output2 = len(self.output_sizes2)
                assert self.n_hidden==self.n_output2, f'hidden_sizes should be equal to output_sizes2, {self.n_hidden}!={self.n_output2}'
            else:
                self.output_sizes2 = None
                self.n_output2=0

        else:
            self.output_sizes = None
            self.n_output=0
            if output_sizes2 is not None:
                print(f'Setting output_sizes2 requires setting output_sizes first')
            self.output_sizes2 = None
            self.n_output2=0


        self.cell_bias=cell_bias
        self.out_bias=out_bias
        self.out_bias2=out_bias2
        self.bidir = bidir

        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)
        self.stack_output = stack_output
        
        # set dropouts
        if hasattr(dropout, '__len__'): # different droupout at each layer
            if len(dropout) < self.n_hidden-1:
                self.dropouts = list(dropout)
                while len(self.dropouts) < self.n_hidden-1: self.dropouts.append(0.0)
            else:
                self.dropouts = list(dropout[0:self.n_hidden-1])
        else:
            self.dropouts = [ dropout for _ in range(self.n_hidden-1) ]
        self.dropouts.append(0.0) # for last layer, no dropout

        # build & initialize internal parameters
        self.parameters_module = self.build_parameters(dtype, device)
        self.reset_parameters() # reset_parameters should be the lass call before exiting __init__

    def _build_parameters_(self, hidden_names, hidden_activations, 
                            output_names, output_activations, 
                            output_names2, output_activations2, last_activations, 
                            dtype, device):
        if self.n_output>0:
            if self.n_output2>0:
                names=hidden_names
                input_sizes=(self.input_size,) + self.output_sizes2[:-1]
                n = len(hidden_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.hidden_sizes, self.hidden_sizes):
                    for j in range(n):
                        if hidden_activations[j] is None:
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.cell_bias, dtype=dtype, device=device))
                        else:
                            weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                    self.cell_bias, hidden_activations[j], dtype=dtype, device=device))

                for name,weight in zip(hidden_names, weights): setattr(self, name, nn.ModuleList(weight))
                


                names=names+output_names

                n = len(output_names)
                weights = [[] for _ in range(n)]
                
                for in_features, cat_features, out_features in zip(input_sizes, self.hidden_sizes, self.output_sizes):
                    for j in range(n):
                        if output_activations[j] is None:
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.out_bias, dtype=dtype, device=device))
                        else:
                            weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                    self.out_bias, output_activations[j], dtype=dtype, device=device))

                for name,weight in zip(output_names, weights): setattr(self, name, nn.ModuleList(weight))
                



                names=names+output_names2

                n = len(output_names2)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                #input_sizes=(self.input_size,) + self.output_sizes[:-1]
                for i,(in_features, cat_features, out_features) in enumerate(zip(self.hidden_sizes, self.output_sizes, self.output_sizes2)):
                    if i==is_last:
                        for j in range(n):
                            if last_activations[j] is None:
                                weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                        self.out_bias2, dtype=dtype, device=device))
                            else:
                                weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                        self.out_bias2, last_activations[j], dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            if output_activations2[j] is None:
                                weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                        self.out_bias2, dtype=dtype, device=device))
                            else:
                                weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                        self.out_bias2, output_activations2[j], dtype=dtype, device=device))

                for name,weight in zip(output_names2, weights): setattr(self, name, nn.ModuleList(weight))
                
            else:
                names=hidden_names
                input_sizes=(self.input_size,) + self.output_sizes[:-1]
                n = len(hidden_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.hidden_sizes, self.hidden_sizes):
                    for j in range(n):
                        if hidden_activations[j] is None:
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.cell_bias, dtype=dtype, device=device))
                        else:
                            weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                    self.cell_bias, hidden_activations[j], dtype=dtype, device=device))

                for name,weight in zip(hidden_names, weights): setattr(self, name, nn.ModuleList(weight))
                
                names=names+output_names

                n = len(output_names)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                for i,(in_features, cat_features, out_features) in enumerate(zip(input_sizes, self.hidden_sizes, self.output_sizes)):
                    if i==is_last:
                        for j in range(n):
                            if last_activations[j] is None:
                                weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                        self.out_bias, dtype=dtype, device=device))
                            else:
                                weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                        self.out_bias, last_activations[j], dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            if output_activations[j] is None:
                                weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                        self.out_bias, dtype=dtype, device=device))
                            else:
                                weights[j].append(LinearActivated(in_features + cat_features, out_features, 
                                        self.out_bias, output_activations[j], dtype=dtype, device=device))

                for name,weight in zip(output_names, weights): setattr(self, name, nn.ModuleList(weight))

        else:
            names=hidden_names
            n = len(hidden_names)
            weights = [[] for _ in range(n)]
            input_sizes=(self.input_size,) + self.hidden_sizes[:-1]
            for in_features, out_features in zip(input_sizes, self.hidden_sizes):
                for j in range(n):
                    if hidden_activations[j] is None:
                        weights[j].append(nn.Linear(in_features + out_features, out_features, 
                                self.cell_bias, dtype=dtype, device=device))
                    else:
                        weights[j].append(LinearActivated(in_features + out_features, out_features, 
                                self.cell_bias, hidden_activations[j], dtype=dtype, device=device))

            for name,weight in zip(hidden_names, weights): setattr(self, name, nn.ModuleList(weight))
        return tuple([getattr(self, name) for name in names])

    def _build_activations_(self, activation_arg, name):
        self.modular_activation=None
        def no_act(x): return x
        if activation_arg is None: activation_arg=no_act
        if hasattr(activation_arg, '__len__'):
            # activation_arg is like activation_arg=(nn.Tanh, {})
            actModule = activation_arg[0]
            actArgs = activation_arg[1]
            setattr(self, name, actModule(**actArgs))
            self.modular_activation= True #<--- modular activations
        else:
            # activation_arg is like activation_arg=tt.tanh
            setattr(self, name, activation_arg)
            self.modular_activation= False

    def reset_parameters(self):
        for modulelist in self.parameters_module:
            for hs,m in zip(self.hidden_sizes,modulelist):
                stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
                for w in m.parameters(): nn.init.uniform_(w, -stdv, stdv)

    def init_hidden(self, batch_size, dtype=None, device=None):
        return tuple([ 
            [tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.hidden_sizes]  \
              for _ in range(self.n_states)  ])

    def forward(self, Xt, h=None, future=0):
        r""" Applies forward pass through the entire input sequence 
        
        :param Xt:  input sequence
        :param H:   hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        """
        if h is None: h=self.init_hidden(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
        Ht=[h]
        Yt = [] #<---- outputs at each timestep
        for xt in tt.split(Xt, 1, dim=self.seq_dim): 
            x, h = xt.squeeze(dim=self.seq_dim), Ht[-1]
            y, h_ = self.forward_one(x, h)
            Yt.append(y)
            Ht.append(h_)

        #<--- IMP: future arg will work only when (input_size == hidden_size of the last layer)
        for _ in range(future):
            x, h = Yt[-1], Ht[-1]
            y, h_ = self.forward_one(x, h)
            Yt.append(y)
            Ht.append(h_)

        out = tt.stack(Yt, dim=self.seq_dim) if self.stack_output else Yt
        hidden = Ht[-1]
        return  out, hidden


    @tt.no_grad()
    def copy_torch(self, model):
        sd = model.state_dict()
        for i in range(self.n_hidden):
            ihW, hhW = sd[f'weight_ih_l{i}'], sd[f'weight_hh_l{i}']

            for n, ihw in enumerate(self.W_names):
                ll = getattr(self, ihw)
                ll[i].L.weight.copy_(
                        tt.cat((ihW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:],
                        hhW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:]), dim=1)
                        )
            if self.cell_bias:
                ihB = sd[f'bias_ih_l{i}']
                hhB = sd[f'bias_hh_l{i}']

                for n, ihw in enumerate(self.W_names):
                    ll = getattr(self, ihw)
                    ll[i].L.bias.copy_(
                            ihB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]] + \
                            hhB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]]
                            )
                    
class ELMAN(RNN):
    r"""
    Defines an Elman RNN, additional arguments are as follows

    :param activation_gate: activation function at Elman gate 
    :param activation_out:  activation at first output
    :param activation_out2: activation at second output
    :param activation_last: activation at the last layer of the final output

    :ref: `Elman RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`__
    """

    def __init__(self, input_size, hidden_sizes, output_sizes=None, output_sizes2=None, dropout=0, 
            batch_first=False, stack_output=False, cell_bias=True, out_bias=True, out_bias2=True, 
            bidir=False, dtype=None, device=None,
            activation_gate=tt.sigmoid, activation_out=None, activation_out2=None, activation_last=None) -> None:
        self.activation_gate = activation_gate
        self.activation_out = activation_out
        self.activation_out2 = activation_out2
        self.activation_last = activation_last
        self.n_states=1
        super().__init__(input_size, hidden_sizes, output_sizes, output_sizes2, dropout, batch_first, stack_output, cell_bias, out_bias, out_bias2, bidir, dtype, device)

    def build_parameters(self, dtype, device):
        self.W_names = ('ihL', )
        hidden_names = ('ihL',)
        hidden_activations = (self.activation_gate,)
        
        if self.n_output>0: 
            output_names = ('iyL',)
            output_activations = (self.activation_out, )
            last_activations = (self.activation_last, )
            if self.n_output2>0:
                output_names2 = ('yyL',)
                output_activations2 = (self.activation_out2, )
                self.forward_one = self.forward_one_yy
            else:
                output_names2 = None
                output_activations2 = None
                self.forward_one = self.forward_one_y
        else:
            output_names = None
            output_activations = None
            output_names2 = None
            output_activations2 = None
            last_activations = None
            self.forward_one = self.forward_one_x
        return self._build_parameters_( 
            hidden_names, 
            hidden_activations, 
            output_names, 
            output_activations, 
            output_names2, 
            output_activations2, 
            last_activations, 
            dtype, device)
        
    def forward_one_x(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            x = self.ihL[i]( xh )
            H.append(x)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_y(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            d = self.ihL[i]( xh )
            H.append(d)
            x = self.iyL[i]( xh ) 
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_yy(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            d = self.ihL[i]( xh )
            H.append(d)
            x = self.iyL[i]( xh ) 
            x = tt.concat( (x, d), dim=-1)
            x = self.yyL[i]( x )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

class GRU(RNN):
    r"""
    Defines an Gated Recurrent Unit, additional arguments are as follows

    :param activation_r_gate: activation function at R gate 
    :param activation_z_gate: activation function at Z gate 
    :param activation_n_gate: activation function at N gate 
    :param activation_out:  activation at first output
    :param activation_out2: activation at second output
    :param activation_last: activation at the last layer of the final output

    :ref: `GRU RNN <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`__
    """
    def __init__(self, input_size, hidden_sizes, output_sizes=None, output_sizes2=None, dropout=0, 
                batch_first=False, stack_output=False, cell_bias=True, out_bias=True, out_bias2=True,  
                bidir=False, dtype=None, device=None,
                activation_r_gate=tt.sigmoid, activation_z_gate=tt.sigmoid, activation_n_gate=tt.sigmoid, activation_out=None, activation_out2=None, activation_last=None) -> None:
        self.activation_r_gate = activation_r_gate
        self.activation_z_gate = activation_z_gate
        self.activation_n_gate = activation_n_gate
        self.activation_out = activation_out
        self.activation_out2 = activation_out2
        self.activation_last = activation_last
        self.n_states=1
        super().__init__(input_size, hidden_sizes, output_sizes, output_sizes2, dropout, batch_first, stack_output, cell_bias, out_bias, out_bias2, bidir, dtype, device)

    def build_parameters(self, dtype, device):
        self.W_names = ('irL', 'izL', 'inL')
        hidden_names = ('irL', 'izL', 'inL')
        hidden_activations = (self.activation_r_gate, self.activation_z_gate, self.activation_n_gate)
        if self.n_output>0: 
            output_names = ('iyL',)
            output_activations = (self.activation_out, )
            last_activations = (self.activation_last, )
            if self.n_output2>0:
                output_names2 = ('yyL',)
                output_activations2 = (self.activation_out2, )
                self.forward_one = self.forward_one_yy
            else:
                output_names2 = None
                output_activations2 = None
                self.forward_one = self.forward_one_y
        else:
            output_names = None
            output_activations = None
            output_names2 = None
            output_activations2 = None
            last_activations = None
            self.forward_one = self.forward_one_x
        return self._build_parameters_( 
            hidden_names, 
            hidden_activations, 
            output_names, 
            output_activations, 
            output_names2, 
            output_activations2, 
            last_activations, 
            dtype, device)

    def forward_one_x(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            R = self.irL[i]( xh )
            Z = self.izL[i]( xh )
            xr = tt.concat( (x, R*h[i]), dim=-1)
            N = self.inL[i]( xr )
            x = (1-Z) * N + (Z * h[i])  #x = (1-Z) * h[i] + (Z * N) 
            H.append(x)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_y(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            R = self.irL[i]( xh ) 
            Z = self.izL[i]( xh )
            xr = tt.concat( (x, R*h[i]), dim=-1)
            N = self.inL[i]( xr )
            d = (1-Z) * N + (Z * h[i])  #x = (1-Z) * h[i] + (Z * N) 
            H.append(d)
            x = self.iyL[i]( xh )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
        
    def forward_one_yy(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            R = self.irL[i]( xh ) 
            Z = self.izL[i]( xh )
            xr = tt.concat( (x, R*h[i]), dim=-1)
            N = self.inL[i]( xr )
            d = (1-Z) * N + (Z * h[i])  #x = (1-Z) * h[i] + (Z * N) 
            H.append(d)
            x = self.iyL[i]( xh )
            x = tt.concat( (x, d), dim=-1)
            x = self.yyL[i]( x )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

class LSTM(RNN):
    r"""
    Defines an Long Short Tem Memory RNN, additional arguments are as follows

    :param activation_i_gate: activation function at I gate 
    :param activation_f_gate: activation function at F gate 
    :param activation_g_gate: activation function at G gate 
    :param activation_o_gate: activation function at O gate 
    :param activation_cell_out:  activation at cell state output
    :param activation_out:  activation at first output
    :param activation_out2: activation at second output
    :param activation_last: activation at the last layer of the final output

    :ref: `LSTM RNN <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`__
    """
    def __init__(self, input_size, hidden_sizes, output_sizes=None,output_sizes2=None, dropout=0, batch_first=False, stack_output=False, cell_bias=True, out_bias=True, out_bias2=True,  bidir=False, dtype=None, device=None,
                activation_i_gate=tt.sigmoid, activation_f_gate=tt.sigmoid, activation_g_gate=tt.sigmoid, activation_o_gate=tt.sigmoid, activation_cell=tt.tanh, activation_out=None, activation_out2=None, activation_last=None) -> None:
        self.activation_i_gate = activation_i_gate
        self.activation_f_gate = activation_f_gate
        self.activation_g_gate = activation_g_gate
        self.activation_o_gate = activation_o_gate
        self.activation_out = activation_out
        self.activation_out2 = activation_out2
        self.activation_cell = activation_cell
        self.activation_last = activation_last
        self.n_states=2
        super().__init__(input_size, hidden_sizes, output_sizes, output_sizes2, dropout, batch_first, stack_output, cell_bias, out_bias, out_bias2, bidir, dtype, device)

    def build_parameters(self, dtype, device):
        self.W_names = ('iiL', 'ifL', 'igL', 'ioL')
        hidden_names = ('iiL', 'ifL', 'igL', 'ioL')
        hidden_activations = (self.activation_i_gate, self.activation_f_gate, self.activation_g_gate, self.activation_o_gate)
        self._build_activations_(self.activation_cell, 'actC')
        if self.n_output>0: 
            output_names = ('iyL',)
            output_activations = (self.activation_out, )
            last_activations = (self.activation_last, )
            if self.n_output2>0:
                output_names2 = ('yyL',)
                output_activations2 = (self.activation_out2, )
                self.forward_one = self.forward_one_yy
            else:
                output_names2 = None
                output_activations2 = None
                self.forward_one = self.forward_one_y
        else:
            output_names = None
            output_activations = None
            output_names2 = None
            output_activations2 = None
            last_activations = None
            self.forward_one = self.forward_one_x
        return self._build_parameters_( 
            hidden_names, 
            hidden_activations, 
            output_names, 
            output_activations, 
            output_names2, 
            output_activations2, 
            last_activations, 
            dtype, device)
        
    def forward_one_x(self, x, s):
        H,C=[],[]
        h,c = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            I =  self.iiL[i]( xh ) 
            F =  self.ifL[i]( xh ) 
            G =  self.igL[i]( xh ) 
            O =  self.ioL[i]( xh ) 
            c_ = F*c[i] + I*G
            x = O * self.actC(c_)
            H.append(x)
            C.append(c_)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_y(self, x, s):
        H,C=[],[]
        h,c = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            I =  self.iiL[i]( xh ) 
            F =  self.ifL[i]( xh ) 
            G =  self.igL[i]( xh ) 
            O =  self.ioL[i]( xh ) 
            c_ = F*c[i] + I*G
            d = O * self.actC(c_)
            H.append(d)
            C.append(c_)
            x =  self.iyL[i]( xh ) 
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_yy(self, x, s):
        H,C=[],[]
        h,c = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            I =  self.iiL[i]( xh ) 
            F =  self.ifL[i]( xh ) 
            G =  self.igL[i]( xh ) 
            O =  self.ioL[i]( xh ) 
            c_ = F*c[i] + I*G
            d = O * self.actC(c_)
            H.append(d)
            C.append(c_)
            x =  self.iyL[i]( xh ) 
            x = tt.concat( (x, d), dim=-1)
            x = self.yyL[i]( x )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

class JANET(RNN):
    r"""
    Defines an Just Another NETwork RNN, additional arguments are as follows

    :param activation_f_gate: activation function at F gate 
    :param activation_g_gate: activation function at G gate 
    :param activation_out:  activation at first output
    :param activation_out2: activation at second output
    :param activation_last: activation at the last layer of the final output
    :param beta:    the ``beta`` hyperparameter

    :ref: `JANET RNN <https://arxiv.org/pdf/1804.04849.pdf>`__
    """
    def __init__(self, input_size, hidden_sizes, output_sizes=None, output_sizes2=None, dropout=0, batch_first=False, stack_output=False, cell_bias=True, out_bias=True, out_bias2=True,  bidir = False, dtype=None, device=None,
                activation_f_gate=tt.sigmoid, activation_g_gate=tt.tanh, activation_out=None, activation_out2=None, activation_last=None, beta=0.0) -> None:
        self.activation_f_gate = activation_f_gate
        self.activation_g_gate = activation_g_gate
        self.activation_out = activation_out
        self.activation_out2 = activation_out2
        self.activation_last = activation_last
        self.beta=beta
        self.n_states=1
        super().__init__(input_size, hidden_sizes, output_sizes, output_sizes2, dropout, batch_first, stack_output, cell_bias, out_bias, out_bias2, bidir, dtype, device)

    def build_parameters(self, dtype, device):
        self.W_names = ('ifL', 'igL')
        hidden_names = ('ifL', 'igL')
        hidden_activations = (self.activation_f_gate, self.activation_g_gate)
        if self.n_output>0: 
            output_names = ('iyL',)
            output_activations = (self.activation_out, )
            last_activations = (self.activation_last, )
            if self.n_output2>0:
                output_names2 = ('yyL',)
                output_activations2 = (self.activation_out2, )
                self.forward_one = self.forward_one_yy
            else:
                output_names2 = None
                output_activations2 = None
                self.forward_one = self.forward_one_y
        else:
            output_names = None
            output_activations = None
            output_names2 = None
            output_activations2 = None
            last_activations = None
            self.forward_one = self.forward_one_x
        return self._build_parameters_( 
            hidden_names, 
            hidden_activations, 
            output_names, 
            output_activations, 
            output_names2, 
            output_activations2, 
            last_activations, 
            dtype, device)

    def forward_one_x(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F =  self.ifL[i]( xh ) - self.beta
            G =  self.igL[i]( xh )
            x = F*h[i] + (1-F)*G
            H.append(x)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_y(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F =  self.ifL[i]( xh ) - self.beta
            G =  self.igL[i]( xh )
            d = F*h[i] + (1-F)*G
            H.append(d)
            x = self.iyL[i]( xh )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_yy(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F =  self.ifL[i]( xh ) - self.beta
            G =  self.igL[i]( xh )
            d = F*h[i] + (1-F)*G
            H.append(d)
            x = self.iyL[i]( xh )
            x = tt.concat( (x, d), dim=-1)
            x = self.yyL[i]( x )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

class MGU(RNN):
    r"""
    Defines an Minimal Gated Unit, additional arguments are as follows

    :param activation_f_gate: activation function at F gate 
    :param activation_g_gate: activation function at G gate 
    :param activation_out:  activation at first output
    :param activation_out2: activation at second output
    :param activation_last: activation at the last layer of the final output

    :ref: `MGU RNN <https://arxiv.org/pdf/1603.09420.pdf>`__
    """
    def __init__(self, input_size, hidden_sizes, output_sizes=None, output_sizes2=None, dropout=0, batch_first=False, stack_output=False, cell_bias=True, out_bias=True, out_bias2=True, bidir = False, dtype=None, device=None,
                activation_f_gate=tt.sigmoid, activation_g_gate=tt.tanh, activation_out=None, activation_out2=None, activation_last=None) -> None:
        self.activation_f_gate = activation_f_gate
        self.activation_g_gate = activation_g_gate
        self.activation_out = activation_out
        self.activation_out2 = activation_out2
        self.activation_last = activation_last
        self.n_states=1
        super().__init__(input_size, hidden_sizes, output_sizes, output_sizes2, dropout, batch_first, stack_output, cell_bias, out_bias, out_bias2, bidir, dtype, device)

    def build_parameters(self, dtype, device):
        self.W_names = ('ifL', 'igL')
        hidden_names = ('ifL', 'igL')
        hidden_activations = (self.activation_f_gate, self.activation_g_gate)
        if self.n_output>0: 
            output_names = ('iyL',)
            output_activations = (self.activation_out, )
            last_activations = (self.activation_last, )
            if self.n_output2>0:
                output_names2 = ('yyL',)
                output_activations2 = (self.activation_out2, )
                self.forward_one = self.forward_one_yy
            else:
                output_names2 = None
                output_activations2 = None
                self.forward_one = self.forward_one_y
        else:
            output_names = None
            output_activations = None
            output_names2 = None
            output_activations2 = None
            last_activations = None
            self.forward_one = self.forward_one_x
        return self._build_parameters_( 
            hidden_names, 
            hidden_activations, 
            output_names, 
            output_activations, 
            output_names2, 
            output_activations2, 
            last_activations, 
            dtype, device)

    def forward_one_x(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F =self.ifL[i]( xh )
            xf = tt.concat( (x, F*h[i]), dim=-1)
            G = self.igL[i]( xf )
            x = (1-F)*h[i] + F*G
            # or x = F*h[i] + (1-F)*G
            H.append(x)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_y(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F = self.ifL[i]( xh )
            xf = tt.concat( (x, F*h[i]), dim=-1)
            G = self.igL[i]( xf )
            d = (1-F)*h[i] + F*G
            # or x = F*h[i] + (1-F)*G
            H.append(d)
            x = self.iyL[i]( xh ) 
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)

    def forward_one_yy(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F = self.ifL[i]( xh )
            xf = tt.concat( (x, F*h[i]), dim=-1)
            G = self.igL[i]( xf )
            d = (1-F)*h[i] + F*G
            # or x = F*h[i] + (1-F)*G
            H.append(d)
            x = self.iyL[i]( xh ) 
            x = tt.concat( (x, d), dim=-1)
            x = self.yyL[i]( x )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=