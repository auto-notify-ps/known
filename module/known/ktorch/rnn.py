# ktorch
__doc__=r"""
:py:mod:`known/ktorch/rnn.py`
"""

import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import math

__all__ = [
    'GRNN', 'RNN', 'ELMAN', 'GRU', 'LSTM', 'MGU', 'JANET'
]


class GRNN(nn.Module):
    r""" Generalized RNN 
        * takes any core module and applies recurance on it through forward method
        * can apply bi-directional forward methods 
        * works with existing RNN modules
        * when inheriting this class make sure to implement the following functions: 
            - ``timesteps_in()``
            - ``init_states()``
            - ``get_input_at()``
            - ``forward_one()``

    :param core_forward: underlying core_forward module on which forward recurance is applied
    :param core_backward: underlying core_backward module on which reverse recurance is applied
    """

    def __init__(self, core_forward, core_backward=None) -> None:
        r"""
        :param core_forward: underlying core_forward module on which forward recurance is applied
        :param core_backward: underlying core_backward module on which reverse recurance is applied
        """
        super().__init__()
        self.core_forward = core_forward
        self.core_backward = core_backward
        self.bidir = (core_backward is not None)
        self.forward = self.forward_bi_dir if self.bidir else self.forward_uni_dir

    def forward_uni_dir(self, Xt, H=None, future=0):
        r""" Applies one-way forward pass through the entire input sequence
        
        :param Xt:  input sequence
        :param H:   hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        """
        Yf, Hf = self.call_core(self.core_forward, False, Xt, H, future)
        return Yf, Hf
    
    def forward_bi_dir(self, Xt, H=(None, None), future=0):
        r""" Applies two-way forward pass through the entire input sequence
        
        :param Xt:  input sequence
        :param H:   tuple of hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        """
        Yf, Hf = self.call_core(self.core_forward, False, Xt, H[0], future)
        Yb, Hb = self.call_core(self.core_backward, True, Xt, H[1], future)
        return (Yf, Yb), (Hf, Hb)

    def call_core(self, core, reverse, Xt, H=None, future=0):
        r""" Applies forward pass through the entire input sequence for the given core
        
        :param core: selected core
        :param reverse: if `True`, reverse the input sequence
        :param Xt:  input sequence
        :param H:   hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        """
        # X = input sequence
        # H = hidden states for each layer at timestep (t-1)
        # future = no of future steps to output

        timesteps = self.timesteps_in(core, Xt) #<<---- how many timesteps in the input sequence X ? 

        # H should contain hidden states for each layer at the last timestep
        # if no hidden-state supplied, create a hidden states for each layer
        if H is None: H=self.init_states(core, Xt) 


        Ht = [H] #<==== hidden states at each timestep
        Yt = []  #<---- output of last cell at each timestep
        for t in (reversed(range(timesteps)) if reverse else range(timesteps)): #<---- for each timestep 
            x = self.get_input_at(core, Xt, t) #<--- input at this time step
            h = Ht[-1] #<---- hidden states for each layer at this timestep
            y, h_ = self.forward_one(core, x, h)

            Yt.append(y)
            Ht.append(h_)
        
        #<--- IMP: future arg will work only when (input_size == hidden_size of the last layer)
        for _ in range(future):
            x = Yt[-1]#<--- input at this time step
            h = Ht[-1] #<---- hidden states for each layer at this timestep
            y, h_ = self.forward_one(core, x, h)

            Yt.append(y)
            Ht.append(h_)

        if core.stack_output: Yt = tt.stack(Yt, dim=core.seq_dim)
        return Yt , Ht[-1]

    @staticmethod
    def timesteps_in(core, Xt): 
        r""" returns the number of timesteps in an input sequence `Xt`"""
        return Xt.shape[core.seq_dim]

    @staticmethod
    def init_states(core, Xt): 
        r""" returns the set of all hidden states required for the `core` module based on input sequence `Xt` """
        return core.init_states(Xt.shape[core.batch_dim], Xt.dtype, Xt.device)
    
    @staticmethod
    def get_input_at(core, Xt, t): 
        r""" returns the input at time step `t` from an input sequence `Xt` """
        return (Xt[:, t, :] if core.batch_first else Xt[t, :, :])

    @staticmethod
    def forward_one(core, x, h): 
        r""" Applies forward pass through the a single timestep of the input

        :param core: selected core
        :param x:  input at current timestep
        :param h:  hidden states from previous timestep
        """
        return core.forward_one(x, h)


class RNN(nn.Module):

    """ Recurrent Neural Network base class
        * additional parameters defined for 2 output -  `i2o` and `o2o`
        * can choose custom activation at each gate and each output 
        * can choose custom activation at seperatly for last layer
        * if i2o_sizes is None, no additional weights are defined for `i2o`
        * if o2o_sizes is None, no additional weights are defined for `o2o`

    :param input_size:      no of features in the input vector
    :param i2h_sizes:       no of features in the hidden vector (`i2h`)
    :param i2o_sizes:    optional, no of features in the first output vector (`i2o`)
    :param o2o_sizes:   optional, no of features in the second output vector (`o2o`)
    :param dropout:         probability of dropout, dropout is not applied at the last layer
    :param batch_first:     if True, `batch_size` is assumed as the first dimension in the input
    :param stack_output:    if True, stacks the output of all timesteps into a single tensor, otherwise keeps them in a list
    :param i2h_bias:       if True, uses bias at the cell level gates (`i2h`)
    :param i2o_bias:        if True, uses bias at first output (`i2o`)
    :param o2o_bias:       if True, uses bias at second output (`o2o`)
    :param i2h_activations:    activations at cell level, keep `None` to use default
    :param i2o_activation:    activations at i2o level, keep `None` to use default
    :param o2o_activation:       activations at o2o level, keep `None` to use default
    :param last_activation:        last output activations, keep `None` to use `No Activation`
    :param hypers:       hyper-parameters dictionary
    
    .. note:: 
        * Do not use this class directly, it is meant to provide a base class from which other RNN modules are inherited
        * Activation arguments can be a tuple like ``(nn.Tanh, {})`` or a callable like ``torch.tanh``
        * if ``batch_first`` is True, accepts input of the form ``(batch_size, seq_len, input_size)``, otherwise ``(seq_len, batch_size, input_size)``

    """
    
    def __init__(self,
                input_size,      
                i2h_sizes,      
                i2o_sizes=None,  
                o2o_sizes=None,  
                dropout=0.0,        
                batch_first=False,  
                stack_output=False, 
                i2h_bias = True, 
                i2o_bias = True,
                o2o_bias = True,
                i2h_activations=None,
                i2o_activation=None,
                o2o_activation=None,
                last_activation=None,
                hypers=None,
                dtype=None,
                device=None,
                ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.i2h_sizes = tuple(i2h_sizes)
        self.n_hidden = len(self.i2h_sizes)
        self.n_last = self.n_hidden-1
        if i2o_sizes is not None:
            self.i2o_sizes = tuple(i2o_sizes)
            self.n_output = len(self.i2o_sizes)
            assert self.n_hidden==self.n_output, f'i2h_sizes should be equal to i2o_sizes, {self.n_hidden}!={self.n_output}'


            if o2o_sizes is not None:
                self.o2o_sizes = tuple(o2o_sizes)
                self.n_output2 = len(self.o2o_sizes)
                assert self.n_hidden==self.n_output2, f'i2h_sizes should be equal to o2o_sizes, {self.n_hidden}!={self.n_output2}'
            else:
                self.o2o_sizes = None
                self.n_output2=0

        else:
            self.i2o_sizes = None
            self.n_output=0
            if o2o_sizes is not None:
                print(f'Setting o2o_sizes requires setting i2o_sizes first')
            self.o2o_sizes = None
            self.n_output2=0


        self.i2h_bias=i2h_bias
        self.i2o_bias=i2o_bias
        self.o2o_bias=o2o_bias


        #self.i2h_activations = i2h_activations
        #self.i2o_activation = i2o_activation
        #self.o2o_activation = o2o_activation
        #self.last_activation = last_activation
        #self.hypers = 
        

        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)
        self.stack_output = stack_output
        
        self.null_state = [None for _ in self.i2h_sizes]
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
        if hypers is None: hypers = {} 
        self.parameters_module = self._build_parameters_(*self.make_parameters(i2h_activations, **hypers),  
                i2o_activation, o2o_activation, last_activation, dtype, device)
        self.reset_parameters() # reset_parameters should be the lass call before exiting __init__

    def make_parameters(self, i2h_activations, hypers):
        # return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
        raise NotImplemented

    def no_act(self, x): return x
    
    def _build_parameters_(self, 
                    has_cell_state, i2h_names, i2h_activations, act_names, hypers,
                    i2o_activation, o2o_activation, last_activation, dtype, device):
        self.W_names = i2h_names
        for k,v in hypers.items(): setattr(self, k, v)
        i2o_names = None
        o2o_names = None
        self.init_states = self.init_states_has_cell_state if has_cell_state else self.init_states_no_cell_state
        if self.n_output>0: 
            self._build_activations_(last_activation, 'lastA')

            i2o_names = ('i2oL',)
            if i2o_activation is None: i2o_activation = tt.relu
            self._build_activations_(i2o_activation, 'i2oA')
            if self.n_output2>0:
                o2o_names = ('o2oL',)
                if o2o_activation is None: o2o_activation = tt.relu
                self._build_activations_(o2o_activation, 'o2oA')
                self.forward_one = self.forward_one_o2o_with_cell_state if has_cell_state else self.forward_one_o2o_no_cell_state
            else:
                self.forward_one = self.forward_one_i2o_with_cell_state if has_cell_state else self.forward_one_i2o_no_cell_state
        else:
            self.forward_one = self.forward_one_i2h_with_cell_state if has_cell_state else self.forward_one_i2h_no_cell_state

        for a,n in zip(i2h_activations, act_names): self._build_activations_(a, n)


        #i2o_names, o2o_names = self._build_meta_(has_cell_state)
        if self.n_output>0:
            if self.n_output2>0:
                names=i2h_names
                input_sizes=(self.input_size,) + self.o2o_sizes[:-1]
                n = len(i2h_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2h_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2h_bias, dtype=dtype, device=device))
                        
                for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
                

                names=names+i2o_names
                n = len(i2o_names)
                weights = [[] for _ in range(n)]
                
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2o_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2o_bias, dtype=dtype, device=device))

                for name,weight in zip(i2o_names, weights): setattr(self, name, nn.ModuleList(weight))
                
                names=names+o2o_names
                n = len(o2o_names)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                for i,(in_features, cat_features, out_features) in enumerate(zip(self.i2h_sizes, self.i2o_sizes, self.o2o_sizes)):
                    if i==is_last:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.o2o_bias, dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.o2o_bias, dtype=dtype, device=device))

                for name,weight in zip(o2o_names, weights): setattr(self, name, nn.ModuleList(weight))
                
            else:
                names=i2h_names
                input_sizes=(self.input_size,) + self.i2o_sizes[:-1]
                n = len(i2h_names)
                weights = [[] for _ in range(n)]
                for in_features, cat_features, out_features in zip(input_sizes, self.i2h_sizes, self.i2h_sizes):
                    for j in range(n):
                        weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                self.i2h_bias, dtype=dtype, device=device))
                        
                for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
                
                names=names+i2o_names
                n = len(i2o_names)
                weights = [[] for _ in range(n)]
                is_last = len(input_sizes)-1
                for i,(in_features, cat_features, out_features) in enumerate(zip(input_sizes, self.i2h_sizes, self.i2o_sizes)):
                    if i==is_last:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.i2o_bias, dtype=dtype, device=device))
                    else:
                        for j in range(n):
                            weights[j].append(nn.Linear(in_features + cat_features, out_features, 
                                    self.i2o_bias, dtype=dtype, device=device))

                for name,weight in zip(i2o_names, weights): setattr(self, name, nn.ModuleList(weight))

        else:
            names=i2h_names
            n = len(i2h_names)
            weights = [[] for _ in range(n)]
            input_sizes=(self.input_size,) + self.i2h_sizes[:-1]
            for in_features, out_features in zip(input_sizes, self.i2h_sizes):
                for j in range(n):
                    #print(f'{j=}, {in_features=}, {out_features=}')
                    weights[j].append(nn.Linear(in_features + out_features, out_features, 
                            self.i2h_bias, dtype=dtype, device=device))

            for name,weight in zip(i2h_names, weights): setattr(self, name, nn.ModuleList(weight))
        return tuple([getattr(self, name) for name in names])

    def _build_activations_(self, activation_arg, name):
        if activation_arg is None: activation_arg=self.no_act
        if hasattr(activation_arg, '__len__'):
            # activation_arg is like activation_arg=(nn.Tanh, {})
            actModule = activation_arg[0]
            actArgs = activation_arg[1]
            setattr(self, name, actModule(**actArgs))
        else:
            # activation_arg is like activation_arg=tt.tanh
            setattr(self, name, activation_arg)

    def reset_parameters(self):
        for modulelist in self.parameters_module:
            for hs,m in zip(self.i2h_sizes,modulelist):
                stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
                for w in m.parameters(): nn.init.uniform_(w, -stdv, stdv)

    def init_states_no_cell_state(self, batch_size, dtype=None, device=None):
        return \
        ([tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes],)

    def init_states_has_cell_state(self, batch_size, dtype=None, device=None):
        return \
        ([tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes],
        [tt.zeros(size=(batch_size, hs), dtype=dtype, device=device) for hs in self.i2h_sizes])

    def forward_one_i2h_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, _ = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = tt.dropout(h_, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_i2h_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, _ = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = tt.dropout(h_, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
    
    def forward_one_i2o_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, xh = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = self.lastA(self.i2oL[i]( xh ) ) if i==self.n_last else self.i2oA(self.i2oL[i]( xh ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)
    
    def forward_one_i2o_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, xh = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = self.lastA(self.i2oL[i]( xh ) ) if i==self.n_last else self.i2oA(self.i2oL[i]( xh ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
        
    def forward_one_o2o_with_cell_state(self, x, s):
        H,C = [], []
        h,c = s
        for i in range(self.n_hidden):
            h_, c_, xh = self.i2h_logic(x, h[i], c[i], i)
            H.append(h_)
            C.append(c_)
            x = tt.concat( (self.i2oA(self.i2oL[i]( xh )), h_), dim=-1)
            x = self.lastA(self.o2oL[i]( x ) ) if i==self.n_last else self.o2oA(self.o2oL[i]( x ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,C)

    def forward_one_o2o_no_cell_state(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            h_, xh = self.i2h_logic(x, h[i], i)
            H.append(h_)
            x = tt.concat( (self.i2oA(self.i2oL[i]( xh )), h_), dim=-1)
            x = self.lastA(self.o2oL[i]( x ) ) if i==self.n_last else self.o2oA(self.o2oL[i]( x ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
        return x, (H,)
    
    def forward(self, Xt, h=None, future=0):
        r""" Applies forward pass through the entire input sequence 
        
        :param Xt:  input sequence
        :param H:   hidden states from previous timestep
        :param future:  Number of future timesteps to predict, works only when ``input_size == output_size``
        """
        if h is None: h=self.init_states(Xt.shape[self.batch_dim], Xt.dtype, Xt.device)
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
                ll[i].weight.copy_(
                        tt.cat((ihW[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i],:],
                        hhW[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i],:]), dim=1)
                        )
            if self.i2h_bias:
                ihB = sd[f'bias_ih_l{i}']
                hhB = sd[f'bias_hh_l{i}']

                for n, ihw in enumerate(self.W_names):
                    ll = getattr(self, ihw)
                    ll[i].bias.copy_(
                            ihB[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i]] + \
                            hhB[n*self.i2h_sizes[i]:(n+1)*self.i2h_sizes[i]]
                            )

    
class ELMAN(RNN):
    r"""
    Defines `Elman RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ihL', ) 
        act_names =     ('actX', )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.tanh, )
        assert len(i2h_activations)==1, f'need 1 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        h_ = self.actX(self.ihL[i]( xh ))
        return h_, xh


class GRU(RNN):
    r"""
    Defines `GRU RNN <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('irL', 'izL',  'inL')
        act_names =     ('actR','actZ', 'actN', )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==3, f'need 3 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        R = self.actR(self.irL[i]( xh ))
        Z = self.actZ(self.izL[i]( xh ))
        xr = tt.concat( (xi, R*hi), dim=-1)
        N = self.actN(self.inL[i]( xr ))
        h_ = (1-Z) * N + (Z * hi)  #x = (1-Z) * h[i] + (Z * N) 
        return h_, xh


class LSTM(RNN):
    r"""
    Defines `LSTM RNN <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`__
    """
    
    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=True
        i2h_names =     ('iiL', 'ifL', 'igL', 'ioL')
        act_names =     ('actI','actF', 'actG', 'actO', 'actC' )
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.sigmoid, tt.tanh, tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==5, f'need 5 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam

    def i2h_logic(self, xi, hi, ci, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        I = self.actI(self.iiL[i]( xh ))
        F = self.actF(self.ifL[i]( xh ))
        G = self.actG(self.igL[i]( xh ))
        O = self.actO(self.ioL[i]( xh ))
        c_ = F*ci + I*G
        h_ = O * self.actC(c_)
        return h_, c_, xh


class MGU(RNN):
    r"""
    Defines `MGU RNN <https://arxiv.org/pdf/1603.09420.pdf>`__
    """

    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ifL', 'igL')
        act_names =     ('actF', 'actG')
        hyperparam=dict()
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==2, f'need 2 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
    
    def i2h_logic(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        F = self.actF(self.ifL[i]( xh )) #self.ifL[i]( xh - self.beta ) 
        xf = tt.concat( (xi, F*hi), dim=-1)
        G = self.actG(self.igL[i]( xf ))
        h_ = (1-F)*hi + F*G
        return h_, xh
    

class JANET(RNN):
    r"""
    Defines `JANET RNN <https://arxiv.org/pdf/1804.04849.pdf>`__
    """

    def make_parameters(self, i2h_activations, **hypers):
        has_cell_state=False
        i2h_names =     ('ifL', 'igL')
        act_names =     ('actF', 'actG')
        hyperparam=dict(beta = hypers.pop('beta', 1.0))
        self.i2h_logic = self.i2h_logic_without_beta  if hyperparam['beta'] == 0 else self.i2h_logic_with_beta
        if not i2h_activations: i2h_activations = (tt.sigmoid, tt.tanh, )
        assert len(i2h_activations)==2, f'need 2 activation for {__class__}'
        return has_cell_state, i2h_names, i2h_activations, act_names, hyperparam
        
    def i2h_logic_without_beta(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        F = self.actF(self.ifL[i]( xh )) #self.ifL[i]( xh - self.beta ) 
        G = self.actG(self.igL[i]( xh ))
        h_ = F*hi + (1-F)*G
        return h_, xh
    
    def i2h_logic_with_beta(self, xi, hi, i ):
        xh = tt.concat( (xi, hi), dim=-1)
        S = self.ifL[i]( xh )
        F = self.actF( S ) 
        F_ = self.actF( S - self.beta) 
        G = self.actG(self.igL[i]( xh ))
        h_ = F*hi + (1-F_)*G
        return h_, xh

