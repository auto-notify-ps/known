
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =
import torch as tt
import torch.nn as nn
#import torch.nn.functional as ff
import math
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =


class GRNN(nn.Module):

    """ Generalized RNN 
        - takes any core module and applies Recurance on it through forward method 
        - works with existing RNN and RNNC modules
        - when inheriting this class make sure to implement the following functions: 
            timesteps_in()
            init_states()
            get_input_at()
            forward_one()
    """

    def __init__(self, core) -> None:
        super().__init__()
        self.core = core

    def forward(self, Xt, H=None, future=0):
        # X = input sequence
        # H = hidden states for each layer at timestep (t-1)
        # future = no of future steps to output

        timesteps = self.timesteps_in(Xt) #<<---- how many timesteps in the input sequence X ? 

        # H should contain hidden states for each layer at the last timestep
        # if no hidden-state supplied, create a hidden states for each layer
        if H is None: H=self.init_states(Xt) 


        Ht = [H] #<==== hidden states at each timestep
        Yt = []  #<---- output of last cell at each timestep
        for t in range(timesteps): #<---- for each timestep 
            x = self.get_input_at(Xt, t) #<--- input at this time step
            h = Ht[-1] #<---- hidden states for each layer at this timestep
            y, h_ = self.forward_one(x, h)

            Yt.append(y)
            Ht.append(h_)
        
        #<--- IMP: future arg will work only when (input_size == hidden_size of the last layer)
        for _ in range(future):
            x = Yt[-1]#<--- input at this time step
            h = Ht[-1] #<---- hidden states for each layer at this timestep
            y, h_ = self.forward_one(x, h)

            Yt.append(y)
            Ht.append(h_)

        return Yt, Ht[-1]

    def timesteps_in(self, Xt): return Xt.shape[self.core.seq_dim]

    def init_states(self, Xt): return self.core.init_hidden(Xt.shape[self.core.batch_dim], Xt.dtype, Xt.device)
    
    def get_input_at(self, Xt, t): return (Xt[:, t, :] if self.core.batch_first else Xt[t, :, :])

    def forward_one(self, x, h): return self.core.forward_one(x, h)

class RNN(nn.Module):

    """ Concatenated versions of RNN 
        - parameters are replaced by linear gate modules that have combined weights and common bias
        - input and hidden states are concatenated before passing to linear gates
        - is a little faster than normal RNN """
        
    def __init__(self,
                input_size,         # input features
                hidden_sizes,       # hidden features at each layer
                dropout=0.0,        # dropout after each layer, only if hidden_sizes > 1
                batch_first=False,  # if true, excepts input as (batch_size, seq_len, input_size) else (seq_len, batch_size, input_size)
                stack_output=False, # if true, stack output from all timesteps, else returns a list of outputs
                dtype=None,
                device=None,
                n_states=None
                ) -> None:
        super().__init__()

        # dimensionality
        self.input_size = int(input_size)
        self.hidden_sizes = tuple(hidden_sizes)
        self.n_hidden = len(self.hidden_sizes)
        self.layer_sizes = (self.input_size,) + self.hidden_sizes
        self.batch_first = batch_first
        self.batch_dim = (0 if batch_first else 1)
        self.seq_dim = (1 if batch_first else 0)
        self.stack_output = stack_output
        self.n_states=n_states
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

    def build_parameters(self, dtype, device):
        raise NotImplemented
        self.m = nn.ModuleList([])
        return (self.m, )# should return tuple of a module list

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

class ELMAN(RNN):

    def __init__(self, has_bias, actF, **rnnargs) -> None:
        self.has_bias = has_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):
        ihL=[]
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ihL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
        self.ihL = \
            nn.ModuleList(ihL)
        return (self.ihL, )

    def forward_one(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            xh = tt.concat( (x, h[i]), dim=-1)
            x = self.actF( self.ihL[i]( xh ) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
        return x, (H,)

class GRU(RNN):

    def __init__(self, has_bias, actF, **rnnargs) -> None:
        self.has_bias = has_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):
        irL, izL, inL = \
            [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            irL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            izL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            inL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))

        self.irL, self.izL, self.inL = \
            nn.ModuleList(irL), nn.ModuleList(izL), nn.ModuleList(inL)
        return (self.irL, self.izL, self.inL )

    def forward_one(self, x, s):
        H = []
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            R = tt.sigmoid( self.irL[i]( xh ) )
            Z = tt.sigmoid( self.izL[i]( xh ) )
            xr = tt.concat( (x, R*h[i]), dim=-1)
            N = self.actF( self.inL[i]( xr ) )
            x = (1-Z) * N + (Z * h[i])  #x = (1-Z) * h[i] + (Z * N) 
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            
            H.append(x)
        return x, (H,)

class LSTM(RNN):

    def __init__(self, has_bias, actF, actC, **rnnargs) -> None:
        self.has_bias = has_bias
        self.actF, self.actC = actF, actC
        super().__init__(n_states=2, **rnnargs)

    def build_parameters(self, dtype, device):
        iiL, ifL, igL, ioL = \
            [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            iiL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            ifL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            igL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            ioL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))

        self.iiL, self.ifL, self.igL, self.ioL = \
            nn.ModuleList(iiL), nn.ModuleList(ifL), nn.ModuleList(igL), nn.ModuleList(ioL)
        return (self.iiL, self.ifL, self.igL, self.ioL )

    def forward_one(self, x, s):
        H,C=[],[]
        h,c = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            I = tt.sigmoid( self.iiL[i]( xh ) )
            F = tt.sigmoid( self.ifL[i]( xh ) )
            G = tt.sigmoid( self.igL[i]( xh ) )
            O = tt.sigmoid( self.ioL[i]( xh ) )
            c_ = F*c[i] + I*G
            x = O * self.actC(c_)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
            C.append(c_)
        return x, (H,C)

class JANET(RNN):

    def __init__(self, has_bias, actF, beta, **rnnargs) -> None:
        self.has_bias = has_bias
        self.actF = actF
        self.beta = beta
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):
        ifL, igL= \
            [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ifL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            igL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))

        self.ifL, self.igL = \
            nn.ModuleList(ifL), nn.ModuleList(igL)
        return (self.ifL, self.igL )

    def forward_one(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F = tt.sigmoid( self.ifL[i]( xh ) - self.beta)
            G = tt.sigmoid( self.igL[i]( xh ))
            x = F*h[i] + (1-F)*G
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
        return x, (H,)

class MGU(RNN):

    def __init__(self, has_bias, actF, **rnnargs) -> None:
        self.has_bias = has_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):
        ifL, igL= \
            [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ifL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))
            igL.append(nn.Linear(in_features + out_features, out_features, self.has_bias, dtype=dtype, device=device))

        self.ifL, self.igL = \
            nn.ModuleList(ifL), nn.ModuleList(igL)
        return (self.ifL, self.igL )

    def forward_one(self, x, s):
        H=[]
        h, = s
        for i in range(len(self.hidden_sizes)):
            xh = tt.concat( (x, h[i]), dim=-1)
            F = tt.sigmoid( self.ifL[i]( xh ))
            xf = tt.concat( (x, F*h[i]), dim=-1)
            G = self.actF( self.igL[i]( xf ))
            x = (1-F)*h[i] + F*G
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            # or x = F*h[i] + (1-F)*G
            H.append(x)
        return x, (H,)

