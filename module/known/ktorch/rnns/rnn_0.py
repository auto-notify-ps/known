
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =
import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import math
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =

__all__ = [
    'RNN', 'ELMAN', 'GRU', 'LSTM', 'JANET', 'MGU',
]

class RNN(nn.Module):

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
        self.parameters_weight, self.parameters_bias = self.build_parameters(dtype, device)
        self.reset_parameters() # reset_parameters should be the lass call before exiting __init__

    def build_parameters(self, dtype, device):
        raise NotImplemented
        self.w, self.b = nn.ParameterList([]), nn.ParameterList([])
        return (self.w,), (self.b,) # should return (self.parameters_weight, self.parameters_bias)

    def reset_parameters(self):
        for weight, bias in zip(self.parameters_weight, self.parameters_bias):
            for i,hs in enumerate(self.hidden_sizes):
                stdv = 1.0 / math.sqrt(hs) if hs > 0 else 0
                for w in weight[i]: nn.init.uniform_(w, -stdv, stdv)
                if bias[i] is not None:
                    for b in bias[i]:   nn.init.uniform_(b, -stdv, stdv)

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

    @tt.no_grad()
    def copy_torch(self, model):
        sd = model.state_dict()
        for i in range(self.n_hidden):
            ihW, hhW = sd[f'weight_ih_l{i}'], sd[f'weight_hh_l{i}']

            for n, (iw, hw) in enumerate(self.W_names):
                getattr(self, iw)[i].copy_(ihW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:])
                getattr(self, hw)[i].copy_(hhW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:])
                
            if self.input_bias:
                ihB = sd[f'bias_ih_l{i}']
                for n, (ib, hb) in enumerate(self.B_names): 
                    getattr(self, ib)[i].copy_(ihB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]])

            if self.hidden_bias:
                hhB = sd[f'bias_hh_l{i}']
                for n, (ib, hb) in enumerate(self.B_names): 
                    getattr(self, hb)[i].copy_(hhB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]])

    @tt.no_grad()
    def diff_torch(self, model):
        sd = model.state_dict()
        dd = []
        for i in range(self.n_hidden):
            ihW, hhW = sd[f'weight_ih_l{i}'], sd[f'weight_hh_l{i}']

            for n, (iw, hw) in enumerate(self.W_names):
                dd.append(getattr(self, iw)[i]-(ihW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:]))
                dd.append(getattr(self, hw)[i]-(hhW[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i],:]))
                
            if self.input_bias:
                ihB = sd[f'bias_ih_l{i}']
                for n, (ib, hb) in enumerate(self.B_names): 
                    dd.append(getattr(self, ib)[i]-(ihB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]]))

            if self.hidden_bias:
                hhB = sd[f'bias_hh_l{i}']
                for n, (ib, hb) in enumerate(self.B_names): 
                    dd.append(getattr(self, hb)[i]-(hhB[n*self.hidden_sizes[i]:(n+1)*self.hidden_sizes[i]]))
        return dd


class ELMAN(RNN):

    def __init__(self, input_bias, hidden_bias, actF, **rnnargs) -> None:
        self.input_bias, self.hidden_bias = input_bias, hidden_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):

        self.W_names = ( ('ihW', 'hhW'), )
        self.B_names = ( ('ihB', 'hhB'), )

        ihW, ihB, hhW, hhB = [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ihW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            ihB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hhW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hhB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )
        self.ihW, self.ihB, self.hhW, self.hhB = \
            nn.ParameterList(ihW), nn.ParameterList(ihB), nn.ParameterList(hhW), nn.ParameterList(hhB)
        return \
            (self.ihW, self.hhW), \
            (self.ihB, self.hhB)

    def forward_one(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            x = self.actF( ff.linear(x, self.ihW[i], self.ihB[i]) + ff.linear(h[i], self.hhW[i], self.hhB[i]) )
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
        return x, (H,)


class GRU(RNN):

    def __init__(self, input_bias, hidden_bias, actF, **rnnargs) -> None:
        self.input_bias, self.hidden_bias = input_bias, hidden_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):

        self.W_names = ( ('irW', 'hrW'), ('izW', 'hzW'), ('inW', 'hnW'), )
        self.B_names = ( ('irB', 'hrB'), ('izB', 'hzB'), ('inB', 'hnB'), )

        irW, irB, hrW, hrB, izW, izB, hzW, hzB, inW, inB, hnW, hnB = \
            [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            irW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            irB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hrW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hrB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            izW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            izB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hzW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hzB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            inW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            inB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hnW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hnB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

        self.irW, self.irB, self.hrW, self.hrB = \
            nn.ParameterList(irW), nn.ParameterList(irB), nn.ParameterList(hrW), nn.ParameterList(hrB)
        self.izW, self.izB, self.hzW, self.hzB = \
            nn.ParameterList(izW), nn.ParameterList(izB), nn.ParameterList(hzW), nn.ParameterList(hzB)
        self.inW, self.inB, self.hnW, self.hnB = \
            nn.ParameterList(inW), nn.ParameterList(inB), nn.ParameterList(hnW), nn.ParameterList(hnB)
        return \
            (self.irW, self.hrW, self.izW, self.hzW, self.inW, self.hnW ), \
            (self.irB, self.hrB, self.izB, self.hzB, self.inB, self.hnB )

    def forward_one(self, x, s):
        H = []
        h, = s
        for i in range(self.n_hidden):
            R = tt.sigmoid(ff.linear(x, self.irW[i], self.irB[i]) + ff.linear(h[i], self.hrW[i], self.hrB[i]))
            Z = tt.sigmoid(ff.linear(x, self.izW[i], self.izB[i]) + ff.linear(h[i], self.hzW[i], self.hzB[i]))
            N = self.actF(ff.linear(x, self.inW[i], self.inB[i]) + (R * ff.linear(h[i], self.hnW[i], self.hnB[i])))
            #N = self.actF(ff.linear(x, self.inW[i], self.inB[i]) + ff.linear(R * h[i], self.hnW[i], self.hnB[i]))
            x = (1-Z) * N + (Z * h[i])  #x = (1-Z) * h[i] + (Z * N) 
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            
            H.append(x)
        return x, (H,)


class LSTM(RNN):

    def __init__(self, input_bias, hidden_bias, actF, actC, **rnnargs) -> None:
        self.input_bias, self.hidden_bias = input_bias, hidden_bias
        self.actF, self.actC = actF, actC
        super().__init__(n_states=2, **rnnargs)

    def build_parameters(self, dtype, device):
        
        self.W_names = ( ('iiW', 'hiW'), ('ifW', 'hfW'), ('igW', 'hgW'), ('ioW', 'hoW'), )
        self.B_names = ( ('iiB', 'hiB'), ('ifB', 'hfB'), ('igB', 'hgB'), ('ioB', 'hoB'), )

        iiW, iiB, hiW, hiB, ifW, ifB, hfW, hfB, igW, igB, hgW, hgB, ioW, ioB, hoW, hoB = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            iiW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            iiB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hiW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hiB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            ifW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            ifB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hfW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hfB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            igW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            igB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hgW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hgB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            ioW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            ioB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hoW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hoB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

        self.iiW, self.iiB, self.hiW, self.hiB = \
            nn.ParameterList(iiW), nn.ParameterList(iiB), nn.ParameterList(hiW), nn.ParameterList(hiB)
        self.ifW, self.ifB, self.hfW, self.hfB = \
            nn.ParameterList(ifW), nn.ParameterList(ifB), nn.ParameterList(hfW), nn.ParameterList(hfB)
        self.igW, self.igB, self.hgW, self.hgB = \
            nn.ParameterList(igW), nn.ParameterList(igB), nn.ParameterList(hgW), nn.ParameterList(hgB)
        self.ioW, self.ioB, self.hoW, self.hoB = \
            nn.ParameterList(ioW), nn.ParameterList(ioB), nn.ParameterList(hoW), nn.ParameterList(hoB)
        return \
            (self.iiW, self.hiW, self.ifW, self.hfW, self.igW, self.hgW, self.ioW, self.hoW ), \
            (self.iiB, self.hiB, self.ifB, self.hfB, self.igB, self.hgB, self.ioB, self.hoB )

    def forward_one(self, x, s):
        H,C=[],[]
        h,c = s
        for i in range(self.n_hidden):
            I = tt.sigmoid(ff.linear(x, self.iiW[i], self.iiB[i]) + ff.linear(h[i], self.hiW[i], self.hiB[i]))
            F = tt.sigmoid(ff.linear(x, self.ifW[i], self.ifB[i]) + ff.linear(h[i], self.hfW[i], self.hfB[i]))
            G = self.actF(ff.linear(x, self.igW[i], self.igB[i]) + ff.linear(h[i], self.hgW[i], self.hgB[i]))
            O = tt.sigmoid(ff.linear(x, self.ioW[i], self.ioB[i]) + ff.linear(h[i], self.hoW[i], self.hoB[i]))
            c_ = F*c[i] + I*G
            x = O * self.actC(c_)
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
            C.append(c_)
        return x, (H,C)


class JANET(RNN):

    def __init__(self, input_bias, hidden_bias, actF, beta, **rnnargs) -> None:
        self.input_bias, self.hidden_bias = input_bias, hidden_bias
        self.actF = actF
        self.beta = beta
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):

        self.W_names = ( ('ifW', 'hfW'), ('igW', 'hgW'), )
        self.B_names = ( ('ifB', 'hfB'), ('igB', 'hgB'), )

        ifW, ifB, hfW, hfB, igW, igB, hgW, hgB = \
            [], [], [], [], [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ifW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            ifB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hfW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hfB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            igW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            igB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hgW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hgB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

        self.ifW, self.ifB, self.hfW, self.hfB = \
            nn.ParameterList(ifW), nn.ParameterList(ifB), nn.ParameterList(hfW), nn.ParameterList(hfB)
        self.igW, self.igB, self.hgW, self.hgB = \
            nn.ParameterList(igW), nn.ParameterList(igB), nn.ParameterList(hgW), nn.ParameterList(hgB)
        return \
            (self.ifW, self.hfW, self.igW, self.hgW ), \
            (self.ifB, self.hfB, self.igB, self.hgB )

    def forward_one(self, x, s):
        H=[]
        h, = s
        for i in range(self.n_hidden):
            F = tt.sigmoid(ff.linear(x, self.ifW[i], self.ifB[i]) + ff.linear(h[i], self.hfW[i], self.hfB[i]) - self.beta)
            G = self.actF(ff.linear(x, self.igW[i], self.igB[i]) + ff.linear(h[i], self.hgW[i], self.hgB[i]))
            x = F*h[i] + (1-F)*G
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            H.append(x)
        return x, (H,)


class MGU(RNN):

    def __init__(self, input_bias, hidden_bias, actF, **rnnargs) -> None:
        self.input_bias, self.hidden_bias = input_bias, hidden_bias
        self.actF = actF
        super().__init__(n_states=1, **rnnargs)

    def build_parameters(self, dtype, device):

        self.W_names = ( ('ifW', 'hfW'), ('igW', 'hgW'), )
        self.B_names = ( ('ifB', 'hfB'), ('igB', 'hgB'), )

        ifW, ifB, hfW, hfB, igW, igB, hgW, hgB = \
            [], [], [], [], [], [], [], []
        for i in range(1, len(self.layer_sizes)):
            in_features, out_features = self.layer_sizes[i-1], self.layer_sizes[i]
            ifW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            ifB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hfW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hfB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

            igW.append( nn.Parameter( tt.empty(size=(out_features, in_features ),  dtype=dtype, device=device) ) )
            igB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.input_bias else None )
            hgW.append( nn.Parameter( tt.empty(size=(out_features, out_features),  dtype=dtype, device=device) ) )
            hgB.append( nn.Parameter( tt.empty(size=(out_features,             ),  dtype=dtype, device=device) ) \
                        if self.hidden_bias else None )

        self.ifW, self.ifB, self.hfW, self.hfB = \
            nn.ParameterList(ifW), nn.ParameterList(ifB), nn.ParameterList(hfW), nn.ParameterList(hfB)
        self.igW, self.igB, self.hgW, self.hgB = \
            nn.ParameterList(igW), nn.ParameterList(igB), nn.ParameterList(hgW), nn.ParameterList(hgB)
        return \
            (self.ifW, self.hfW, self.igW, self.hgW ), \
            (self.ifB, self.hfB, self.igB, self.hgB )

    def forward_one(self, x, s):
        H=[]
        h, = s
        for i in range(self.n_hidden):
            F = tt.sigmoid(ff.linear(x, self.ifW[i], self.ifB[i]) + ff.linear(h[i], self.hfW[i], self.hfB[i]) )
            G = self.actF(ff.linear(x, self.igW[i], self.igB[i]) + (F * ff.linear(h[i], self.hgW[i], self.hgB[i])))
            # or G = self.actF(ff.linear(x, self.igW[i], self.igB[i]) + ff.linear(F * h[i], self.hgW[i], self.hgB[i]))
            x = (1-F)*h[i] + F*G
            x = tt.dropout(x, self.dropouts[i], self.training) #<--- dropout only output
            # or x = F*h[i] + (1-F)*G
            H.append(x)
        return x, (H,)

