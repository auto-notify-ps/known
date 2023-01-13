
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =
#import torch as tt
import torch.nn as nn
#import torch.nn.functional as ff
#import math
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

