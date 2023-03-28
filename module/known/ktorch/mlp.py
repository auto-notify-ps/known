#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/ktorch/mlp.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'MLP', 'MLPn', 'DLP',
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
import torch.nn as nn
from .common import dense_sequential
from typing import Any, Union, Iterable, Callable, Dict, Tuple, List
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class MLP(nn.Module):
    r""" Multi layer Perceptron with 1 input vector and 1 output vector 

    :param in_dim: in_features or input_size
    :param layer_dims: size of hidden layers
    :param out_dim: out_features or output_size
    :param actF: activation function at hidden layer 
    :param actL: activation function at last layer 
    """

    def __init__(self, in_dim:int, layer_dims:Iterable[int], out_dim:int, 
                actF:Union[None,Iterable], actL:Union[None,Iterable], 
                dtype=None, device=None):
        r"""
        :param in_dim: in_features or input_size
        :param layer_dims: size of hidden layers
        :param out_dim: out_features or output_size
        :param actF: activation function at hidden layer 
        :param actL: activation function at last layer 
        """
        super().__init__()
        self.model = dense_sequential(in_dim, layer_dims, out_dim, actF, actL, dtype=dtype, device=device )

    def forward(self, x):
        r""" Forward pass 

        :shapes: 
            * input has a shape ``(batch_size, input_size)``
            * output has shape ``(batch_size, output_size)``
        """
        return self.model(x)


class MLPn(nn.Module):
    r""" Multi layer Perceptron with n input vectors and 1 output vector 
    
    :param in_dims: in_features or input_size
    :param layer_dims: size of hidden layers
    :param out_dim: out_features or output_size
    :param actF: activation function at hidden layer 
    :param actL: activation function at last layer 
    """

    def __init__(self, in_dims:Iterable[int], layer_dims:Iterable[int], out_dim:int, 
                actF:Union[None,Iterable], actL:Union[None,Iterable], 
                dtype=None, device=None):
        r"""
        :param in_dims: in_features or input_size
        :param layer_dims: size of hidden layers
        :param out_dim: out_features or output_size
        :param actF: activation function at hidden layer 
        :param actL: activation function at last layer 
        """
        in_dim = sum(in_dims)
        assert in_dim > 0, f'at least on dimension is required!'
        super().__init__()
        self.model = dense_sequential(in_dim, layer_dims, out_dim, actF, actL, dtype=dtype, device=device)

    def forward(self, x): #<--- here x is a tuple, this is easier to use with nn.Sequential
        r""" Forward pass 

        :shapes: 
            * input is a n-tuple where each element has a shape ``(batch_size, input_size)``
            * output has shape ``(batch_size, output_size)``
        """
        return self.model(tt.concat(x, dim=-1))

    
class DLP(nn.Module):
    r""" Decoupled Multi layer Perceptron for dueling-DQN architecture.
    This Module has 2 Decoupled networks. The `base` network takes the input. 
    The output of `base` network is taken by `vnet` and `anet` networks.
    `vnet` represents the value-network which gives the value of a state (in Q-learning)

    :param in_dim: in_features or input_size
    :param layer_dims_net: size of hidden layers in base net
    :param out_dim_net: out_features or output_size in base net
    :param actF_net: activation function at hidden layer in base net 
    :param actL_net: activation function at last layer of base net 
    :param layer_dims_vnet: size of hidden layers in value net
    :param actF_vnet: activation function at hidden layer in value net 
    :param actL_vnet: activation function at last layer in value net 
    :param layer_dims_anet: size of hidden layers in a-net
    :param actF_anet: activation function at hidden layer in a-net 
    :param actL_anet: activation function at last layer in a-net 
    :param out_dim_net: out_features or output_size in a-net (final output)
    
    :ref: `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`__
    """

    def __init__(self, 

        in_dim:int, 

        layer_dims_net:Iterable[int], 
        out_dim_net:int,
        actF_net:Union[None,Iterable], 
        actL_net:Union[None,Iterable],

        layer_dims_vnet:Iterable[int], 
        actF_vnet:Union[None,Iterable], 
        actL_vnet:Union[None,Iterable], 

        layer_dims_anet:Iterable[int], 
        actF_anet:Union[None,Iterable], 
        actL_anet:Union[None,Iterable], 

        out_dim:int, 
        dtype=None, device=None
        ):
        r"""
        :param in_dim: in_features or input_size
        :param layer_dims_net: size of hidden layers in base net
        :param out_dim_net: out_features or output_size in base net
        :param actF_net: activation function at hidden layer in base net 
        :param actL_net: activation function at last layer of base net 
        :param layer_dims_vnet: size of hidden layers in value net
        :param actF_vnet: activation function at hidden layer in value net 
        :param actL_vnet: activation function at last layer in value net 
        :param layer_dims_anet: size of hidden layers in a-net
        :param actF_anet: activation function at hidden layer in a-net 
        :param actL_anet: activation function at last layer in a-net 
        :param out_dim_net: out_features or output_size in a-net (final output)
        """
        super().__init__()
        self.net = dense_sequential(in_dim, layer_dims_net, out_dim_net, actF_net, actL_net, dtype=dtype, device=device)
        self.vnet = dense_sequential(out_dim_net, layer_dims_vnet, 1, actF_vnet, actL_vnet, dtype=dtype, device=device)
        self.anet = dense_sequential(out_dim_net, layer_dims_anet, out_dim, actF_anet, actL_anet, dtype=dtype, device=device)
    
    def forward(self, x):
        r""" Forward pass 

        :shapes: 
            * input has shape ``(batch_size, input_size)``
            * output has shape ``(batch_size, output_size)``
        """
        net_out = self.net(x)
        v_out = self.vnet(net_out)
        a_out = self.anet(net_out)
        return v_out + (a_out -  tt.mean(a_out, dim = -1, keepdim=True))

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=