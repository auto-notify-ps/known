

# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =
import torch as tt
import torch.nn as nn
from .common import dense_sequential
# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =

class MLP(nn.Module):
    r""" Multi layer Perceptron with 1 input vector and 1 output vector """

    def __init__(self, in_dim, layer_dims, out_dim, actF, actL, actFA={}, actLA={}, dtype=None, device=None):
        r"""
        Returns a Sequential Module
        Args:
            in_dim          `integer`       : in_features or input_size
            layer_dims      `List/Tuple`    : size of hidden layers
            out_dim         `integer`       : out_features or output_size
            actF            `nn.Module`     : activation function at hidden layer
            actFA           `dict`          : args while initializing actF
            actL            `nn.Module`     : activation function at last layer
            actLA           `dict`          : args while initializing actL

        Returns:
            `nn.Module` : an instance of nn.Sequential
        """
        super().__init__()
        self.model = dense_sequential(in_dim, layer_dims, out_dim, actF, actL, actFA, actLA, dtype=dtype, device=device )

    def forward(self, x):
        return self.model(x)


class MLPn(nn.Module):
    r""" Multi layer Perceptron with n input vectors and 1 output vector """

    def __init__(self, in_dims, layer_dims, out_dim, actF, actL, actFA={}, actLA={}, dtype=None, device=None):
        r"""
        Returns a Sequential Module
        Args:
            in_dims         `List/Tuple`    : in_features or input_size for all inputs
            layer_dims      `List/Tuple`    : size of hidden layers
            out_dim         `integer`       : out_features or output_size
            actF            `nn.Module`     : activation function at hidden layer
            actFA           `dict`          : args while initializing actF
            actL            `nn.Module`     : activation function at last layer
            actLA           `dict`          : args while initializing actL

        Returns:
            `nn.Module` : an instance of nn.Sequential

        Note:
            the `forward` takes tuple as arguments
        """
        super().__init__()
        self.model = dense_sequential(sum(in_dims), layer_dims, out_dim, actF, actL, actFA, actLA, dtype=dtype, device=device)

    def forward(self, x): #<--- here x is a tuple
        return self.model(tt.concat(x, dim=-1))


class DLP(nn.Module):
    r""" Decoupled Multi layer Perceptron for dueling-DQN architecture 

        :ref:`Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`
        """

    def __init__(self, 

        in_dim, 

        layer_dims_net, 
        out_dim_net,
        actF_net, 
        actL_net,
        actFA_net,
        actLA_net,

        layer_dims_vnet, 
        actF_vnet, 
        actL_vnet, 
        actFA_vnet,
        actLA_vnet,

        layer_dims_anet, 
        actF_anet, 
        actL_anet, 
        actFA_anet,
        actLA_anet,

        out_dim, 
        dtype=None, device=None
        ):
        r"""
        Returns a Sequential Module with 2 Decoupled networks. The `base` network takes the input. 
        The output of `base` network is taken by `vnet` and `anet` networks

        Note: `vnet` represents the value-network which gives the value of a state (in Q-learning)

        Ref: :ref:`Dueling Network Architectures for Deep Reinforcement Learning<https://arxiv.org/abs/1511.06581>`

        Args:
            in_dim          `integer`       : in_features or input_size
            layer_dims_*  `List/Tuple`    : size of hidden layers for * net
            out_dim_*     `integer`       : out_features or output_size for * net
            actF_*        `nn.Module`     : activation function at hidden layer for * net
            actFA_*       `dict`          : args while initializing actF for * net
            actL_*        `nn.Module`     : activation function at last layer for * net
            actLA_*       `dict`          : args while initializing actL for * net
            out_dim         `integer`       : out_features or output_size


        Returns:
            `nn.Module` : an instance of nn.Sequential
        """
        super().__init__()
        self.net = dense_sequential(in_dim, layer_dims_net, out_dim_net, actF_net, actL_net, actFA_net, actLA_net, dtype=dtype, device=device)
        self.vnet = dense_sequential(out_dim_net, layer_dims_vnet, 1, actF_vnet, actL_vnet, actFA_vnet, actLA_vnet, dtype=dtype, device=device)
        self.anet = dense_sequential(out_dim_net, layer_dims_anet, out_dim, actF_anet, actL_anet, actFA_anet, actLA_anet, dtype=dtype, device=device)
    
    def forward(self, x):
        net_out = self.net(x)
        v_out = self.vnet(net_out)
        a_out = self.anet(net_out)
        return v_out + (a_out -  tt.mean(a_out, dim = -1, keepdim=True))

# @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ = @ =