__doc__=r"""
:py:mod:`known/ktorch/common.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'numel', 'arange', 'shares_memory', 
    'copy_parameters', 'show_parameters', 'diff_parameters', 'show_dict',
    'copy_state', 'save_state', 'load_state', 'make_clone', 'make_clones', 'clone_model', 
    'no_activation', 'build_activation', 'dense_sequential',
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
from torch import Tensor
import torch.nn as nn
from io import BytesIO
from typing import Any, Union, Iterable, Callable, Dict, Tuple, List
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def numel(shape)->int: 
    r""" Returns the number of elements in an array of given shape

    .. note:: for ``torch.Tensor`` use ``Tensor.numel()``
    """
    return tt.prod(tt.tensor(shape)).item()

def arange(shape, start:int=0, step:int=1, dtype=None) -> Tensor: 
    r""" Similar to ``torch.arange`` but reshapes the tensor to given shape """
    return tt.arange(start=start, end=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

def shares_memory(a, b) -> bool: 
    r""" Checks if two tensors share same underlying storage, in which case, changing values of one will change values in the other as well.

    .. note:: This is different from ``Tensor.is_set_to(Tensor)`` function which checks the shape as well.
    """
    return (a.storage().data_ptr() == b.storage().data_ptr())

@tt.no_grad()
def copy_parameters(module_from, module_to) -> None:
    r""" Copies module parameters, both modules are supposed to be identical """
    for pt,pf in zip(module_to.parameters(), module_from.parameters()): pt.copy_(pf)
    
@tt.no_grad()
def show_parameters(module, values:bool=False) -> int:
    r""" Prints the parameters using ``nn.Module.parameters()``

    :param module: an instance of ``nn.Module``
    :param values: If `True`, prints the full parameter tensor 

    :returns:   total number of parameters in the module
    """
    nparam=0
    for i,p in enumerate(module.parameters()):
        iparam = p.numel()
        nparam += iparam
        print(f'#[{i+1}]\tShape[{p.shape}]\tParams: {iparam}')
        if values: print(f'{p}')
    print(f'Total Parameters: {nparam}')
    return nparam

@tt.no_grad()
def show_dict(module, values:bool=False) -> None:
    r""" Prints the state dictionary using ``nn.Module.state_dict()``
    
    :param module: an instance of ``nn.Module``
    :param values: If `True`, prints the full state tensor 
    """
    sd = module.state_dict()
    for i,(k,v) in enumerate(sd.items()):
        
        print(f'#[{i+1}]\t[{k}]\tShape[{v.shape}]')
        if values: print(f'{v}')
    return 

@tt.no_grad()
def diff_parameters(module1, module2, do_abs:bool=True, do_sum:bool=True) -> List:
    r""" Checks the difference between the parameters of two modules. This can be used to check if two models have exactly the same parameters.

    :param module1: an instance of ``nn.Module``
    :param module: an instance of ``nn.Module``
    :param do_abs: if True, finds the absolute difference
    :param do_sum: if True, finds the sum of difference

    :returns: a list of differences in each parameter or their sum if ``do_sum`` is True.
    """
    d = [ (abs(p1 - p2) if do_abs else (p1 - p2)) for p1,p2 in zip(module1.parameters(), module2.parameters()) ]
    if do_sum: d = [ tt.sum(p) for p in d  ]
    return d

@tt.no_grad()
def copy_state(module_from, module_to):
    r""" simply copy the state dictionary """
    module_to.load_state_dict(module_from.state_dict())

def save_state(model, path:str): 
    r""" simply save the state dictionary """
    tt.save(model.state_dict(), path) 

def load_state(model, path:str): 
    r""" simply load the state dictionary """
    model.load_state_dict(tt.load(path))

def make_clone(model, detach:bool=False, set_eval:bool=False):
    r""" Clone a model using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: ``nn.Module``

    .. seealso::
        :func:`~known.ktorch.common.make_clones`
        :func:`~known.ktorch.common.clone_model`
    """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters(): p.requires_grad=False
    if set_eval: model_copy.eval()
    buffer.close()
    del buffer
    return model_copy

def make_clones(model, n_copies:int, detach:bool=False, set_eval:bool=False):
    r""" Clone a model multiple times using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param n_copies: number of copies to be made
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: tuple of ``nn.Module``

    .. seealso::
        :func:`~known.ktorch.common.make_clone`
        :func:`~known.ktorch.common.clone_model`
    """
    buffer = BytesIO()
    
    tt.save(model, buffer)
    model_copies = []
    for _ in range(n_copies):
        buffer.seek(0)
        model_copy = tt.load(buffer)
        if detach:
            for p in model_copy.parameters(): p.requires_grad=False
        if set_eval: model_copy.eval()
        model_copies.append(model_copy)
    buffer.close()
    del buffer
    return tuple(model_copies)

def clone_model(model, n_copies:int=1, detach:bool=False, set_eval:bool=False):
    r""" Clone a model multiple times using memory buffer

    :param model:    an ``nn.Module`` to clone
    :param n_copies: number of copies to be made
    :param detach:   if True, sets the ``requires_grad`` to `False` on all of the parameters of the cloned model
    :param set_eval: if True, calls ``eval()`` on cloned model

    :returns: single ``nn.Module`` or tuple of ``nn.Module`` based on ``n_copies`` argument

    .. note:: This is similar to :func:`~known.ktorch.common.make_clone` and :func:`~known.ktorch.common.make_clones` but
        returns tuple or a single object based on `n_copies` argument

    """
    assert n_copies>0, f'no of copies must be atleast one'
    return (make_clone(model, detach, set_eval) if n_copies==1 else make_clones(model, n_copies, detach, set_eval))

def no_activation(x): return x

def build_activation( activation, default=None):
    r""" Build an activation from argument 
    
    :param activation: Can be of following types:
        * `None`                uses `no_activation` if `default=None` else uses the `default`
        * `Callable`            directly uses the callable functions (which expects no extra arguments)
        * `(nn.Module, Args)`   calls the `nn.Module` with given `Args`, eg: `activation=(nn.Softmax, {'dim':1})`
    """
    if activation is None:
        if default is None: 
            return no_activation
        else:
            return (default[0](**default[1]) if hasattr(default, '__len__') else default)
    else:
        return (activation[0](**activation[1]) if hasattr(activation, '__len__') else activation)


# NOTE: does not include in __all__
def build_activation_module( activation ):
    r""" Build an activation module from argument 
    
    :param activation: `(nn.Module, Args)`
        calls the `nn.Module` with given `Args`, eg: `activation=(nn.Softmax, {'dim':1})`
    """
    assert activation is not None, f'Argument "activation" cannot be none'
    assert hasattr(activation, '__len__'),f'Expecting an Iterable for activation'
    assert len(activation)>=2,f'Expecting a module and args dict for activation'
    return activation[0](**activation[1])
    
def dense_sequential(in_dim:int, layer_dims:Iterable[int], out_dim:int, 
                    actF:Union[None,Iterable], actL:Union[None,Iterable],
                    use_bias:bool=True, dtype=None, device=None ):
    r""" Creats a stack of fully connected (dense) layers which is usually connected at end of other networks.
    
    :param in_dim:       in_features or input_size
    :param layer_dims:   size of hidden layers (can be empty)
    :param out_dim:      out_features or output_size
    :param actF:         activation function at hidden layer 
    :param actL:         activation function at last layer 
    :param use_bias:     if True, uses bias at all layers

    :returns: An instance of ``nn.Module`` 

    """
    layers = []
    
    # first layer
    if layer_dims:
        layers.append(nn.Linear(in_dim, layer_dims[0], bias=use_bias, dtype=dtype, device=device))
        if actF is not None: layers.append(build_activation_module(actF))
        # remaining layers
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=use_bias, dtype=dtype, device=device))
            if actF is not None: layers.append(build_activation_module(actF))
        # last layer
        layers.append(nn.Linear(layer_dims[-1], out_dim, bias=use_bias, dtype=dtype, device=device))
        if actL is not None: layers.append(build_activation_module(actL))
    else:
        layers.append(nn.Linear(in_dim, out_dim, bias=use_bias, dtype=dtype, device=device))
        if actL is not None: layers.append(build_activation_module(actL))
    return nn.Sequential( *layers )

# class LinearActivated(nn.Module):
#     r""" 
#     A simple linear layer with added activation 
   
#     :param in_features:     in_features or input_size
#     :param out_features:    out_features or output_size
#     :param bias:            if True, uses bias
#     :param activation:      can be a tuple like ``(nn.Tanh, {})`` or a callable like ``torch.tanh``
    
#     :returns: An instance of ``nn.Module`` 

#     .. seealso::
#         :func:`~known.ktorch.common.dense_sequential`
#     """

#     def __init__(self, in_features:int, out_features:int, bias:bool, activation:Union[Callable, Tuple], device=None, dtype=None) -> None:
#         r""" 
#         Initialize a linear layer with added activation 
        
#         :param in_features:     in_features or input_size
#         :param out_features:    out_features or output_size
#         :param bias:            if True, uses bias
#         :param activation:      can be a tuple like ``(nn.Tanh, {})`` or a callable like ``torch.tanh``
        
#         :returns: An instance of ``nn.Module`` 
#         """
#         super().__init__()
#         def no_act(x): return x
#         if activation is None: activation=no_act
#         if hasattr(activation, '__len__'):
#             # activation_arg is like activation_arg=(nn.Tanh, {})
#             actModule = activation[0]
#             actArgs = activation[1]
#             self.A = actModule(**actArgs)
#         else:
#             # activation_arg is like activation_arg=tt.tanh
#             self.A = activation
#         self.L = nn.Linear(in_features, out_features, bias, device, dtype)
#     def forward(self, x): 
#         r""" Forward pass 

#         :shapes: 
#             * input has a shape ``(batch_size, input_size)``
#             * output has shape ``(batch_size, output_size)``
#         """
#         return self.A(self.L(x))


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=