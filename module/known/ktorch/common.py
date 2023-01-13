# -----------------------------------------------------------------------------------------------------
import torch as tt
import torch.nn as nn
from io import BytesIO
# -----------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------
# basic
# ------------------------------------------------------------------
def numel(shape): 
    r""" returns no of total elements (or addresses) in a multi-dim array 
        Note: for torch tensor use Tensor.numel()"""
    return tt.prod(tt.tensor(shape)).item()

def arange(shape, start=0, step=1, dtype=None): 
    r""" returns arange for multi-dimensional array (reshapes) """
    return tt.arange(start=start, end=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

def shares_memory(a, b) -> bool: 
    r""" checks if two tensors share same underlying storage, in which case, changing values of one will change values in other as well
        Note: this is different from Tensor.is_set_to(Tensor) function which checks shape as well"""
    return (a.storage().data_ptr() == b.storage().data_ptr())
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# module related
# ------------------------------------------------------------------

@tt.no_grad()
def copy_parameters(module_from, module_to):
    r""" copies only parameters, both models are supposed to be identical """
    for pt,pf in zip(module_to.parameters(), module_from.parameters()): pt.copy_(pf)

@tt.no_grad()
def show_parameters(module, values=False):
    r""" prints the parameters using `nn.Module.parameters` iterator, use `values=True` to print full parameter tensor """
    nparam=0
    for i,p in enumerate(module.parameters()):
        iparam = p.numel()
        nparam += iparam
        print(f'#[{i+1}]\tShape[{p.shape}]\tParams: {iparam}')
        if values: print(f'{p}')
    print(f'Total Parameters: {nparam}')
    return nparam

@tt.no_grad()
def show_dict(module, values=False):
    r""" prints the parameters using `nn.Module.parameters` iterator, use `values=True` to print full parameter tensor """
    sd = module.state_dict()
    for i,(k,v) in enumerate(sd.items()):
        
        print(f'#[{i+1}]\t[{k}]\tShape[{v.shape}]')
        if values: print(f'{v}')
    return 

@tt.no_grad()
def diff_parameters(module1, module2, do_abs=True, do_sum=True):
    d = [ (abs(p1 - p2) if do_abs else (p1 - p2)) for p1,p2 in zip(module1.parameters(), module2.parameters()) ]
    if do_sum: d = [ tt.sum(p) for p in d  ]
    return d

def save_state(path, model): tt.save(model.state_dict(), path) # simply save the state dictionary

def load_state(path, model): model.load_state_dict(tt.load(path)) # simply load the state dictionary

def make_clone(model, detach=False, set_eval=False):
    r""" Clone a model using memory buffer
        NOTE: use `detach=True` to sets the `requires_grad` to `False` on all of the parameters of the cloned model. 
        NOTE: use `set_eval=True` to call `eval()` on the cloned model. 

        Returns: nn.Module or any torch object
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

def make_clones(model, n_copies, detach=False, set_eval=False):
    r""" Clone a model multiple times using memory buffer
        NOTE: use `detach=True` to sets the `requires_grad` to `False` on all of the parameters of the cloned model. 
        NOTE: use `set_eval=True` to call `eval()` on the cloned model. 

        Returns: Tuple of (nn.Module or any torch object)
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

def clone_model(model, n_copies=1, detach=False, set_eval=False):
    r""" Clone a model using memory buffer

        Args:
            model       `nn.Module/object`  : a model or any torch object that can be saved/loaded using torch.save/load
            n_copies    `integer`           : no of copies, default = 1
            detach      `bool`              : if True, sets the `requires_grad` to `False` on all of the parameters of the cloned model. 
            set_eval    `bool`              : if True, calls `eval()` on the cloned model. 

        Returns: 
            nn.Module or any torch object               if n_copies=1
            
            Tuple of (nn.Module or any torch object)    if n_copies>1
    """
    assert n_copies>0, f'no of copies must be atleast one'
    return (make_clone(model, detach, set_eval) if n_copies==1 else make_clones(model, n_copies, detach, set_eval))

def dense_sequential(in_dim, layer_dims, out_dim, actF, actL, actFA={}, actLA={}, use_bias=True, use_biasL=True, dtype=None, device=None ):
    r"""
    Creats a stack of fully connected (dense) layers which is usually connected at end of other networks
    Args:
        in_dim          `integer`       : in_features or input_size
        layer_dims      `List/Tuple`    : size of hidden layers
        out_dim         `integer`       : out_features or output_size
        actF            `nn.Module`     : activation function at hidden layer
        actFA           `dict`          : args while initializing actF
        actL            `nn.Module`     : activation function at last layer
        actLA           `dict`          : args while initializing actL
        use_bias        `bool`          : if True, uses bias at hidden layers
        use_biasL       `bool`          : if True, uses bias at last layer

    Returns:
        `nn.Module` : an instance of nn.Sequential
    """
    layers = []
    # first layer
    layers.append(nn.Linear(in_dim, layer_dims[0], bias=use_bias, dtype=dtype, device=device))
    if actF is not None: layers.append(actF(**actFA))
    # remaining layers
    for i in range(len(layer_dims)-1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=use_bias, dtype=dtype, device=device))
        if actF is not None: layers.append(actF(**actFA))
    # last layer
    layers.append(nn.Linear(layer_dims[-1], out_dim, bias=use_biasL, dtype=dtype, device=device))
    if actL is not None: layers.append(actL(**actLA))
    return nn.Sequential( *layers )

