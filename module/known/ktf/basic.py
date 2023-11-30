

import torch as tt
import torch.nn as nn
from torch.nn.init import xavier_uniform_
import matplotlib.pyplot as plt

__all__ = [ 'Mod', 'Aview',]


class Mod:
    
    @staticmethod
    def Count(module, requires_grad=None): 
        r""" Counts the total number of parameters (numel) in a params
        
        :param requires_grad: 
            if None, counts all parameters
            if True, counts trainable parameters
            if False, counts non-trainiable (frozen) parameters
        """
        return sum( ([ p.numel() for p in module.parameters() ]) if requires_grad is None else \
                    ([ p.numel() for p in module.parameters()    if p.requires_grad is requires_grad ]) )

    @staticmethod
    def Show(module, values:bool=False):
        r""" Prints the parameters of a params
        
        :param values: if True, prints the full tensors otherwise prints only shape
        """
        nos_trainable, nos_frozen = 0, 0
        print('=====================================')
        for i,p in enumerate(module.parameters()):
            iparam = p.numel()
            if p.requires_grad:
                nos_trainable += iparam
            else:
                nos_frozen += iparam
            print(f'#[{i}]\tShape[{p.shape}]\tParams: {iparam}\tTrainable: {p.requires_grad}')
            if values: 
                print('=====================================')
                print(f'{p}')
                print('=====================================')
        print(f'\nTotal Parameters: {nos_trainable+nos_frozen}\tTrainable: {nos_trainable}\tFrozen: {nos_frozen}')
        print('=====================================')
        return 

    @staticmethod
    def State(module, values=False):
        r""" prints the parameters using `nn.Module.parameters` iterator, use `values=True` to print full parameter tensor """
        sd = module.state_dict()
        for i,(k,v) in enumerate(sd.items()):
            print(f'#[{i+1}]\t[{k}]\tShape[{v.shape}]')
            if values: print(f'{v}')
        return 

    @staticmethod
    def Dense(in_dim, layer_dims, out_dim, 
            actFs, bias=True, dtype=None, device=None ):
        r"""
        Creats a stack of fully connected (dense) layers which is usually connected at end of other networks
        Args:
            in_dim          `integer`       : in_features or input_size
            layer_dims      `List/Tuple`    : size of hidden layers
            out_dim         `integer`       : out_features or output_size
            actFs           `nn.Module`     : activation function at hidden layer
            bias            `bool`          : if True, uses bias at hidden layers

        Returns:
            `nn.Module` : an instance of nn.Sequential
        """
        layers = []
        # first layer
        layers.append(nn.Linear(in_dim, layer_dims[0], bias=bias, dtype=dtype, device=device))
        if actFs: layers.append(actFs.pop(0))
        # remaining layers
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=bias, dtype=dtype, device=device))
            if actFs: layers.append(actFs.pop(0))
        # last layer
        layers.append(nn.Linear(layer_dims[-1], out_dim, bias=bias, dtype=dtype, device=device))
        if actFs: layers.append(actFs.pop(0))
        return nn.Sequential( *layers )

    @staticmethod
    def Clones(module, n_copies:int):
        from io import BytesIO
        r""" Replicates a params by storing it in a buffer and retriving many copies
        NOTE: this will preserve the ```require_grad``` attribute on all tensors. """
        #from io import BytesIO
        if n_copies<1: return None
        buffer = BytesIO()
        tt.save(module, buffer)
        model_copies = []
        for _ in range(n_copies):
            buffer.seek(0)
            model_copy = tt.load(buffer)
            model_copies.append(model_copy)
        buffer.close()
        del buffer
        return model_copies

    @staticmethod
    def Clone(module): return __class__.Clones(module, 1).pop()

    @staticmethod
    def SetGrad(module, requires:bool, *names):
        r""" Sets requires_grad attribute on tensors in params
        if no names are provided, sets requires_grad on all tensors 
        NOTE: careful with *names, if a buffer's name is provided
            and it is in the state_dict then its grad will be enabled
            which is undesirable.
            not providing any names will target the parameters only
        """
        if names: # if we know which params to freeze, we can provide them
            state_dict = module.state_dict() 
            for n in names: state_dict[n].requires_grad_(requires)
        else: # if we want to do on all params
            for p in module.parameters(): p.requires_grad_(requires)
        return module

    @staticmethod
    def CausalMask(block_size, diagonal=1, dtype=None, device=None): 
        return tt.triu(tt.full((block_size, block_size), -tt.inf, dtype=dtype, device=device), diagonal=diagonal)

    @staticmethod
    def CausalCrossMask(d_block_size, e_block_size, diagonal=1, dtype=None, device=None): 
        return tt.triu(tt.full((d_block_size, e_block_size), -tt.inf, dtype=dtype, device=device), diagonal=diagonal)

    # "Secrete Functions"
    @staticmethod
    @tt.no_grad()
    def reset_parameters_xavier_uniform(module):
        for p in module.parameters():
            if p.dim() > 1: xavier_uniform_(p)

class Aview:
    dca = dict(width=4, values=False, ticks=True, verbose=0,  cmap='hot',      vmin=0, vmax=1)
    dsa = dict(width=4, values=False, ticks=True, verbose=0,  cmap='binary',   vmin=0, vmax=1)
    esa = dict(width=4, values=False, ticks=True, verbose=0,  cmap='copper',   vmin=0, vmax=1)


    @tt.no_grad()
    def view_attention(coder, cross,  batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        
        #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
        fig, axs = plt.subplots(coder.num_layers, 1, figsize=(width, coder.num_layers*width))
        if title: fig.suptitle(f'{title}')
        single = coder.num_layers==1


        for l in range(coder.num_layers):
            # batch_size, n_heads, SEQ, SEQ
            if batch_index is None:
                if head_index is None:
                    if cross:
                        w = coder.layers[l].attention_weights_cross.detach().cpu().mean(dim=0)
                    else:
                        w = coder.layers[l].attention_weights.detach().cpu().mean(dim=0)
                else:
                    if cross:
                        w = coder.layers[l].attention_weights_cross[head_index].detach().cpu()
                    else:
                        w = coder.layers[l].attention_weights[head_index].detach().cpu()
            else:
                if head_index is None:
                    if cross:
                        w = coder.layers[l].attention_weights_cross[batch_index, :].detach().cpu().mean(dim=0)
                    else:
                        w = coder.layers[l].attention_weights[batch_index, :].detach().cpu().mean(dim=0)
                else:
                    if cross:
                        w = coder.layers[l].attention_weights_cross[batch_index, head_index].detach().cpu()
                    else:
                        w = coder.layers[l].attention_weights[batch_index, head_index].detach().cpu()

            ax = axs if single else axs[l]
            ax.matshow(w.numpy(), **matshow)
            ax.set_xlabel(f'layer: {l+1}')
            if ticks:
                ax.set_xticks(range(w.shape[1]))
                ax.set_yticks(range(w.shape[0]))
            if values:
                for i in range(w.shape[0]):
                    for j in range(w.shape[1]):
                        ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
            if verbose: print(f'Layer: {l} :: \n{w}')

        if save:
            fig.savefig(save)
            plt.close()
        else:
            plt.show()

    def view_attention_encoder(former, batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        __class__.view_attention(former.encoder, False, batch_index=batch_index, head_index=head_index, width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    

    def view_attention_decoder(former, cross, batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        __class__.view_attention(former.decoder, cross, batch_index=batch_index, head_index=head_index, width=width, 
                values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow) 

    def view_attention_encoders(former, batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        for encoder in former.encoders: __class__.view_attention(encoder, False, batch_index=batch_index, head_index=head_index, width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    


    @tt.no_grad()
    def view_mh_attention(coder, cross,  batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):

        for l in range(coder.num_layers):
            num_heads = coder.layers[l].num_heads
            fig, axs = plt.subplots(1,  num_heads, figsize=(width*num_heads, width))
            if title:  fig.suptitle(f'{title}')
            single_head = num_heads==1
            # batch_size, n_heads, SEQ, SEQ
            for h in range(num_heads):
                if batch_index is None:
                    if cross:
                        w = coder.layers[l].attention_weights_cross[h].detach().cpu()
                    else:
                        w = coder.layers[l].attention_weights[h].detach().cpu()
                else:
                    if cross:
                        w = coder.layers[l].attention_weights_cross[batch_index, h].detach().cpu()
                    else:
                        w = coder.layers[l].attention_weights[batch_index, h].detach().cpu()

                if single_head: ax = axs
                else: ax = axs[h]

                
                ax.matshow(w.numpy(), **matshow)
                ax.set_xlabel(f'layer: {l+1}, head: {h+1}')
                if ticks:
                    ax.set_xticks(range(w.shape[1]))
                    ax.set_yticks(range(w.shape[0]))
                if values:
                    for i in range(w.shape[0]):
                        for j in range(w.shape[1]):
                            ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
                                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
                if verbose: print(f'layer: {l+1}, head: {h+1} :: \n{w}')

            if save: fig.savefig(f'{save}_{l}.png')
            else: plt.show()
            plt.close()


    def view_mh_attention_encoder(former, batch_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        __class__.view_mh_attention(former.encoder, False, batch_index=batch_index,  width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    

    def view_mh_attention_decoder(former, cross, batch_index=None,  width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        __class__.view_mh_attention(former.decoder, cross, batch_index=batch_index,  width=width, 
                values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow) 

    def view_mh_attention_encoders(former, batch_index=None,  width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        for encoder in former.encoders: __class__.view_mh_attention(encoder, False, batch_index=batch_index, width=width, 
           values=values, ticks=ticks, verbose=verbose, save=save, title=title, **matshow)    


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

