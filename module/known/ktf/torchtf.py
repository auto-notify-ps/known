
from typing import Optional, Any, Union, Callable
import torch as tt
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

#from .modular import duplicate as _get_clones
__all__ = [
    'TransformerEncoder',       'TransformerDecoder',       'TransformerCross',
    'TransformerEncoderLayer',  'TransformerDecoderLayer',  'TransformerCrossLayer',
    'Encoder', 'Decoder',
    'CODER_DEFAULT_ARGS',
    ]


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

CODER_DEFAULT_ARGS = dict(
    d_model=            512, 
    nhead=              8, 
    dim_feedforward =   2048, 
    dropout =           0.1,
    activation=         F.gelu,
    normF =             nn.LayerNorm,
    normA =             dict(eps=1e-6),
    norm_first =        False,
    attention2F =       (nn.MultiheadAttention, dict(batch_first=True)),
)


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, normF, normA, norm_first, attention2F, device=None, dtype=None) -> None:
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads=nhead

        self.self_attn = attention2F[0](d_model, self.num_heads, dropout=dropout, **attention2F[1], **self.factory)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **self.factory)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **self.factory)

        self.norm_first = norm_first
        self.norm1 = normF(d_model, **normA, **self.factory)
        self.norm2 = normF(d_model, **normA, **self.factory)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, tt.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, tt.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False, store_attention=False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif tt.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif store_attention:
            why_not_sparsity_fast_path="Need to store attention weights."
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if tt.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif tt.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return tt._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal, store_attention=store_attention)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal, store_attention=store_attention))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False, store_attention=False) -> Tensor:
        x, w = self.self_attn.forward(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=store_attention, average_attn_weights=False, )
        if store_attention: self.attention_weights = w
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']
    
    def __init__(self, coderA, num_layers, norm=None, enable_nested_tensor=True, mask_check=True, dtype=None, device=None):
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.coderA=coderA
        self.layers = nn.ModuleList(  [  TransformerEncoderLayer(**coderA, **self.factory)  for _ in range(self.num_layers) ] ) #_get_clones(encoder_layer, num_layers)
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
        

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None, store_attention=False) -> Tensor:

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        if not isinstance(first_layer, TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{str_first_layer} was not TransformerEncoderLayer"
        elif first_layer.norm_first :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm_first was True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not first_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = f" {str_first_layer}.self_attn.batch_first was not True"
        elif not first_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{str_first_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not first_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f" {str_first_layer}.activation_relu_or_gelu was not True"
        elif not (first_layer.norm1.eps == first_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{str_first_layer}.norm1.eps was not equal to {str_first_layer}.norm2.eps"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif not self.enable_nested_tensor:
            why_not_sparsity_fast_path = "enable_nested_tensor was not True"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not tt._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif first_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif tt.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        elif store_attention:
            why_not_sparsity_fast_path="Need to store attention weights."

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )

            if tt.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not (src.is_cuda or 'cpu' in str(src.device)):
                why_not_sparsity_fast_path = "src is neither CUDA nor CPU"
            elif tt.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = tt._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        # Prevent type refinement
        make_causal = (is_causal is True)

        if is_causal is None:
            if mask is not None:
                sz = mask.size(0)
                causal_comparison = tt.triu(
                    tt.ones(sz, sz, device=mask.device) * float('-inf'), diagonal=1
                ).to(mask.dtype)

                if tt.equal(mask, causal_comparison):
                    make_causal = True

        is_causal = make_causal

        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers, store_attention=store_attention)

        if convert_to_nested:
            output = output.to_padded_tensor(0.)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, normF, normA, norm_first, attention2F, device=None, dtype=None) -> None:
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_heads=nhead

        self.self_attn = attention2F[0](d_model, self.num_heads, dropout=dropout, **attention2F[1], **self.factory)
        self.multihead_attn = attention2F[0](d_model, self.num_heads, dropout=dropout, **attention2F[1], **self.factory)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **self.factory)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **self.factory)

        self.norm_first = norm_first
        self.norm1 = normF(d_model, **normA, **self.factory)
        self.norm2 = normF(d_model, **normA, **self.factory)
        self.norm3 = normF(d_model, **normA, **self.factory)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        store_attention=False,
    ) -> Tensor:

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,  store_attention=store_attention)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal,  store_attention=store_attention)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,  store_attention=store_attention))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal,  store_attention=store_attention))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,  store_attention=False) -> Tensor:
        x, w = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=store_attention, average_attn_weights=False,)
        if store_attention: self.attention_weights = w
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,  store_attention=False) -> Tensor:
        x, w = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=store_attention, average_attn_weights=False,)
        if store_attention: self.attention_weights_cross = w
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerDecoder(nn.Module):

    __constants__ = ['norm']

    def __init__(self, coderA, num_layers, norm=None, dtype=None, device=None):
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.coderA=coderA
        self.layers = nn.ModuleList(  [  TransformerDecoderLayer(**coderA, **self.factory)  for _ in range(self.num_layers) ] )
        self.norm = norm
        

    def forward(self, tgt: Tensor, memory: Tensor, multi_memory: bool=False, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, store_attention=False) -> Tensor:

        output = tgt

        if multi_memory:
            # expects list of  memory, memory_mask, memory_key_padding_mask
            for mod, mem, mem_mask, mem_key_pad_mask in zip(self.layers, memory, memory_mask, memory_key_padding_mask):
                output = mod(output, mem, tgt_mask=tgt_mask,
                            memory_mask=mem_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=mem_key_pad_mask,
                            store_attention=store_attention)
        else:
            for mod in self.layers:
                output = mod(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            store_attention=store_attention)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerCrossLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, normF, normA, norm_first, attention2F, device=None, dtype=None) -> None:
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.multihead_attn = attention2F[0](d_model, self.num_heads, dropout=dropout, **attention2F[1], **self.factory)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **self.factory)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **self.factory)

        self.norm_first = norm_first
        #self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = normF(d_model, **normA, **self.factory)
        self.norm3 = normF(d_model, **normA, **self.factory)
        #self.dropout1 = Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        #tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        #tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        #tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
        store_attention=False,
    ) -> Tensor:

        x = tgt
        if self.norm_first:
            #x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,  store_attention=store_attention)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal,  store_attention=store_attention)
            x = x + self._ff_block(self.norm3(x))
        else:
            #x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,  store_attention=store_attention))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal,  store_attention=store_attention))
            x = self.norm3(x + self._ff_block(x))

        return x


    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False,  store_attention=False) -> Tensor:
        x, w = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=store_attention, average_attn_weights=False,)
        if store_attention: self.attention_weights_cross = w
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerCross(nn.Module):

    __constants__ = ['norm']

    def __init__(self, coderA, num_layers, norm=None, dtype=None, device=None):
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_layers = num_layers
        self.coderA = coderA
        self.layers = nn.ModuleList(  [  TransformerCrossLayer(**coderA, **self.factory)  for _ in range(self.num_layers) ] )
        self.norm = norm


    def forward(self, tgt: Tensor, memory: Tensor, multi_memory: bool=False, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, store_attention=False) -> Tensor:

        output = tgt

        if multi_memory:
            for mod,mem,mem_mask,mem_key_pad_mask in zip(self.layers, memory, memory_mask, memory_key_padding_mask):
                output = mod(output, mem, tgt_mask=tgt_mask,
                            memory_mask=mem_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=mem_key_pad_mask,
                            store_attention=store_attention)
        else:
            for mod in self.layers:
                output = mod(output, memory, tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            store_attention=store_attention)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Encoder(TransformerEncoder):

    def __init__(self, vocab_size, pose,
                 coderA, num_layers, norm=None, 
                 #enable_nested_tensor=True, 
                 #mask_check=True, 
                 dtype=None, device=None):
        super().__init__(coderA, num_layers, norm, True, True, dtype, device)
        #self.coderA['vocab_size'] = vocab_size
        self.embeder = nn.Embedding(vocab_size, self.coderA['d_model'], **self.factory)
        self.pose=pose
        self.do_store_attention(False)


    def do_store_attention(self, do_store): self.store_attention=do_store

    def forward(self, 
                src: Tensor, 
                mask: Tensor | None = None, 
                src_key_padding_mask: Tensor | None = None, 
                #is_causal: bool | None = None, 
                #store_attention=False
                ) -> Tensor:
        srcE = self.pose(self.embeder(src))
        return super().forward(srcE, mask, src_key_padding_mask, None, self.store_attention)


    # @tt.no_grad()
    # def view_attention(self, batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        
    #     #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
    #     fig, axs = plt.subplots(self.num_layers, 1, figsize=(width, self.num_layers*width))
    #     if not title: title=f'{__class__}'
    #     fig.suptitle(f'{title}')
    #     single = self.num_layers==1


    #     for l in range(self.num_layers):
    #         # batch_size, n_heads, SEQ, SEQ
    #         if batch_index is None:
    #             if head_index is None:
    #                 w = self.layers[l].attention_weights.detach().cpu().mean(dim=0)
    #             else:
    #                 w = self.layers[l].attention_weights[head_index].detach().cpu()
    #         else:
    #             if head_index is None:
    #                 w = self.layers[l].attention_weights[batch_index, :].detach().cpu().mean(dim=0)
    #             else:
    #                 w = self.layers[l].attention_weights[batch_index, head_index].detach().cpu()

    #         ax = axs if single else axs[l]
    #         ax.matshow(w.numpy(), **matshow)
    #         ax.set_xlabel(f'layer: {l+1}')
    #         if ticks:
    #             ax.set_xticks(range(w.shape[1]))
    #             ax.set_yticks(range(w.shape[0]))
    #         if values:
    #             for i in range(w.shape[0]):
    #                 for j in range(w.shape[1]):
    #                     ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
    #                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #         if verbose: print(f'Layer: {l} :: \n{w}')

    #     if save:
    #         fig.savefig(save)
    #         plt.close()
    #     else:
    #         plt.show()

class Decoder(TransformerDecoder):
    def __init__(self, vocab_size, pose,
                 coderA, num_layers, norm=None, 
                 dtype=None, device=None):
        super().__init__(coderA, num_layers, norm, dtype, device)
        #self.coderA['vocab_size'] = vocab_size
        self.embeder = nn.Embedding(vocab_size, self.coderA['d_model'], **self.factory)
        self.pose=pose
        self.do_store_attention(False)
    
    def do_store_attention(self, do_store): self.store_attention=do_store

    def forward(self, 
                tgt: Tensor, 
                memory: Tensor, 
                multi_memory: bool = False, 
                tgt_mask: Tensor | None = None, 
                memory_mask: Tensor | None = None, 
                tgt_key_padding_mask: Tensor | None = None, 
                memory_key_padding_mask: Tensor | None = None, 
                #store_attention=False
                ) -> Tensor:
        tgtE = self.pose(self.embeder(tgt))
        return super().forward(tgtE, memory, multi_memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, self.store_attention)
    


    # @tt.no_grad()
    # def view_attention(self, cross,  batch_index=None, head_index=None, width=16, values=False, ticks=False, verbose=0, save='', title='', **matshow):
        
    #     #plt.matshow(cdec.decoder.layers[0].attention_weights_cross[0,:,:])
    #     fig, axs = plt.subplots(self.num_layers, 1, figsize=(width, self.num_layers*width))
    #     if not title: title=f'{__class__}'
    #     fig.suptitle(f'{title}')
    #     single = self.num_layers==1


    #     for l in range(self.num_layers):
    #         # batch_size, n_heads, SEQ, SEQ
    #         if batch_index is None:
    #             if head_index is None:
    #                 if cross:
    #                     w = self.layers[l].attention_weights_cross.detach().cpu().mean(dim=0)
    #                 else:
    #                     w = self.layers[l].attention_weights.detach().cpu().mean(dim=0)
    #             else:
    #                 if cross:
    #                     w = self.layers[l].attention_weights_cross[head_index].detach().cpu()
    #                 else:
    #                     w = self.layers[l].attention_weights[head_index].detach().cpu()
    #         else:
    #             if head_index is None:
    #                 if cross:
    #                     w = self.layers[l].attention_weights_cross[batch_index, :].detach().cpu().mean(dim=0)
    #                 else:
    #                     w = self.layers[l].attention_weights[batch_index, :].detach().cpu().mean(dim=0)
    #             else:
    #                 if cross:
    #                     w = self.layers[l].attention_weights_cross[batch_index, head_index].detach().cpu()
    #                 else:
    #                     w = self.layers[l].attention_weights[batch_index, head_index].detach().cpu()

    #         ax = axs if single else axs[l]
    #         ax.matshow(w.numpy(), **matshow)
    #         ax.set_xlabel(f'layer: {l+1}')
    #         if ticks:
    #             ax.set_xticks(range(w.shape[1]))
    #             ax.set_yticks(range(w.shape[0]))
    #         if values:
    #             for i in range(w.shape[0]):
    #                 for j in range(w.shape[1]):
    #                     ax.text(j, i, '{:0.2f}'.format(w[i,j].item()), ha='center', va='center',
    #                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    #         if verbose: print(f'Layer: {l} :: \n{w}')

    #     if save:
    #         fig.savefig(save)
    #         plt.close()
    #     else:
    #         plt.show()
