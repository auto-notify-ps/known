
import torch as tt
import torch.nn as nn
from torch import Tensor

from .basic import Mod
from .torchtf import Encoder, Decoder

__all__ = [ 
    'Seq2SeqTransformer', 
    'MultiSeq2SeqTransformer', 
    'SeqEncodingTransformer', 
    'MultiSeqEncodingTransformer'
]


class Seq2SeqTransformer(nn.Module):  
        
    def do_store_attention(self, do_store, encoder=True, decoder=True):
        if decoder: self.decoder.do_store_attention(do_store)
        if encoder: self.encoder.do_store_attention(do_store)    

    def __init__(self,custom_encoder:Encoder, custom_decoder:Decoder,
                 n_outputs:int,
                 dense_layer_dims:list, 
                 dense_actFs:list, 
                 dense_bias=True, 
                 xavier_init=False,
                 device=None, dtype=None) -> None:
        assert custom_encoder.coderA['d_model']==custom_decoder.coderA['d_model'], f'Expecting same embedding dimension!'
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoder = custom_encoder
        self.decoder = custom_decoder
        self.embed_size = custom_decoder.coderA['d_model']
        self.n_outputs = n_outputs
        self.dense = Mod.Dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_outputs,
            actFs=dense_actFs, bias=dense_bias, **self.factory)
        if xavier_init: Mod.reset_parameters_xavier_uniform(self)

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,
            ) -> Tensor:

        memory = self.encoder.forward(
            src=src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask)
        
        output = self.decoder.forward(
            tgt=tgt, 
            memory=memory,
            multi_memory=False,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return self.dense(output)

class MultiSeq2SeqTransformer(nn.Module): 
        
    def do_store_attention(self, do_store, encoder=True, decoder=True):
        if decoder: self.decoder.do_store_attention(do_store)
        if encoder: 
            for encoder in self.encoders: encoder.do_store_attention(do_store)    

    def __init__(self,custom_encoders:list[Encoder], custom_decoder:Decoder,
                 catenate:bool,
                 n_outputs:int,
                 dense_layer_dims:list, 
                 dense_actFs:list, 
                 dense_bias=True, 
                 edl_mapping=None,
                 xavier_init=False,
                 device=None, dtype=None) -> None:
        self.catenate = catenate
        if catenate:
            lenc = len(custom_encoders)
            for custom_encoder in custom_encoders:
                assert custom_encoder.coderA['d_model']*lenc==custom_decoder.coderA['d_model'], f'Expecting same embedding dimension!'
        else:
            for custom_encoder in custom_encoders:
                assert custom_encoder.coderA['d_model']==custom_decoder.coderA['d_model'], f'Expecting same embedding dimension!'
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoders = nn.ModuleList(custom_encoders)
        self.n_encoders=len(self.encoders)
        self.decoder = custom_decoder
        self.embed_size = custom_decoder.coderA['d_model']
        self.n_outputs = n_outputs
        self.set_encoder_decoder_layer_mapping(edl_mapping)
        self.dense = Mod.Dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_outputs,
            actFs=dense_actFs, bias=dense_bias, **self.factory)
        self.default_null_mask=[None for _ in range(self.n_encoders)]
        self.default_mem_mask=[None for _ in range(self.decoder.num_layers)]
        
        if xavier_init: Mod.reset_parameters_xavier_uniform(self)


    def set_encoder_decoder_layer_mapping(self, mapping):
        if mapping is None:  self.encoder_decoder_layer_mapping = None
        else:
            assert len(mapping) == self.decoder.num_layers, f'Require one mapping index for each decoder layer : {self.decoder.num_layers}'
            for i,m in enumerate(mapping): assert isinstance(m, int) and m>=0 and m<=self.n_encoders, f'Invalid mapping {m=} and index {i=}'
            self.encoder_decoder_layer_mapping = mapping

    def forward(self, src: Tensor, tgt: Tensor, src_mask = None, tgt_mask  = None,
                src_key_padding_mask = None,
                tgt_key_padding_mask = None, 
                memory_mask=None,
                memory_key_padding_mask=None,
            ) -> Tensor:
        if src_mask is None: src_mask=self.default_null_mask
        if src_key_padding_mask is None: src_key_padding_mask=self.default_null_mask
        if self.catenate:
            memory = tt.cat([encoder.forward(
                        src=isrc,
                        mask=isrc_mask, 
                        src_key_padding_mask=isrc_key_padding_mask) for encoder,isrc,isrc_mask,isrc_key_padding_mask in \
                            zip(self.encoders, src, src_mask, src_key_padding_mask)], dim=-1)
        else:
            memory_ = [encoder.forward(
                        src=isrc,
                        mask=isrc_mask, 
                        src_key_padding_mask=isrc_key_padding_mask) for encoder,isrc,isrc_mask,isrc_key_padding_mask in \
                            zip(self.encoders, src, src_mask,src_key_padding_mask)]
            
            memory =  [ memory_[encoder_index] for encoder_index in self.encoder_decoder_layer_mapping ]  if self.encoder_decoder_layer_mapping else memory_
    
        # NOTE: for catenate=True, the block_size of all encoder must be the same, 
        # but this not the case of catenate=False where encoders can have different sequence lengths
        output = self.decoder.forward(
            tgt=tgt, 
            memory=memory,
            multi_memory=not self.catenate,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask if self.catenate else (self.default_mem_mask if memory_mask is None else memory_mask),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask if self.catenate else (self.default_mem_mask if memory_key_padding_mask is None else memory_key_padding_mask),)
        return self.dense(output)

class SeqEncodingTransformer(nn.Module): 

    def do_store_attention(self, do_store): self.encoder.do_store_attention(do_store)    

    def __init__(self,custom_encoder:Encoder,
                 n_outputs:int,
                 dense_layer_dims:list, 
                 dense_actFs:list, 
                 dense_bias=True, 
                 xavier_init=False,
                 device=None, dtype=None) -> None:
        #assert custom_encoder.coderA['d_model']==custom_decoder.coderA['d_model'], f'Expecting same embedding dimension!'
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoder = custom_encoder
        self.embed_size = custom_encoder.coderA['d_model']
        self.n_outputs = n_outputs
        self.dense = Mod.Dense(
            in_dim=self.embed_size, layer_dims=dense_layer_dims, out_dim=self.n_outputs,
            actFs=dense_actFs, bias=dense_bias, **self.factory)
        if xavier_init: Mod.reset_parameters_xavier_uniform(self)

    def forward(self, src: Tensor, 
                src_mask = None, 
                src_key_padding_mask = None,
            ) -> Tensor:
        return self.dense(self.encoder.forward(
            src=src,
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask))

class MultiSeqEncodingTransformer(nn.Module):

    def do_store_attention(self, do_store):
        for encoder in self.encoders: encoder.do_store_attention(do_store)    

    def __init__(self,custom_encoders:list[Encoder],
                 catenate:bool,
                 n_outputs:int,
                 dense_layer_dims:list, 
                 dense_actFs:list, 
                 dense_bias=True, 
                 xavier_init=False,
                 device=None, dtype=None) -> None:
        self.catenate = catenate
        for custom_encoder in custom_encoders:
            assert custom_encoder.coderA['d_model']==custom_encoders[0].coderA['d_model'], f'Expecting same embedding dimension!'
        self.factory = {'device': device, 'dtype': dtype}
        super().__init__()
        self.encoders = nn.ModuleList(custom_encoders)
        self.n_encoders=len(self.encoders)
        self.embed_size = custom_encoders[0].coderA['d_model']
        self.n_outputs = n_outputs
        self.dense = Mod.Dense(
            in_dim=self.embed_size*self.n_encoders, layer_dims=dense_layer_dims, out_dim=self.n_outputs,
            actFs=dense_actFs, bias=dense_bias, **self.factory)
        if xavier_init: Mod.reset_parameters_xavier_uniform(self)


    def forward(self, 
                src: Tensor, 
                src_mask = None, 
                src_key_padding_mask = None,
            ) -> Tensor:
        if src_mask is None: src_mask=self.default_null_mask
        if src_key_padding_mask is None: src_key_padding_mask=self.default_null_mask
        output = tt.cat([encoder.forward(
                    src=isrc,
                    mask=isrc_mask, 
                    src_key_padding_mask=isrc_key_padding_mask) for encoder,isrc,isrc_mask,isrc_key_padding_mask in zip(self.encoders, src, src_mask,src_key_padding_mask)], dim=-1)

        return self.dense(output)

