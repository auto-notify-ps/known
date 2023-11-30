


__all__ = ['Optimizers']

class Optimizers:

    @staticmethod
    def GPTAdamW(model, device, weight_decay=1e-2, learning_rate=1e-3,  betas=(0.9, 0.999), eps=1e-8, verbose=False): 
        import inspect
        from torch.optim import AdamW
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(AdamW).parameters
        use_fused = fused_available and device == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, **extra_args)
        if verbose: print(f"using fused AdamW: {use_fused}")

        return optimizer
