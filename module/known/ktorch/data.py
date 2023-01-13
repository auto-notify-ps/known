# -----------------------------------------------------------------------------------------------------
import torch as tt
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# -----------------------------------------------------------------------------------------------------


class SeqDataset(Dataset):

    def __init__(self, signal, seqlen, squeeze_label=False) -> None:
        assert(signal.ndim>1), f'Must have at least two dimension'
        assert(signal.shape[0]>0), f'Must have at least one sample'
        self.signal = signal
        self.seqlen = seqlen
        self.features = signal.shape[1]
        self.count = len(self.signal)
        self.length = self.count - self.seqlen
        self.squeeze_label=squeeze_label
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        x = self.signal[index : index+self.seqlen]
        y = self.signal[index+self.seqlen : index+self.seqlen+1]
        if self.squeeze_label: y = y.squeeze(0)
        return x, y

    def dataloader(self, batch_size=1, shuffle=None):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


    def read_csv(csv, cols, reverse, normalize=False):
        df = pd.read_csv(csv) 
        C = []

        for c in cols:
            signal = df[c].to_numpy()
            if reverse: signal = signal[::-1]
            if normalize:
                scalar = np.maximum(np.abs(np.max(signal)), np.abs(np.min(signal)))
                if scalar!=0: signal=signal/scalar
            C.append(signal)
        return np.transpose(np.stack(C))

    def from_csv(csv, cols, seqlen, reverse, normalize=False, squeeze_label=False, dtype=None):
        return __class__(tt.from_numpy(
            np.copy(  __class__.read_csv(csv, cols, reverse, normalize=normalize) )).to(dtype=dtype), seqlen, squeeze_label=squeeze_label)

    def to_csv(ds, csv, cols):
        signal = np.transpose(ds.signal.numpy())
        df = pd.DataFrame({col:sig for sig,col in zip(signal, cols) }  )
        df.to_csv(csv)
        return csv

    def split_csv(csv, cols, splits, file_names):
        signal = __class__.read_csv(csv, cols, False)
        siglen = len(signal)
        split_sum=0
        splits_ratio = []
        s=0
        for ratio,file_name in zip(splits,file_names):
            e=int(ratio*siglen)+s
            signal_slice = signal[s:e]
            splits_ratio.append((s,e,e-s))
            split_sum+=(e-s)
            s=e
            df = pd.DataFrame({col: signal_slice[:,j] for j,col in enumerate(cols)})
            if (file_name is not None):
                if len(file_name)>0: df.to_csv(file_name)
        return siglen, splits, splits_ratio, split_sum

    def generate(
        genF, # a function like lambda rng, genX, dimS: y
        colS,
        normalize=False,
        seed=None,
        file_name=None, verbose=0): 
        rng = np.random.default_rng(seed)
        generated_priceS = []
        for dimS in range(len(colS)): generated_priceS.append(genF(rng, dimS))
        dd = {}
        for col,generated_price in zip(colS,generated_priceS):
            if normalize:
                scalar = np.maximum(np.abs(np.max(generated_price)), np.abs(np.min(generated_price)))
                if scalar!=0: generated_price/=scalar
            dd[col] = generated_price
        
        df = pd.DataFrame(dd)
        if (file_name is not None):
            if len(file_name)>0: df.to_csv(file_name)
        if verbose>0:
            print(df.info())
            if verbose>1:
                print(df.describe())
                if verbose>2:
                    print(df)
        return df 

    def auto_split_csv(file_name, colS, splits, split_names=None):
        if not split_names : split_names =   [ str(i+1) for i in range(len(splits)) ] 
        #[ 'train', 'val', 'test'] #[ 'train', 'test']
        sepi = file_name.rfind('.')
        fn, fe = file_name[:sepi], file_name[sepi:]
        file_names = [ f'{fn}_{sn}{fe}' for sn in split_names  ]
        return __class__.split_csv(csv=file_name, cols=colS, splits=splits, file_names=file_names)

    def generateS(
            genF, # a function like lambda rng, genX, dimS: y
            colS,
            normalize=False,
            seed=None,
            file_name=None, splits=None, split_names=None, verbose=0): 
        df = __class__.generate(genF, colS, normalize, seed, file_name, verbose)
        if file_name is not None and splits is not None:
            return __class__.auto_split_csv(file_name, colS, splits, split_names)
        else:
            return df


