# -----------------------------------------------------------------------------------------------------
import torch as tt
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# -----------------------------------------------------------------------------------------------------


class SeqDataset(Dataset):

    def __init__(self, signal, seqlen) -> None:
        self.signal = signal
        self.seqlen = seqlen
        self.length = len(self.signal)
        self.count = self.length - self.seqlen
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        s = self.signal[index : index+self.seqlen+1]
        return  s[:-1].reshape(self.seqlen, 1), s[-1:]

    def dataloader(self, batch_size=1, shuffle=None):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def from_csv(csv, col, reverse, seqlen, dtype=None):
        df = pd.read_csv(csv) 
        signal = df[col].to_numpy()
        if reverse: signal = signal[::-1]   
        return __class__(tt.from_numpy(np.copy(signal)).to(dtype=dtype), seqlen)

    def to_csv(ds, csv, col, reverse):
        signal = ds.signal.numpy()
        if reverse: signal = signal[::-1]   
        df = pd.DataFrame({col:signal})   
        df.to_csv(csv)
        return csv



