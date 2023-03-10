#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/ktorch/data.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'SeqDataset', 'Vocab', 'LangDataset',
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import torch as tt
from torch import Tensor
#import torch.nn as nn
import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pandas import DataFrame
import os
#import math
from typing import Any, Union, Iterable, Callable, Dict, Tuple, List
from ..basic import ndigs, int2base
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


class SeqDataset(Dataset):
    r"""
    Sequential Dataset - for multi-dimensional time serise data.
    Wraps a multi-dimensional signal and provides sub-sequences of fixed length as data samples.
    The signal is stored in a multi-dimensional torch tensor.

    :param signal: multi-dimensional sequential data, a tensor of at least two dimensions where the 
        first dimension is the time dimension and other dimensions represent the features.
    :param seqlen: sequence of this length are extracted from the signal.
    :param squeeze_label: if True, squeeze the labels at dimension 0.

    .. note:: 
        * Underlying signal is stored as ``torch.Tensor``
        * ``pandas.DataFrame`` object is used for disk IO
        * Call :func:`~known.ktorch.data.SeqDataset.dataloader` to get a torch Dataloader instance
        * We may want to squeeze the labels when connecting a dense layer towards the end of a model. Squeezing labels sets appropiate shape of `predicted` argument for a loss functions.
    """

    def __init__(self, signal:Tensor, seqlen:int, squeeze_label:bool=False) -> None:
        r""" 
        :param signal: multi-dimensional sequential data, a tensor of at least two dimensions where the 
            first dimension is the time dimension and other dimensions represent the features.
        :param seqlen: sequence of this length are extracted from the signal.
        :param squeeze_label: if `True`, squeeze the labels at dimension 0.
        """
        super().__init__()
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

    def dataloader(self, batch_size:int=1, shuffle:bool=None) -> DataLoader:
        r""" Returns a Dataloader object from this Dataset """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def read_csv(csv:str, cols:Iterable[str], reverse:bool, normalize:bool=False) -> ndarray:
        r""" 
        Reads data from a csv file using ``pandas.read_csv`` call. 
        The csv file is assumed to have time serise data in each of its columns i.e. 
        the columns are the features of the signal where as each row is a sample.

        :param csv:         path of csv file to read from
        :param cols:        name of columns to take features from
        :param reverse:     if `True`, reverses the sequence
        :param normalize:   if `True`, normalizes the data using min-max scalar

        :returns: time serise signal as ``ndarray``
        """
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

    @staticmethod
    def from_csv(csv:str, cols:Iterable[str], seqlen:int, reverse:bool, normalize:bool=False, 
                squeeze_label:bool=False, dtype=None) -> Dataset:
        r"""
        Creates a dataset instance from a csv file using :func:`~known.ktorch.data.SeqDataset.read_csv`.

        :param csv:         path of csv file to read from
        :param cols:        name of columns to take features from
        :param seqlen:      sequence of this length are extracted from the signal.
        :param reverse:     if `True`, reverses the sequence
        :param normalize:   if `True`, normalizes the data using min-max scalar
        :param squeeze_label: if `True`, squeeze the labels at dimension 0.
        :param dtype:       ``torch.dtype``

        :returns: an instance of this class

        .. note:: 
            This functions calls :func:`~known.ktorch.data.SeqDataset.read_csv`

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.to_csv`
        """
        return __class__(tt.from_numpy(
                np.copy(  __class__.read_csv(
                csv, cols, reverse, normalize=normalize) )).to(dtype=dtype), seqlen, squeeze_label=squeeze_label)

    @staticmethod
    def to_csv(ds, csv:str, cols:Iterable[str]):
        r"""
        Writes a dataset to a csv file using the ``pandas.DataFrame.to_csv`` call.
        
        :param csv:     path of csv file to write to
        :param cols:    column names to be used for each feature dimension

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.from_csv`
        """
        signal = np.transpose(ds.signal.numpy())
        df = pd.DataFrame({col:sig for sig,col in zip(signal, cols) }  )
        df.to_csv(csv)
        return

    @staticmethod
    def split_csv(csv:str, cols:Iterable[str], splits:Iterable[float], file_names:Iterable[str])-> Tuple:
        r"""
        Splits a csv according to provided ratios and saves each split to a seperate file.
        This can be used to split an existing csv dataset into train, test and validation sets.

        :param csv:         path of csv file to split
        :param cols:        name of columns to take features from
        :param splits:      ratios b/w 0 to 1 indicating size of splits
        :param file_names:  file names (full paths) used to save the splits

        :returns:
            * `siglen`: length of signal that was read from input csv
            * `splits`: same as provided in the argument
            * `splits_indices`: size of each split, indices of the signal that were the splitting points (start, end)
            * `split_sum`: sum of all splits, should be less than or equal to signal length

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.auto_split_csv`
        """
        assert len(splits)==len(file_names), f'splits and file_names must be of equal size'
        signal = __class__.read_csv(csv, cols, False)
        siglen = len(signal)
        split_sum=0
        splits_indices = []
        s=0
        for ratio,file_name in zip(splits,file_names):
            e=int(ratio*siglen)+s
            signal_slice = signal[s:e]
            splits_indices.append((s,e,e-s))
            split_sum+=(e-s)
            s=e
            df = pd.DataFrame({col: signal_slice[:,j] for j,col in enumerate(cols)})
            if (file_name is not None):
                if len(file_name)>0: df.to_csv(file_name)
        return siglen, splits, splits_indices, split_sum

    @staticmethod
    def auto_split_csv(csv:str, cols:Iterable[str], splits:Iterable[float], split_names:Union[Iterable[str], None]=None)-> Tuple:
        r"""
        Wraps ``split_csv``. 
        Produces splits in the same directory and auto generates the file names if not provided.

        :param csv:         path of csv file to split
        :param cols:        name of columns to take features from
        :param splits:      ratios b/w 0 to 1 indicating size of splits
        :param split_names: filename suffix used to save the splits

        .. note:: The splits are saved in the same directory as the csv file but they are suffixed with
            provided ``split_names``. For example, if ``csv='my_file.csv'`` and ``split_names=('train', 'test')``
            then two new files will be created named ``my_file_train.csv`` and ``my_file_test.csv``.
            In case ``split_names=None``, new files will have names ``my_file_1.csv`` and ``my_file_2.csv``.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.split_csv`
        """
        if not split_names : split_names =   [ str(i+1) for i in range(len(splits)) ] 
        sepi = csv.rfind('.')
        fn, fe = csv[:sepi], csv[sepi:]
        file_names = [ f'{fn}_{sn}{fe}' for sn in split_names  ]
        return __class__.split_csv(csv=csv, cols=cols, splits=splits, file_names=file_names)

    @staticmethod
    def generate(csv:str, genF:Callable, colS:Iterable[str], normalize:bool=False, seed:Union[int, None]=None) -> DataFrame: 
        r"""
        Generates a synthetic time serise dataset and saves it to a csv file

        :param csv: path of file to write to using ``pandas.DataFrame.to_csv`` call (should usually end with .csv)
        :param genF: a function like ``lambda rng, dim: y``,
            that generates sequential data at a given dimension ``dim``, where ``rng`` is a numpy RNG
        :param colS: column names for each dimension/feature
        :param normalize:   if True, normalizes the data using min-max scalar
        :param seed: seed for ``np.random.default_rng``
        
        :returns: ``pandas.DataFrame``

        .. note:: use ``DataFrame.info()`` and ``DataFrame.describe()`` to get information on generated data.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.generateS`
        """
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
        if (csv is not None):
            if len(csv)>0: df.to_csv(csv)
        return df 

    @staticmethod
    def generateS(csv:str, genF:Callable, colS:Iterable[str], normalize:bool=False, seed:Union[int, None]=None,
                splits:Iterable[float]=None, split_names:Union[Iterable[str], None]=None) -> Union[Tuple,DataFrame]: 
        r"""
        Generate a synthetic time serise dataset with splits and save to csv files

        :returns: ``pandas.DataFrame`` if no spliting is performed else returns the outputs from ``split_csv``

        .. note:: This is the same as calling :func:`~known.ktorch.data.SeqDataset.generate` and :func:`~known.ktorch.data.SeqDataset.auto_split_csv` one after other. 
            If ``splits`` arg is `None` then no splitting is performed.

        .. seealso:: 
            :func:`~known.ktorch.data.SeqDataset.generate`
        """
        df = __class__.generate(csv, genF, colS, normalize, seed)
        if splits is not None:
            return __class__.auto_split_csv(csv, colS, splits, split_names)
        else:
            return df

class Vocab:
    r"""
    Represents a vocabulary i.e. a collection of all possible `words` in a `language`.

    :param words: an iterable that contains all the words
    :param nbase: optional, the base-n number system to be used when 
        :func:`~known.ktorch.data.LangDataset.embed_base` is used for embedding
    """

    def __init__(self, words:Iterable, nbase:int=2) -> None:
        r"""
        Create new vocabulary

        :param words: an iterable that contains all the words
        :param nbase: optional, the base-n number system to be used when 
            :func:`~known.ktorch.data.LangDataset.embed_base` is used for embedding
        """
        self.words = words
        self.vlen = len(words)
        self.nbase = nbase

        assert self.vlen>=self.nbase, f'Vocal size [{self.vlen}] should be more than nBase [{self.nbase}]'

        self.one_len = self.vlen
        self.base_len = ndigs(self.vlen, self.nbase)

        self.vocad = {k:v for v,k in enumerate(self.words)}

    def embed_one_hot(self, word:str, dtype=None) -> Tensor:
        r""" One-Hot vector encoding, ordinal is set to 1, rest are 0s 
        
        .. seealso:: 
            :func:`~known.ktorch.data.Vocab.embed_one_cold`
        """
        if word in self.vocad:
            embeded = tt.zeros((self.vlen,), dtype=dtype )
            embeded[self.vocad[word]] += 1
            return embeded
        else:
            raise Exception(f'Word [{word}] not found in vocab!')
        
    def embed_one_cold(self, word:str, dtype=None) -> Tensor:
        r""" One-Cold vector encoding, ordinal is set to 0, rest are 1s 

        .. seealso:: 
            :func:`~known.ktorch.data.Vocab.embed_one_hot`
        """
        if word in self.vocad: 
            embeded = tt.ones((self.vlen,), dtype=dtype )
            embeded[self.vocad[word]] -= 1
            return embeded
        else:
            raise Exception(f'Word [{word}] not found in vocab!')
        
    def embed_base(self, word:str, dtype=None) -> Tensor:
        r""" Base-N vector encoding, converts the index of this word to base-n number system """
        if word in self.vocad: 
            return tt.tensor(int2base(self.vocad[word], self.nbase, self.base_len), dtype=dtype)
        else:
            raise Exception(f'Word [{word}] not found in vocab!')

class LangDataset(Dataset):
    r"""
    Language Dataset - for language based models. Abstracts a set of `words` from an underlying :class:`~known.ktorch.data.Vocab`.
    This dataset can be manually created from scratch.

    :param words: an iterable that contains all the words
    :param embed: embedding scheme

        * use ``embed=1`` for one-hot encoding
        * use ``embed=0`` for one-cold encoding
        * use ``embed>1`` for base-n encoding, where n = embed
        
    .. note:: 
        * Use :func:`~known.ktorch.data.LangDataset.add_class` and :func:`~known.ktorch.data.LangDataset.add_samples` to add classes and samples (words) to the dataset.
        * Call :func:`~known.ktorch.data.LangDataset.dataloader` to get a torch Dataloader instance
    """

    def __init__(self, words:Iterable, embed:int=1, dtype=None) -> None:
        r"""
        :param words: an iterable that contains all the words
        :param embed: embedding scheme
            * use ``embed=1`` for one-hot encoding
            * use ``embed=0`` for one-cold encoding
            * use ``embed>1`` for base-n encoding, where n = embed
        """
        super().__init__()
        if embed>1:
            self.vocab = Vocab(words, embed)
            self.embed = self.vocab.embed_base
        else:
            self.vocab = Vocab(words)
            self.embed = self.vocab.embed_one_hot if embed==1 else self.vocab.embed_one_cold
        
        self.classes = {}
        self.n_classes = 0
        self.data = []
        self.data_str = []
        self.data_labels = []
        self.dtype = dtype

    def dataloader(self, batch_size:int=1, shuffle:bool=None) -> DataLoader:
        r""" Returns a Dataloader object from this Dataset """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

    def add_class(self, *labels):
        r""" add new class labels to this dataset """
        for label in labels:
            if label not in self.classes: 
                self.classes[label] = [self.n_classes, 0] # index, no of samples
                self.n_classes+=1
            else:
                print(f'Class label [{label}] already exists!')
    
    def add_samples(self, label, *samples):
        r""" add new samples to a class label, adds the class label if it doesn't exist """
        if label not in self.classes: 
            self.classes[label] = [self.n_classes, 0] 
            self.n_classes+=1
        self.classes[label][1] += len(samples)
        
        self.data_str.extend( [(sample, label) for sample in samples] )
        self.data_labels.extend([self.classes[label][0] for _ in samples])
        self.data.extend( [(tt.stack([self.embed(s, dtype=self.dtype) for s in sample]),(self.classes[label][0]))  for sample in samples] )
        
    def class_one_hot(self, label:str) -> Tensor:
        r""" returns one-hot vector for class label to be used as classification label """
        if label in self.classes: 
            hot_label = tt.zeros((self.n_classes,), dtype=tt.long)
            hot_label[self.classes[label][0]] += 1
            return hot_label
        else:
            print(f'Class label [{label}] does not exist!')
            return None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
        #sample, label = self.data[index] # sample is a sequence
        #return tt.stack([ self.embed(s, dtype=self.dtype) for s in sample ]), self.class_one_hot(label)

    def get_word_from_class(self, label, index=None):
        if label in self.classes: 
            c = self.classes[label][0]
            n = np.where(np.array(self.data_labels)==c)[0]
            if index is None: 
                index = np.random.choice(n)
            else:
                index=n[index]
            return index, self.data[index], self.data_str[index]


    def save(self, path):
        import json
        ds = { label:[None, []] for label in self.classes }
        for sample,label in self.data_str: 
            ds[label][1].append(sample)
            ds[label][0]=self.classes[label][0]
        f=open(path, 'w')
        json.dump(ds, f, indent=4)
        f.close()

    def load(self, path):
        import json
        f=open(path, 'r')
        ds = json.load(f)
        f.close()

        label_order = [ None for _ in ds ]
        for k,v in ds.items(): label_order[v[0]] = k
        for label in label_order: self.add_samples(label, *ds[label][1])

        