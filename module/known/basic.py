#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'now', 'numel', 'arange', 'save_json', 'load_json', 'save_pickle', 'load_pickle',
     
    'Symbols', 'Verbose', 'BaseConvert', 'IndexedDict', 'Remap', 
    #==============================
    
    'onehot', 'onecold', 'dict_sort', 
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import datetime
from typing import Any, Union, Iterable #, BinaryIO, cast, Dict, Optional, Type, Tuple, IO
import numpy as np
from numpy import ndarray
from math import floor, log
import json, pickle
from collections import UserDict
#import pickle, pathlib, io
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def now(year:bool=True, month:bool=True, day:bool=True, 
        hour:bool=True, minute:bool=True, second:bool=True, mirco:bool=True, 
        start:str='', sep:str='', end:str='') -> str:
    r""" Unique Identifier - useful in generating unique identifiers based on current timestamp. 
    Helpful in generating unique filenames based on timestamps. 
    
    .. seealso::
        :func:`~known.basic.Verbose.strU`
    """
    form = []
    if year:    form.append("%Y")
    if month:   form.append("%m")
    if day:     form.append("%d")
    if hour:    form.append("%H")
    if minute:  form.append("%M")
    if second:  form.append("%S")
    if mirco:   form.append("%f")
    assert (form), 'format should not be empty!'
    return (start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end)

def numel(shape:Iterable) -> int: 
    r""" Returns the number of elements in an array of given shape. """
    return np.prod(np.array(shape))

def arange(shape:Iterable, start:int=0, step:int=1, dtype=None) -> ndarray: 
    r""" Similar to ``np.arange`` but reshapes the array to given shape. """
    return np.arange(start=start, stop=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

def save_json(o:Any, path:str, **kwargs) -> None:
    r""" save object to json file """
    with open(path, 'w') as f: json.dump(o, f, **kwargs)

def load_json(path:str) -> Any:
    r""" load json file to object """
    with open(path, 'r') as f: o = json.load(f)
    return o

def save_pickle(o:Any, path:str,**kwargs):
    r""" save object to pickle file """
    with open(path, 'wb') as f: pickle.dump(o, f,**kwargs)

def load_pickle(path:str):
    r""" load pickle file to object """
    with open(path, 'rb') as f: o = pickle.load(f)
    return o

def onehot(total:int, index:int, **kwargs):
    res = np.zeros(total, **kwargs)
    res[index]=1
    return res

def onecold(total:int, index:int, **kwargs):
    res = np.ones(total, **kwargs)
    res[index]=0
    return res

def dict_sort(D:dict, assending:bool=True, fin=lambda x:x, fmid=lambda x:x, fout=lambda x:x, return_dict=True):
    r""" if D is like :: dict[str, numbers], then sorts  the numbers and keys
        
    1. ```fin``` is applied on each number and it is appended to a list
    2. ```fmid``` is applied on the list
    3. create np.array from the output of fmid
    4. ```fout``` is applied on the ndarray
    5. argsort is applied on ndarray
    
    returns keys, values and indices in sorted order
    """
    # sorts the values and keys into lists
    K,V = [], []
    for k,v in D.items():
        K.append(k)
        V.append(fin(v))
    K=np.array(K)
    V=fout(np.array(fmid(V)))
    S = np.argsort(V)
    if assending:
        ks = K[S]
        vs = V[S]
        ss = S 
    else:
        ks = K[S][::-1]
        vs = V[S][::-1]
        ss = S[::-1]
    
    return {k:v for k,v in zip(ks,vs)} if return_dict else (ks,vs,ss)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class BaseConvert:
    r""" Number System Conversion """

    @staticmethod
    def ndigs(num:int, base:int) -> int:
        r""" 
        Returns the number of digits required to represent a base-10 number in the given base.

        :param num:     base-10 number to be represented
        :param base:    base-n number system
        """
        return 1 + (0 if num==0 else floor(log(num, base)))

    @staticmethod
    def int2base(num:int, base:int, digs:int) -> list:
        r""" 
        Convert base-10 integer to a base-n list of fixed no. of digits 

        :param num:     base-10 number to be represented
        :param base:    base-n number system
        :param digs:    no of digits in the output

        :returns:       represented number as a list of ordinals in base-n number system

        .. seealso::
            :func:`~known.basic.base2int`
        """
        if not digs: digs=__class__.ndigs(num, base)
        res = [ 0 for _ in range(digs) ]
        q = num
        for i in range(digs): # <-- do not use enumerate plz
            res[i]=q%base
            q = floor(q/base)
        return res

    @staticmethod
    def base2int(num:Iterable, base:int) -> int:
        """ 
        Convert an iterbale of digits in base-n system to base-10 integer

        :param num:     iterable of base-n digits
        :param base:    base-n number system

        :returns:       represented number as a integer in base-10 number system

        .. seealso::
            :func:`~known.basic.int2base`
        """
        res = 0
        for i,n in enumerate(num): res+=(base**i)*n
        return res

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Symbols:
    r"""
    contains some special symbols described in the table below

    .. list-table:: 
        :widths: 5 3 5 3
        :header-rows: 1

        * - Name
          - Symbol
          - Name
          - Symbol
        * - CORRECT
          - ✓
          - INCORRECT
          - ✗
        * - ALPHA
          - α
          - BETA
          - β
        * - GAMMA
          - γ
          - DELTA
          - δ
        * - EPSILON
          - ε
          - ZETA
          - ζ
        * - ETA
          - η
          - THETA
          - θ
        * - KAPPA
          - κ
          - LAMBDA
          - λ
        * - MU
          - μ 
          - XI
          - ξ
        * - PI
          - π
          - ROH
          - ρ
        * - SIGMA
          - σ
          - PHI
          - φ
        * - PSI
          - Ψ
          - TAU
          - τ
        * - OMEGA
          - Ω
          - TRI
          - Δ
    """
    
    CORRECT =       '✓'
    INCORRECT =     '✗'
    ALPHA =         'α'
    BETA =          'β'
    GAMMA =         'γ'
    DELTA =         'δ'
    EPSILON =       'ε'
    ZETA =          'ζ'
    ETA =           'η'
    THETA =         'θ'
    KAPPA =         'κ'
    LAMBDA =        'λ'
    MU =            'μ' 
    XI =            'ξ'
    PI =            'π'
    ROH =           'ρ'
    SIGMA =         'σ'
    PHI =           'φ'
    PSI =           'Ψ'
    TAU =           'τ'
    OMEGA =         'Ω'
    TRI =           'Δ'

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Verbose:
    r""" Contains shorthand helper functions for printing outputs and representing objects as strings.

    .. note::
        This class contains only static methods.
    """
    DEFAULT_DATE_FORMAT = ["%Y","%m","%d","%H","%M","%S","%f"]
    r""" Default date format for :func:`~known.basic.Verbose.strU` """

    DASHED_LINE = "=-=-=-=-==-=-=-=-="
    DOCSTR_FORM = lambda x: f'\t!docstr:\n! - - - - - - - - - - - - - - - - -\n{x}\n- - - - - - - - - - - - - - - - - !'

    @staticmethod
    def strN(s:str, n:int) -> str:  
        r""" Repeates a string n-times """
        return ''.join([s for _ in range(n)])

    @staticmethod
    def _recP_(a, level, index, pindex, tabchar='\t', show_dim=False):
        # helper function for recP - do not use directly
        if index<0: index=''
        dimstr = ('* ' if level<1 else f'*{level-1} ') if show_dim else ''
        pindex = f'{pindex}{index}'
        if len(a.shape)==0:
            print(f'{__class__.strN(tabchar, level)}[ {dimstr}@{pindex}\t {a} ]') 
        else:
            print(f'{__class__.strN(tabchar, level)}[ {dimstr}@{pindex} #{a.shape[0]}')
            for i,s in enumerate(a):
                __class__._recP_(s, level+1, i, pindex, tabchar, show_dim)
            print(f'{__class__.strN(tabchar, level)}]')

    @staticmethod
    def recP(arr:Iterable, show_dim:bool=False) -> None: 
        r"""
        Recursive Print - print an iterable recursively with added indentation.

        :param arr:         any iterable with ``shape`` property.
        :param show_dim:    if `True`, prints the dimension at the start of each item
        """
        __class__._recP_(arr, 0, -1, '', '\t', show_dim)
    
    @staticmethod
    def strA(arr:Iterable, start:str="", sep:str="|", end:str="") -> str:
        r"""
        String Array - returns a string representation of an iterable for printing.
        
        :param arr:     input iterable
        :param start:   string prefix
        :param sep:     item seperator
        :param end:     string postfix
        """
        res=start
        for a in arr: res += (str(a) + sep)
        return res + end

    @staticmethod
    def strD(arr:Iterable, sep:str="\n", cep:str=":\n", caption:str="") -> str:
        r"""
        String Dict - returns a string representation of a dict object for printing.
        
        :param arr:     input dict
        :param sep:     item seperator
        :param cep:     key-value seperator
        :param caption: heading at the top
        """
        res=f"=-=-=-=-==-=-=-=-={sep}DICT #[{len(arr)}] : {caption}{sep}{__class__.DASHED_LINE}{sep}"
        for k,v in arr.items(): res+=str(k) + cep + str(v) + sep
        return f"{res}{__class__.DASHED_LINE}{sep}"

    @staticmethod
    def strU(form:Union[None, Iterable[str]], start:str='', sep:str='', end:str='') -> str:
        r""" 
        String UID - returns a formated string of current timestamp.

        :param form: the format of timestamp, If `None`, uses the default :data:`~known.basic.Verbose.DEFAULT_DATE_FORMAT`.
            Can be selected from a sub-set of ``["%Y","%m","%d","%H","%M","%S","%f"]``.
            
        :param start: UID prefix
        :param sep: UID seperator
        :param end: UID postfix

        .. seealso::
            :func:`~known.basic.uid`
        """
        if not form: form = __class__.DEFAULT_DATE_FORMAT
        return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end

    @staticmethod
    def show(x:Any, cep:str='\t\t:', sw:str='__', ew:str='__') -> None:
        r"""
        Show Object - describes members of an object using the ``dir`` call.

        :param x:       the object to be described
        :param cep:     the name-value seperator
        :param sw:      argument for ``startswith`` to check in member name
        :param ew:      argument for ``endswith`` to check in member name

        .. note:: ``string.startswith`` and ``string.endswith`` checks are performed on each member of the object 
            and only matching member are displayed. This is usually done to prevent showing dunder members.
        
        .. seealso::
            :func:`~known.basic.Verbose.showX`
        """
        for d in dir(x):
            if not (d.startswith(sw) or d.endswith(ew)):
                v = ""
                try:
                    v = getattr(x, d)
                except:
                    v='?'
                print(d, cep, v)

    @staticmethod
    def showX(x:Any, cep:str='\t\t:') -> None:
        """ Show Object (Xtended) - describes members of an object using the ``dir`` call.

        :param x:       the object to be described
        :param cep:     the name-value seperator

        .. note:: This is the same as :func:`~known.basic.Verbose.show` but skips ``startswith`` and ``endswith`` checks,
            all members are shown including dunder members.

        .. seealso::
            :func:`~known.basic.Verbose.show`
        """
        for d in dir(x):
            v = ""
            try:
                v = getattr(x, d)
            except:
                v='?'
            print(d, cep, v)

    @staticmethod
    def dir(x:Any, doc=False, filter:str='', sew=('__','__')):
        """ Calls ```dir``` on given argument and lists the name and types of non-dunder members.

        :param filter: csv string of types to filter out like `type,function,module`, keep blank for no filter
        :param doc: shows docstring ```__doc``` 
            If ```doc``` is True, show all member's ```__doc__```.
            If ```doc``` is False, does not show any ```__doc__```. 
            If ```doc``` is a string, show ```__doc__``` of specific types only given by csv string.

        :param sew: 2-Tuple (start:str, end:str) - excludes member names that start and end with specific chars, 
            used to exclude dunder methods by default
        """
        #if self_doc: print( f'{type(x)}\n{x.__doc__}\n' )
        if sew: sw, ew = f'{sew[0]}', f'{sew[1]}'
        doc_is_specified = (isinstance(doc, str) and bool(doc))
        if doc_is_specified: doc_match =[ t for t in doc.replace(' ','').split(',') if t ]
        if filter: filter_match =[ t for t in filter.replace(' ','').split(',') if t ]
        counter=1
        for k in dir(x):
            if sew:
                if (k.startswith(sw) and k.endswith(ew)): continue
            m = getattr(x,k)
            n = str(type(m)).split("'")[1]
            if filter:
                if not (n in filter_match):  continue
            s = f'[{counter}] {k} :: {n}'#.encode('utf-16')

            if doc:
                if doc_is_specified:
                    if n in doc_match: 
                        d = __class__.DOCSTR_FORM(m.__doc__)
                    else:
                        d=''
                else:
                    d = __class__.DOCSTR_FORM(m.__doc__)
            else:
                d = ''
            counter+=1
            print(f'{s}{d}')


    @staticmethod
    def info(x:Any, show_object:bool=False):
        r""" Shows the `type`, `length` and `shape` of an object and optionally shows the object as well.

        :param x:           the object to get info about
        :param show_object: if `True`, prints the object itself

        .. note:: This is used to check output of some functions without having to print the full output
            which may take up a lot of console space. Useful when the object are of nested types.

        .. seealso::
            :func:`~known.basic.Verbose.infos`
        """
        print(f'type: {type(x)}')
        if hasattr(x, '__len__'):
            print(f'len: {len(x)}')
        if hasattr(x, 'shape'):
            print(f'shape: {x.shape}')
        if show_object:
            print(f'object:\n{x}')

    @staticmethod
    def infos(X:Iterable, show_object=False):
        r""" Shows the `type`, `length` and `shape` of each object in an iterable 
        and optionally shows the object as well.

        :param x:           the object to get info about
        :param show_object: if `True`, prints the object itself

        .. seealso::
            :func:`~known.basic.Verbose.info`
        """
        for t,x in enumerate(X):
            print(f'[# {t}]')
            __class__.info(x, show_object=show_object)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Remap:
    r""" 
    Provides a mapping between ranges, works with scalars, ndarrays and tensors.

    :param Input_Range:     *FROM* range for ``i2o`` call, *TO* range for ``o2i`` call
    :param Output_Range:    *TO* range for ``i2o`` call, *FROM* range for ``o2i`` call

    .. note::
        * :func:`~known.basic.REMAP.i2o`: maps an input within `Input_Range` to output within `Output_Range`
        * :func:`~known.basic.REMAP.o2i`: maps an input within `Output_Range` to output within `Input_Range`

    Examples::

        >>> mapper = REMAP(Input_Range=(-1, 1), Output_Range=(0,10))
        >>> x = np.linspace(mapper.input_low, mapper.input_high, num=5)
        >>> y = np.linspace(mapper.output_low, mapper.output_high, num=5)

        >>> yt = mapper.i2o(x)  #<--- should be y
        >>> xt = mapper.o2i(y) #<----- should be x
        >>> xE = np.sum(np.abs(yt - y)) #<----- should be 0
        >>> yE = np.sum(np.abs(xt - x)) #<----- should be 0
        >>> print(f'{xE}, {yE}')
        0, 0
    """

    def __init__(self, Input_Range:tuple, Output_Range:tuple) -> None:
        r"""
        :param Input_Range:     `from` range for ``i2o`` call, `to` range for ``o2i`` call
        :param Output_Range:    `to` range for ``i2o`` call, `from` range for ``o2i`` call
        """
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)

    def set_input_range(self, Range:tuple) -> None:
        r""" set the input range """
        self.input_low, self.input_high = Range
        self.input_delta = self.input_high - self.input_low

    def set_output_range(self, Range:tuple) -> None:
        r""" set the output range """
        self.output_low, self.output_high = Range
        self.output_delta = self.output_high - self.output_low

    def backward(self, X):
        r""" maps ``X`` from ``Output_Range`` to ``Input_Range`` """
        return ((X - self.output_low)*self.input_delta/self.output_delta) + self.input_low

    def forward(self, X):
        r""" maps ``X`` from ``Input_Range`` to ``Output_Range`` """
        return ((X - self.input_low)*self.output_delta/self.input_delta) + self.output_low

    def __call__(self, X, backward=False):
        return self.backward(X) if backward else self.forward(X)
    
    def swap_range(self):
        Input_Range, Output_Range = (self.output_low, self.output_high), (self.input_low, self.input_high)
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class IndexedDict(UserDict):
    r""" Implements an Indexed dict where values can be addressed using both index(int) and keys(str) """

    def __init__(self, **members) -> None:
        self.names = []
        super().__init__(*[], **members)
    
    def keys(self): return enumerate(self.names, 0) # for i,k in self.keys()

    def items(self): return enumerate(self.data.items(), 0) # for i,(k,v) in self.items()

    def __len__(self): return len(self.data)

    def __getitem__(self, name): 
        if isinstance(name, int): name = self.names[name]
        if name in self.data: 
            return self.data[name]
        else:
            raise KeyError(name)

    def __setitem__(self, name, item): 
        if isinstance(name, int): name = self.names[name]
        if name not in self.data: self.names.append(name)
        self.data[name] = item

    def __delitem__(self, name): 
        index = None
        if isinstance(name, int):  
            index = name
            name = self.names[name]
        if name in self.data: 
            del self.names[self.names.index(name) if index is None else index]
            del self.data[name]

    def __iter__(self): return iter(self.names)

    def __contains__(self, name): return name in self.data

    # Now, add the methods in dicts but not in MutableMapping

    def __repr__(self) -> str:
        return f'{__class__} :: {len(self)} Members'
    
    def __str__(self) -> str:
        items = ''
        for i,k in enumerate(self):
            items += f'[{i}] \t {k} : {self[i]}\n'
        return f'{__class__} :: {len(self)} Members\n{items}'
    
    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        inst.__dict__["names"] = self.__dict__["names"].copy()
        return inst

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



# def pname(path:str, sep:str='.'): 
#     r""" Path Name - retuns the path except file extension using ``path[0:path.rfind(sep)]`` 
    
#     .. seealso::
#         :func:`~known.basic.pext`
#         :func:`~known.basic.psplit`
#     """
#     return path[0:path.rfind(sep)]

# def pext(path:str, sep:str='.'): 
#     r""" Path Extension - retuns the extension from a path using ``path[path.rfind(sep):]`` 

#     .. seealso::
#         :func:`~known.basic.pname`
#         :func:`~known.basic.psplit`
#     """
#     return path[path.rfind(sep):]

# def psplit(path:str, sep:str='.'): 
#     r""" Path Split - splits the path into name and extension

#     :returns: 2-tuple (Name, Ext)

#     .. note:: This is the same as using :func:`~known.basic.pname` and :func:`~known.basic.pext` together. 
#         This may be used to create copies of a file by adding a suffix to its name witout changing the extension.

#     """
#     return (path[0:path.rfind(sep)], path[path.rfind(sep):])
