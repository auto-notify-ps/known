#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [ 
    'Fake', 
    'Infinity', 
    'Kio', 
    'Remap',  
    'BaseConvert', 
    'IndexedDict', 
    'Verbose', 
    'Symbols', 
    'Table', 
    'Fuzz',
    'Mailer',
    ]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import datetime, os
from math import floor, log, ceil
from collections import UserDict
from io import BytesIO
from zipfile import ZipFile
import smtplib, mimetypes
from email.message import EmailMessage

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Fake: # a fake object
    def __init__(self, **attributes):
        for name,value in attributes.items(): setattr(self, name, value)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Infinity: 
    # emulates infinity with comparision operators < <=, >, >=, ==
    #   +inf    =  Infinity(1)
    #   -inf    =  Infinity(-1)
    # emulates Empty and Universal sets with 'in' keyword
    #   Universal Set (everything) = Infinity(1)
    #   Null/Empty Set (nothing) = Infinity(-1)

    def __init__(self, sign=1) -> None: self.sign = (sign>=0) # has positive
    
    def __gt__(self, other): return self.sign       # +inf is greater than everything / -inf is greater than nothing
    def __ge__(self, other): return self.sign       

    def __lt__(self, other): return not self.sign   # -inf is less than everything    / +inf is less than nothing
    def __le__(self, other): return not self.sign

    def __eq__(self, other): return False           # inf is not equal to anything, not even itself

    def __contains__(self, x): return self.sign     # universal set contains everything (always true), empty set contains nothing (always false)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Kio:
    r""" provides input/out methods for loading saving python objects using json and pickle """

    import json, pickle
    IOFLAG = dict(
        json=   (json,      ''     ), 
        pickle= (pickle,    'b'    ),
        )

    @staticmethod
    def get_ioas(): return list(__class__.IOFLAG.keys())

    @staticmethod
    def save_buffer(o, ioas:str, seek0=False) -> None:
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        buffer = BytesIO()
        s_module.dump(o, buffer)
        if seek0: buffer.seek(0) # prepares for reading
        return buffer

    @staticmethod
    def load_buffer(buffer:BytesIO, ioas:str, seek0=True): 
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        if seek0: buffer.seek(0) # prepares for reading
        return s_module.load(buffer)

    @staticmethod
    def save_file(o, path:str, ioas:str, **kwargs):
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        with open(path, f'w{s_flag}') as f: s_module.dump(o, f, **kwargs)
        return path
    @staticmethod
    def load_file(path:str, ioas:str):
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        with open(path, f'r{s_flag}') as f: o = s_module.load(f)
        return o
    
    @staticmethod
    def save_as_json(o, path:str, **kwargs):    return __class__.save_file(o, path, 'json', **kwargs)
    @staticmethod
    def load_as_json(path:str):                     return __class__.load_file(path, 'json')

    @staticmethod
    def save_as_pickle(o, path:str, **kwargs):  return __class__.save_file(o, path, 'pickle', **kwargs)
    @staticmethod
    def load_as_pickle(path:str):                   return __class__.load_file(path, 'pickle')

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Remap:
    r""" 
    Provides a mapping between ranges, works with scalars, ndarrays and tensors.

    :param Input_Range:     *FROM* range for ``forward`` call, *TO* range for ``backward`` call
    :param Output_Range:    *TO* range for ``forward`` call, *FROM* range for ``forward`` call
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

class BaseConvert:
    
    r""" Number System Conversion 
    
    A number is abstract concept that has many representations using sets of symbols

    A base-n number system uses a set of n digits to represent any number
    This is called the representation of the number

    Given one representation, we only need to convert to another

    """

    @staticmethod
    def zeros(n): return [0 for _ in range(n)]

    @staticmethod
    def convert(digits, base_from, base_to, reversed=True):
        r""" convers from one base to another 
        
        :param digits:      iterable of digits in base ```base_from```. NOTE: digits are Natural Numbers starting at 0. base 'b' will have digits between [0, b-1]
        :param base_from:   int - the base to convert from
        :param base_to:     int - the base to convert to
        :param reversed:    bool - if True, digits are assumed in reverse (human readable left to right)
                            e.g. if reversed is True then binary digits iterable [1,0,0] will represent [4] in decimal otherwise it will represent [1] in decimal
        """

        digits_from =  [int(abs(d)) for d in digits] # convert to int data-type
        if reversed: digits_from = digits_from[::-1]
        ndigits_from = len(digits_from)
        mult_from = [base_from**i for i in range(ndigits_from)]
        repr_from = sum([ui*vi for ui,vi in zip(digits_from,mult_from, strict=True)]) #dot(digits_from , mult_from)

        #ndc = base_from**ndigits_from
        ndigits_to = ceil(log(repr_from,base_to))
        digits_to =  __class__.zeros(ndigits_to) 
        n = int(repr_from)
        for d in range(ndigits_to):
            digits_to[d] = n%base_to
            n=n//base_to

        if reversed: digits_to = digits_to[::-1]
        return tuple(digits_to)


    @staticmethod
    def ndigits(num:int, base:int): return ceil(log(num,base))

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
        
        ndigits = digs if digs else ceil(log(num,base)) 
        digits =  __class__.zeros(ndigits)
        n = num
        for d in range(ndigits):
            digits[d] = n%base
            n=n//base
        return digits

    @staticmethod
    def base2int(num, base:int) -> int:
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
        return int(res)


    SYM_BIN = { f'{i}':i for i in range(2) }
    SYM_OCT = { f'{i}':i for i in range(8) }
    SYM_DEC = { f'{i}':i for i in range(10) }
    SYM_HEX = {**SYM_DEC , **{ s:(i+10) for i,s in enumerate(('A', 'B', 'C', 'D', 'E', 'F'))}}
    
    @staticmethod
    def n_syms(n): return { f'{i}':i for i in range(n) }

    @staticmethod
    def to_base_10(syms:dict, num:str):
        b = len(syms)
        l = [ syms[n] for n in num[::-1] ]
        return __class__.base2int(l, b)

    @staticmethod
    def from_base_10(syms:dict, num:int, joiner='', ndigs=None):
        base = len(syms)
        #print(f'----{num=} {type(num)}, {base=}, {type(base)}')
        if not ndigs: ndigs = (1 + (0 if num==0 else floor(log(num, base))))  # __class__.ndigs(num, base)
        ss = tuple(syms.keys())
        S = [ ss[i]  for i in __class__.int2base(num, base, ndigs) ]
        return joiner.join(S[::-1])


    @staticmethod
    def int2hex(num:int, joiner=''): return __class__.from_base_10(__class__.SYM_HEX, num, joiner)
  
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

class Verbose:
    r""" Contains shorthand helper functions for printing outputs and representing objects as strings.
            Methods ending with '_' in name return Strings instead of printing them
    """
    
    #-----------------------------------------------------------------------
    """ SECTION: HRS - Human Readable String  for sizes"""
    #-----------------------------------------------------------------------
    HRS_MAPPER = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40) # 2 chars for keys

    @staticmethod
    def hrs2bytes(hrsize:str): return round(float(hrsize[:-2])*__class__.HRS_MAPPER.get(hrsize[-2:].upper(), 0))
    @staticmethod
    def bytes2hrs(size:int, unit:str, roundoff=2): return f"{round(size/(__class__.HRS_MAPPER[unit]),roundoff)}{unit}"

    @staticmethod
    def bytes2bb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'BB', roundoff)
    @staticmethod
    def bytes2kb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'KB', roundoff)
    @staticmethod
    def bytes2mb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'MB', roundoff)
    @staticmethod
    def bytes2gb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'GB', roundoff)
    @staticmethod
    def bytes2tb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'TB', roundoff)

    @staticmethod
    def bytes2auto(size:int, roundoff=2):
        if      size<__class__.HRS_MAPPER["KB"]: return __class__.bytes2bb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["MB"]: return __class__.bytes2kb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["GB"]: return __class__.bytes2mb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["TB"]: return __class__.bytes2gb(size, roundoff)
        else                                   : return __class__.bytes2tb(size, roundoff)
    #-----------------------------------------------------------------------


    #-----------------------------------------------------------------------
    """ SECTION: StrX - human readable string representation of objects """
    #-----------------------------------------------------------------------

    DEFAULT_DATE_FORMAT = ["%Y","%m","%d","%H","%M","%S","%f"] #  Default date format for :func:`~known.basic.Verbose.strU` 
    DASHED_LINE = "=-=-=-=-==-=-=-=-="

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
    def recP(arr, show_dim:bool=False) -> None: 
        r"""
        Recursive Print - print an iterable recursively with added indentation.

        :param arr:         any iterable with ``shape`` property.
        :param show_dim:    if `True`, prints the dimension at the start of each item
        """
        __class__._recP_(arr, 0, -1, '', '\t', show_dim)
    
    @staticmethod
    def strA_(arr, start:str="", sep:str="|", end:str="") -> str:
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
    def strA(arr, start:str="", sep:str="|", end:str="") -> None: print(__class__.strA_(arr, start, sep, end))
    
    @staticmethod
    def strD_(arr, sep:str="\n", cep:str=":\n", caption:str="") -> str:
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
    def strD(arr, sep:str="\n", cep:str=":\n", caption:str="") -> None: print(__class__.strD_(arr, sep, cep, caption))

    @staticmethod
    def strU(form, start:str='', sep:str='', end:str='') -> str:
        r""" 
        String UID - returns a formated string of current timestamp.

        :param form: the format of timestamp, If `None`, uses the default :data:`~known.basic.Verbose.DEFAULT_DATE_FORMAT`.
            Can be selected from a sub-set of ``["%Y","%m","%d","%H","%M","%S","%f"]``.
            
        :param start: UID prefix
        :param sep: UID seperator
        :param end: UID postfix

        """
        if not form: form = __class__.DEFAULT_DATE_FORMAT
        return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end

    @staticmethod
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


    #-----------------------------------------------------------------------
    """ SECTION: show/info - human readable information about pbjects """
    #-----------------------------------------------------------------------

    DOCSTR_FORM = lambda x: f'\t!docstr:\n! - - - - - - - - - - - - - - - - -\n{x}\n- - - - - - - - - - - - - - - - - !'

    @staticmethod
    def show_(x, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
        res = ""
        for d in dir(x):
            if not (d.startswith(sw) or d.endswith(ew)):
                v = ""
                try:
                    v = getattr(x, d)
                except:
                    v='?'
                res+=f'({d} {cep} {v}{sep}'
        return res

    @staticmethod
    def show(x, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
        r"""
        Show Object - describes members of an object using the ``dir`` call.

        :param x:       the object to be described
        :param cep:     the name-value seperator
        :param sw:      argument for ``startswith`` to check in member name
        :param ew:      argument for ``endswith`` to check in member name

        .. note:: ``string.startswith`` and ``string.endswith`` checks are performed on each member of the object 
            and only matching member are displayed. This is usually done to prevent showing dunder members.
        
        .. seealso::
            :func:`~known.basic.Verbose.dir`
        """
        print(__class__.show_(x, cep=cep, sw=sw, ew=ew))

    @staticmethod
    def dir(x, doc=False, filter:str='', sew=('__','__')):
        """ Calls ```dir``` on given argument and lists the name and types of non-dunder members.

        :param filter: csv string of types to filter out like `type,function,module`, keep blank for no filter
        :param doc: shows docstring ```__doc``` 
            If ```doc``` is True, show all member's ```__doc__```.
            If ```doc``` is False, does not show any ```__doc__```. 
            If ```doc``` is a string, show ```__doc__``` of specific types only given by csv string.

        :param sew: 2-Tuple (start:str, end:str) - excludes member names that start and end with specific chars, 
            used to exclude dunder methods by default

        .. seealso::
            :func:`~known.basic.Verbose.show`
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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Symbols:
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
        DOT=            '●'
        SUN=            '⚙'
        ARROW1=         '↦'
        ARROW2=         '⇒'
        ARROW3=         '↪'

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Table:

    @staticmethod
    def _Create(*columns):
        data = {None:[f'{col}' for col in columns]} # this is to make sure that col names are always on top
        return data

    @staticmethod
    def Create(columns:tuple, primary_key:str, cell_delimiter=',', record_delimiter='\n'):
        # should be called on a new object after init\
        table = __class__()
        table.data = __class__._Create(*columns)
        table.pk = primary_key
        table.pkat = table.data[None].index(table.pk)
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def _Import(path, key_at, cell_delimiter, record_delimiter): 
        with open(path, 'r', encoding='utf-8') as f: 
            s = f.read()
            lines = s.split(record_delimiter)
            cols = lines[0].split(cell_delimiter) #<--- only if None:cols was added as a first entry (using Create method)
            data = {None:cols}
            if isinstance(key_at, str): key_at = cols.index(key_at)
            assert key_at>=0,f'Invlaid key {key_at}'
            for line in lines[1:]:
                if line:
                    cells = line.split(cell_delimiter)
                    data[f'{cells[key_at]}'] = cells
        return data
    
    @staticmethod
    def Import(path, key_at, cell_delimiter=',', record_delimiter='\n'): 
        table = __class__()
        table.data = __class__._Import(path, key_at, cell_delimiter, record_delimiter)
        if isinstance(key_at, str): key_at = table[None].index(key_at)
        table.pk = table.data[None][key_at]
        table.pkat = key_at
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def _Export(data, path, cell_delimiter, record_delimiter): 
        with open(path, 'w', encoding='utf-8') as f: 
            for v in data.values(): f.write(cell_delimiter.join(v)+record_delimiter)

    @staticmethod
    def Export(table, path): 
        __class__._Export(table.data, path, table.cell_delimiter, table.record_delimiter)

    @property
    def cols(self): return self[None]

    @property
    def keys(self): return set([k for k in self.data.keys() if k is not None])

    # get row as dict
    def __call__(self, key): return {k:v for k,v in zip(self[None], self[key])}

    # get row as it is (list)
    def __getitem__(self, key): return self.data[key]

    # set row based on if its a dict or a list (note: key is irrelavant here)
    def __setitem__(self, key, row):
        assert key is not None, f'Cannot set `None` key'
        assert isinstance(key, str), f'Expected string type keys but got {type(key)}'
        if isinstance(row, dict):
            # dict does not require to match exact length
            assert self.pk not in row , f'Cannot change primary key' 
            indexof = {k:self[None].index(k) for k in row}
            if not key in self.data: self[key] = None 
            self.data[key][self.pkat] = key
            for k,v in row.items(): self.data[key][indexof[k]] = str(v)
            

        elif isinstance(row, (list, tuple)):
            # must exactly match no of cols, 
            # use ... to keep values unchanged 
            # value are initialized to None
            assert len(row) == len(self[None]), f'Rows are expected to have length {len(self[None])} but got {len(row)}'
            assert row[self.pkat] is ... , f'Cannot change primary key'
            if not key in self.data: self[key] = None 
            self.data[key][self.pkat] = key
            for i,v in enumerate(row):
                if not (v is ...) : self.data[key][i] = str(v)
        elif row is None: 
            # (re)initializes the row
            self.data[key] = ['' for _ in self[None]]
            self.data[key][self.pkat] = key
        else: raise AssertionError(f'Excepting a dict/list/tuple/None but got {type(row)}')

    # del row based on key
    def __delitem__(self, key):
        if key is not None: del self.data[key]

    def __contains__(self, key): return key in self.data

    # quick export > file
    def __gt__(self, other):__class__._Export(self.data, f'{other}', self.cell_delimiter, self.record_delimiter)

    # quick import < file
    def __lt__(self, other): self.data = __class__._Import(f'{other}', self.pkat, self.cell_delimiter, self.record_delimiter)

    # total number of rows
    def __len__(self): return len(self.data)-1

    def __str__(self):
        return ...

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Fuzz:
    r""" file system related """

    @staticmethod
    def ExistInfo(path):
        """ returns 4-tuple (exists,isdir,isfile,islink) """
        return os.path.exists(path), os.path.isdir(path), os.path.isfile(path), os.path.islink(path)
    
    @staticmethod
    def Scan(path, exclude_hidden=False, include_size=False, include_extra=False):
        r""" Scans a directory using os.scandir call 
             returns a list of 6 or 9 tuple (name, path, isdir, isfile, islink, size, parent, fname, ext) """
        if exclude_hidden:  
            if include_size: 
                if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size, os.path.dirname(os.path.abspath(x.path)), *__class__.SplitName(x.name)) for x in os.scandir(path) if not x.name.startswith(".")]
                else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size) for x in os.scandir(path) if not x.name.startswith(".")]

            else:            
                if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              , os.path.dirname(os.path.abspath(x.path)), *__class__.SplitName(x.name)) for x in os.scandir(path) if not x.name.startswith(".")]
                else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ) for x in os.scandir(path) if not x.name.startswith(".")]
        else:
            if include_size: 
                if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size, os.path.dirname(os.path.abspath(x.path)), *__class__.SplitName(x.name)) for x in os.scandir(path)]
                else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), x.stat().st_size) for x in os.scandir(path)]
            else:
                if include_extra:   return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ,os.path.dirname(os.path.abspath(x.path)), *__class__.SplitName(x.name)) for x in os.scandir(path)]
                else:               return [(x.name, os.path.abspath(x.path), x.is_dir(), x.is_file(), x.is_symlink(), -1              ) for x in os.scandir(path)]

    @staticmethod
    def ReScan(path, exclude_hidden=False, include_size=False, expand_links=False, include_extra=False):
        r""" Recursively Scans a directory using os.scandir """
        res = []
        pending = [path]
        while pending:
            ls = __class__.Scan(pending.pop(0), exclude_hidden=exclude_hidden, include_size=include_size, include_extra=include_extra)
            for l in ls: # name, path, isdir, isfile, islink, size, parent, (fname, ext)
                res.append(l)
                if l[2]: (pending.append(l[1]) if not l[4] else (pending.append(l[1]) if expand_links else None))  
        return res

    @staticmethod
    def SplitFileName(path:str): return __class__.SplitName(os.path.basename(path))
        
    @staticmethod
    def SplitName(f:str):
        r"""splits a file-name into name.ext 
        Note: make sure to pass os.path.basename() to this function
        Retutns: 2-tuple (name, ext)
            `name` is always a string
            `ext` can be None or a string (None means that there is no "." in file name)
        """
        i = f.rfind('.')
        return (f, None) if i<0 else (f[0:i], f[i+1:])

    @staticmethod
    def RenameFile(path, new_name, keep_ext=False):
        """ rename a file with or without changing its extension """
        dirname, filename = os.path.dirname(path), os.path.basename(path)
        _, ext = __class__.SplitName(f'{filename}')
        if keep_ext and (ext is not None):  name_ = f'{new_name}.{ext}'
        else:                               name_ = f'{new_name}'
        return os.path.join(dirname, name_)
    
    @staticmethod
    def RenameExt(path, new_ext):
        """ change a file extension without renaming it """
        dirname, filename = os.path.dirname(path), os.path.basename(path)
        name, _ = __class__.SplitName(f'{filename}')
        return os.path.join(dirname, f'{name}.{new_ext}')

    def Zip(nested:bool, zip:str, *paths):

        assert paths, f'no paths provided to zip'
        assert zip,   f'zip path not provided'
        if not zip.lower().endswith('.zip'): zip = f'{zip}.zip'  # append .zip to the end of path
        zipn = zip[:-4]
        paths = list(paths)
        tozip = {}
        while paths:
            path = paths.pop(0)
            
            if isinstance(path, (list, tuple)): item_path, arc_path = path
            else: item_path, arc_path = path, path

            if os.path.islink(item_path): continue
            isfile, isdir = os.path.isfile(item_path), os.path.isdir(item_path)
            if not (isfile or isdir): continue
            arc_path = os.path.basename(arc_path)
            
            if isfile:
                assert arc_path not in tozip, f'Duplicate names detected {arc_path} from {path}'
                tozip[arc_path] = os.path.abspath(item_path)
            else:
                for l in __class__.ReScan(item_path): 
                    arc = os.path.join(arc_path, os.path.relpath( l[1], item_path))
                    assert arc not in tozip, f'Duplicate names detected {arc} from {path}'
                    tozip[arc] = l[1]

        with ZipFile(zip, 'w') as zip_object:
            if nested: zip_object.mkdir(zipn)
            for arc_path, file_path in tozip.items():
                if nested: arc_path = os.path.join(zipn, arc_path)
                zip_object.write(filename=file_path, arcname=arc_path)
        return tozip

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Mailer:
    r"""
    Use gmail account to send mails.
    Use `Mail` static method to send mail which requires username and password
        > username must be a gmail address from which mail will be sent - recivers can be in any domain
        > password can be either one of:
            > google (account) password
            > app password

    Its recomended to use app password, to generate one visit 
        > https://myaccount.google.com/apppasswords
    
    App passwords requires enabling 2-step verification first, to enable it visit
        > https://myaccount.google.com/signinoptions/twosv

    
    -> on the reciver side - mark incoming mails as'not spam' at least once
    """
    @staticmethod
    def Mimes(files, default_ctype='application/octet-stream'):
        r""" gets mimetype info all files in a list """
        if isinstance(files, str): files=[f'{files}']
        res = []
        for path in files:
            if not os.path.isfile(path): continue
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None: ctype = default_ctype
            maintype, subtype = ctype.split('/', 1)
            res.append( (path, maintype, subtype) )
        return res
    
    @staticmethod
    def Compose(From:str, Subject:str, To:str, Cc:str, Body:str, Attached:list=None):
        msg = EmailMessage()

        # from, subject and recivers
        msg['From'] = f'{From}' 
        msg['Subject'] = f'{Subject}'
        msg['To'] = To
        if Cc: msg['Cc'] = Cc
        msg.set_content(Body)

        # attachement
        if Attached is None: Attached = []
        assert isinstance(Attached, (list, tuple)), f'Expecting a list or tuple of attached files but got {type(Attached)}'       
        for file_name,main_type,sub_type in __class__.Mimes(Attached):
            with open (file_name, 'rb') as f: file_data = f.read()
            msg.add_attachment(file_data, 
                                maintype=main_type, 
                                subtype=sub_type, 
                                filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def Send(msg:EmailMessage, username:str, password:str, url='smtp.gmail.com', port='587', tls=True):
        r""" send a msg using url:port with provided credentials, calls `starttls` is `tls` is True """
        if not (username and password): return False
        try:
            with smtplib.SMTP(f'{url}', int(port)) as smpt: 
                if tls: smpt.starttls()
                # assert username == msg['From'] #<--- should be?
                smpt.login(username, password) 
                smpt.ehlo()
                smpt.send_message(msg)
        except: return False
        return True

    @staticmethod
    def Mail(
        username, 
        password, 
        Subject, 
        To, 
        Cc, 
        Body, 
        Attached:list=None,
        url='smtp.gmail.com', port='587', tls=True,
    ): return __class__.Send(__class__.Compose(username, Subject, To, Cc, Body, Attached=Attached), 
            username, password, url=url, port=port, tls=True)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


