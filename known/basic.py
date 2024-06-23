#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [ 'HRsizes', 'EveryThing', 'Kio', 'Verbose', 'Remap',  'BaseConvert', 'IndexedDict', 'Zipper', 'Mailer' ]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from typing import Any, Union, Iterable, Callable #, BinaryIO, cast, Dict, Optional, Type, Tuple, IO
import os, platform, datetime, smtplib, mimetypes, json, pickle
from math import floor, log, ceil
from zipfile import ZipFile
from email.message import EmailMessage
from collections import UserDict
from io import BytesIO


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class HRsizes: # covert human readable size (like  "12.5MB") to bytes and visa versa
    mapper = dict(BB=1, KB=2**10, MB=2**20, GB=2**30, TB=2**40) # 2 chars for keys
    @staticmethod
    def tobytes(size:str): return round(float(size[:-2])*__class__.mapper.get(size[-2:].upper(), 0))
    @staticmethod
    def tostr(size:int, roundoff=2):
        if      size<__class__.mapper["KB"]: return f"{size}BB"
        elif    size<__class__.mapper["MB"]: return f"{round(size/(__class__.mapper['KB']),roundoff)}KB"
        elif    size<__class__.mapper["GB"]: return f"{round(size/(__class__.mapper['MB']),roundoff)}MB"
        elif    size<__class__.mapper["TB"]: return f"{round(size/(__class__.mapper['GB']),roundoff)}GB"
        else                               : return f"{round(size/(__class__.mapper['TB']),roundoff)}TB"

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class EveryThing: # use as a set that contains everything (use with 'in' keyword)
    def __contains__(self, x): return True

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Kio:
    r""" provides input/out methods for loading saving python objects using json and pickle """
    IOAS = dict(json=json, pickle=pickle)
    IOFLAG = dict(json='', pickle='b')

    @staticmethod
    def save_buffer(o:Any, ioas:str, seek0=False, **kwargs) -> None:
        d = __class__.IOAS.get(f'{ioas}', None)
        assert d is not None, f'key error {ioas}'
        buffer = BytesIO()
        d.dump(o, buffer)
        if seek0: buffer.seek(0) # prepares for reading
        return buffer

    @staticmethod
    def load_buffer(buffer:BytesIO, ioas:str, seek0=True): 
        d = __class__.IOAS.get(f'{ioas}', None)
        assert d is not None, f'key error {ioas}'
        if seek0: buffer.seek(0) # prepares for reading
        return d.load(buffer)

    @staticmethod
    def save_file(o:Any, path:str, ioas:str, **kwargs) -> None:
        d = __class__.IOAS.get(f'{ioas}', None)
        assert d is not None, f'key error {ioas}'
        with open(path, f'w{__class__.IOFLAG[d]}') as f: d.dump(o, f, **kwargs)
        return path

    @staticmethod
    def load_file(path:str, ioas:str):
        d = __class__.IOAS.get(f'{ioas}', None)
        assert d is not None, f'key error {ioas}'
        with open(path, f'r{__class__.IOFLAG[d]}') as f: o = d.load(f)
        return o

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Verbose:
    r""" Contains shorthand helper functions for printing outputs and representing objects as strings.

    .. note::
        This class contains only static methods.
    """
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
    def strA_(arr:Iterable, start:str="", sep:str="|", end:str="") -> str:
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
    def strA(arr:Iterable, start:str="", sep:str="|", end:str="") -> None: print(__class__.strA_(arr, start, sep, end))
    
    @staticmethod
    def strD_(arr:Iterable, sep:str="\n", cep:str=":\n", caption:str="") -> str:
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
    def strD(arr:Iterable, sep:str="\n", cep:str=":\n", caption:str="") -> None: print(__class__.strD_(arr, sep, cep, caption))

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

    @staticmethod
    def show_(x:Any, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
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
    def show(x:Any, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
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
        print(__class__.show_(x, cep=cep, sw=sw, ew=ew))

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

class Zipper:
    r""" zip API using ZipFile package """

    @staticmethod
    def zip_files(zip_path:str, files, **kwargs):
        r""" zips all (only files) in the list of file paths and saves at 'zip_path' """
        zipped = 0
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'
        with ZipFile(zip_path, 'w', **kwargs) as zip_object:
            for path in files:
                if not os.path.isfile(path): continue
                zip_object.write(f'{path}')
                zipped+=1
        return zipped, zip_path

    @staticmethod
    def get_all_file_paths(directory):
        r""" recursively list all files in a folder """
        file_paths = []
        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory):
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        return file_paths   

    @staticmethod
    def zip_folders(zip_path:str, folders, **kwargs):  
        r""" zip multiple folders into a single zip file """    
        if isinstance(folders, str): folders= [f'{folders}']

        if not zip_path : zip_path = f'{folders[0]}.zip'
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  
        all_files = []
        for folder in folders: all_files.extend(__class__.get_all_file_paths(folder))
        return __class__.zip_files(f'{zip_path}', all_files, **kwargs)
    
    @staticmethod
    def zip_folder(folder:str, **kwargs):
        r""" zip a single folder with the same zip file name """     
        return  __class__.zip_files(f'{folder}.zip', __class__.get_all_file_paths(folder),  **kwargs)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Mailer:
    r""" Use a g-mail account to send mail. 

    .. warning:: security concern
        You should enable 2-factor-auth in gmail and generate an app-password instead of using your gmail password.
        Visit (https://myaccount.google.com/apppasswords) to generate app-password.
        Usually, these type of emails are treated as spam by google, so they must be marked 'not spam' at least once.
        It is recomended to create a seperate gmail account for sending mails.
    """
    
    DEFAULT_CTYPE = 'application/octet-stream'  

    @staticmethod
    def global_alias(prefix=''): return f'{prefix}{os.getlogin()} @ {platform.node()}:{platform.system()}.{platform.release()}'

    @staticmethod
    def get_mime_types(files):
        r""" gets mimetype info all files in a list """
        if isinstance(files, str): files=[f'{files}']
        res = []
        for path in files:
            if not os.path.isfile(path): continue
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None: ctype = __class__.DEFAULT_CTYPE  
            maintype, subtype = ctype.split('/', 1)
            res.append( (path, maintype, subtype) )
        return res

    @staticmethod
    def compose_mail( subject:str, rx:str, cc:str, bcc:str, content:str, signature:str, attached, verbose=True):
        r""" compose an e-mail msg to send later
        
        :param subject: subject
        :param rx: csv recipent email address
        :param cc: csv cc email address
        :param content: main content
        :param attached: list of attached files - is a 2-tupe (attachment_type, (args...) )

        # attach all files in the list :: ('', ('file1.xyz', 'file2.xyz'))
        # zip all the files in the list :: ('zipname.zip', '(file1.xyz', 'file2.xyz'))
        """
        
        msg = EmailMessage()

        # set subject
        msg['Subject'] = f'{subject}'
        if verbose: print(f'SUBJECT: {subject}')

        # set to
        msg['To'] = rx
        if verbose: print(f'TO: {rx}')

        if cc: msg['Cc'] = cc
        if verbose: print(f'CC: {cc}')

        if bcc: msg['Bcc'] = bcc
        if verbose: print(f'BCC: {bcc}')

        # set content
        body = content + signature
        msg.set_content(body)
        if verbose: print(f'MESSAGE: #[{len(body)}] chars.')

        default_attached = []

        attached = [] if attached is None else attached
        for (attach_type, attach_args) in attached:
            if verbose: print(f' ... processing attachement :: {attach_type} :: {attach_args}')

            all_files = []
            for path in attach_args:
                if os.path.isdir(path):
                    all_files.extend(Zipper.get_all_file_paths(path))
                elif os.path.isfile(path):
                    all_files.append(path)
                else:
                    if verbose: print(f'[!] Invalid Path :: {path}, skipped...')

            if not attach_type:  # attach individually
                default_attached.extend(__class__.get_mime_types(all_files))
            else: # make zip
                zipped, zip_path=Zipper.zip_files(attach_type, all_files)
                if verbose: print(f'\t --> zipped {zipped} items @ {zip_path} ')
                if zipped>0:
                    default_attached.extend(__class__.get_mime_types(zip_path))
                else:
                    if verbose: print(f'[!] [{zip_path}] is empty, will not be attched!' )
                    try:
                        os.remove(zip_path)
                        if verbose: print(f'[!] [{zip_path}] was removed.' )
                    except:
                        if verbose: print(f'[!] [{zip_path}] could not be removed.' ) 
                

        # set attached ( name, main_type, sub_type), if sub_type is none, auto-infers using imghdr
        for file_name,main_type,sub_type in default_attached:
            if verbose: print(f'[+] Attaching file [{main_type}/{sub_type}] :: [{file_name}]')
            with open (file_name, 'rb') as f: 
                file_data = f.read()
            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def send_mail(login, msg, verbose=True):
        r""" send a msg using smtp.gmail.com:587 with provided credentials """
        username, password = login()
        if verbose: print(f'[*] Sending Email from {username}')
        msg['From'] = f'{username}' # set from
        with smtplib.SMTP('smtp.gmail.com', 587) as smpt:
            smpt.starttls()
            smpt.login(username, password)
            smpt.ehlo()
            smpt.send_message(msg)
        if verbose: print(f'[*] Sent!')

    @staticmethod
    def send(login:Union[Callable, tuple, list, dict], subject:str, rx:str, cc:str, bcc:str, content:str, signature:str, attached, verbose=True):
        if isinstance(login, (list, tuple)):
            username, password = login
            login = lambda: (username, password)
        elif isinstance(login, dict):
            username, password = login.get('username'), login.get('password')
            login = lambda: (username, password)
        else:   pass # assume callable
        __class__.send_mail(login,  __class__.compose_mail(subject, rx, cc, bcc, content, signature, attached, verbose), verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
