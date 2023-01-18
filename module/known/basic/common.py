#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic/common.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'now', 'fdate', 'pdate', 
    'uid', 'pjs', 'pj', 'pname', 'pext', 'psplit',
    'Fake', 'Verbose'
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import datetime, os
from typing import Any, Union, Iterable
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# Aliased functions

now = datetime.datetime.now
fdate = datetime.datetime.strftime
pdate = datetime.datetime.strptime


def uid(year:bool=True, month:bool=True, day:bool=True, 
        hour:bool=True, minute:bool=True, second:bool=True, mirco:bool=True, 
        start:str='', sep:str='', end:str='') -> str:
    r""" Unique Identifier - useful in generating unique identifiers based on current timestamp. 
    Helpful in generating unique filenames based on timestamps. 
    
    .. seealso::
        :func:`~known.basic.common.Verbose.strU`
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

def pjs(*paths) -> str:
    r""" Paths Join - shorthand for `os.path.join` """
    return os.path.join('', *paths)

def pj(path:str, sep:str='/') -> str: 
    r""" Path Join - shorthand for `os.path.join`

    .. note:: This is similar to :func:`~known.basic.common.pjs` but instead of taking multiple args,
        takes a single string and splits it using the provided seperator.
    """
    return pjs(*path.split(sep))

def pname(path:str, sep:str='.'): 
    r""" Path Name - retuns the path except file extension using ``path[0:path.rfind(sep)]`` 
    
    .. seealso::
        :func:`~known.basic.common.pext`
        :func:`~known.basic.common.psplit`
    """
    return path[0:path.rfind(sep)]

def pext(path:str, sep:str='.'): 
    r""" Path Extension - retuns the extension from a path using ``path[path.rfind(sep):]`` 

    .. seealso::
        :func:`~known.basic.common.pname`
        :func:`~known.basic.common.psplit`
    """
    return path[path.rfind(sep):]

def psplit(path:str, sep:str='.'): 
    r""" Path Split - splits the path into name and extension

    :returns: 2-tuple (Name, Ext)

    .. note:: This is the same as using :func:`~known.basic.common.pname` and :func:`~known.basic.common.pext` together. 
        This may be used to create copies of a file by adding a suffix to its name witout changing the extension.

    """
    return (path[0:path.rfind(sep)], path[path.rfind(sep):])

class Fake:
    r""" Fake Object - an object with members given in a keyword-args dict """
    def __init__(self, **members) -> None:
        for k,v in members.items(): setattr(self, k, v)

class Verbose:
    r""" Contains shorthand helper functions for printing outputs and representing objects as strings.
    
    Also contains some special symbols described in the table below

    .. list-table:: 
        :widths: 5 3 5 3
        :header-rows: 1

        * - Name
          - Symbol
          - Name
          - Symbol
        * - SYM_CORRECT
          - ✓
          - SYM_INCORRECT
          - ✗
        * - SYM_ALPHA
          - α
          - SYM_BETA
          - β
        * - SYM_GAMMA
          - γ
          - SYM_DELTA
          - δ
        * - SYM_EPSILON
          - ε
          - SYM_ZETA
          - ζ
        * - SYM_ETA
          - η
          - SYM_THETA
          - θ
        * - SYM_KAPPA
          - κ
          - SYM_LAMBDA
          - λ
        * - SYM_MU
          - μ 
          - SYM_XI
          - ξ
        * - SYM_PI
          - π
          - SYM_ROH
          - ρ
        * - SYM_SIGMA
          - σ
          - SYM_PHI
          - φ
        * - SYM_PSI
          - Ψ
          - SYM_TAU
          - τ
        * - SYM_OMEGA
          - Ω
          - SYM_TRI
          - Δ

    .. note::
        This class contains only static methods.
    """
    DEFAULT_DATE_FORMAT = ["%Y","%m","%d","%H","%M","%S","%f"]
    r""" Default date format for :func:`~known.basic.common.Verbose.strU` """

    SYM_CORRECT =       '✓'
    SYM_INCORRECT =     '✗'
    SYM_ALPHA =         'α'
    SYM_BETA =          'β'
    SYM_GAMMA =         'γ'
    SYM_DELTA =         'δ'
    SYM_EPSILON =       'ε'
    SYM_ZETA =          'ζ'
    SYM_ETA =           'η'
    SYM_THETA =         'θ'
    SYM_KAPPA =         'κ'
    SYM_LAMBDA =        'λ'
    SYM_MU =            'μ' 
    SYM_XI =            'ξ'
    SYM_PI =            'π'
    SYM_ROH =           'ρ'
    SYM_SIGMA =         'σ'
    SYM_PHI =           'φ'
    SYM_PSI =           'Ψ'
    SYM_TAU =           'τ'
    SYM_OMEGA =         'Ω'
    SYM_TRI =           'Δ'

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

        :param form: the format of timestamp, If `None`, uses the default :data:`~known.basic.common.Verbose.DEFAULT_DATE_FORMAT`.
            Can be selected from a sub-set of ``["%Y","%m","%d","%H","%M","%S","%f"]``.
            
        :param start: UID prefix
        :param sep: UID seperator
        :param end: UID postfix

        .. seealso::
            :func:`~known.basic.common.uid`
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
            :func:`~known.basic.common.Verbose.showX`
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

        .. note:: This is the same as :func:`~known.basic.common.Verbose.show` but skips ``startswith`` and ``endswith`` checks,
            all members are shown including dunder members.

        .. seealso::
            :func:`~known.basic.common.Verbose.show`
        """
        for d in dir(x):
            v = ""
            try:
                v = getattr(x, d)
            except:
                v='?'
            print(d, cep, v)

    @staticmethod
    def info(x:Any, show_object:bool=False):
        r""" Shows the `type`, `length` and `shape` of an object and optionally shows the object as well.

        :param x:           the object to get info about
        :param show_object: if `True`, prints the object itself

        .. note:: This is used to check output of some functions without having to print the full output
            which may take up a lot of console space. Useful when the object are of nested types.

        .. seealso::
            :func:`~known.basic.common.Verbose.infos`
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
            :func:`~known.basic.common.Verbose.info`
        """
        for t,x in enumerate(X):
            print(f'[# {t}]')
            __class__.info(x, show_object=show_object)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
