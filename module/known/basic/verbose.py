__doc__=r"""
===========
Verbose
===========

Contains functions that help to make our outputs properly formatted and pretty on the console.
Shorthands for printing deailed information about objects like arrays and iterables.

"""

__all__ = [
    'now', 'fdate', 'pdate', 
    'strN', 'tabN', 'spaceN', 'recP',
    'strA', 'strD', 'strU', 'uid',
    'show', 'showX', 'about', 'abouts',
    'SpecialSymbols',
]
#-----------------------------------------------------------------------------------------------------
import datetime


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Aliased functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

now = datetime.datetime.now
fdate = datetime.datetime.strftime
pdate = datetime.datetime.strptime


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Printing functions (works for iterables like numpy and torch)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

def strN(s:str, n:int) -> str:  return ''.join([s for _ in range(n)]) # repeates a string n-times 
def tabN(n:int) -> str:         return strN('\t', n)
def spaceN(n:int) -> str:       return strN(' ', n)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def _recP_(a, level, index, pindex, show_dim=False, P=print):
    r""" helper function for recP - do not use directly """
    if index<0: index=''
    dimstr = ('* ' if level<1 else f'*{level-1} ') if show_dim else ''
    pindex = f'{pindex}{index}'
    if len(a.shape)==0:
        P(f'{tabN(level)}[ {dimstr}@{pindex}\t {a} ]') 
    else:
        P(f'{tabN(level)}[ {dimstr}@{pindex} #{a.shape[0]}')
        for i,s in enumerate(a):
            _recP_(s, level+1, i, pindex, show_dim, P)
        P(f'{tabN(level)}]')
def recP(arr, show_dim=False, P=print) -> None: 
    r""" Recursive Print - print an array recursively with added indentation.

    :param arr: an array/tensor/iterable with `.shape` property.
    :param show_dim: `bool`, if true prints the array dimsion at the start of each dimension
    :param P: callable like `print()`
    """
    _recP_(arr, 0, -1, '', show_dim, P) 
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="", sep="|", end="") -> str:
    r""" String Array - returns a string representation of an array/list for printing.
    
    :param arr: an `iterable`
    :param start: `str` - starting string prefix
    :param sep: `str` - seperator of array values
    :param end: `str` - ending string postfix

    :returns: a string representation of array
    """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep=":\n", caption="") -> str:
    r""" String Dict - returns a string representation of a dict object for printing.
    
    :param arr: an `iterable`
    :param sep: `str` - seperator of array values
    :param cep: `str` - seperator of key-value pairs
    :param caption: `str` - heading added at the top

    :returns: a string representation of dict
    """
    res="=-=-=-=-==-=-=-=-={}DICT #[{}] : {}{}=-=-=-=-==-=-=-=-={}".format(sep, len(arr), caption, sep, sep)
    for k,v in arr.items():
        res+=str(k) + cep + str(v) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strU(form=["%Y","%m","%d","%H","%M","%S","%f"], start='', sep='', end='') -> str:
    r""" String UID - formated time stamp based UID ~ default form=["%Y","%m","%d","%H","%M","%S","%f"] 

    > This may be useful in generating unique filenames based on timestamps.

    :param form: the format of datetime stamp
    :param start: UID prefix
    :param sep: UID seperator
    :param end: UID postfix (usually file extentions can be added here)
    """
    return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def uid(year=True, month=True, day=True, hour=True, minute=True, second=True, mirco=True, start='', sep='', end='') -> str:
    r""" Unique Identification - useful in generating unique filenames based on timestamps.
    
    > based on `strU` but takes `bool` arguments instead of datetime format.

    :param year: `bool` - use year in uid
    :param month: `bool` - use month in uid
    :param day: `bool` - use day in uid
    :param hour: `bool` - use hour in uid
    :param minute: `bool` - use minute in uid
    :param second: `bool` - use second in uid
    :param mirco: `bool` - use mirco-second in uid
    :param start: UID prefix
    :param sep: UID seperator
    :param end: UID postfix (usually file extentions can be added here)

    :returns: `str` - uid string
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
    return strU(form=form, start=start, sep=sep, end=end)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(x, cep='\t\t:', sw='__', ew='__', P = print) -> None:
    """ Show Object - describes members of an object using the `dir` call.

    Note: `string.startswith` and `string.endswith` checks are performed on each property/member of the object 
    and only matching properties are displayed. This is usually done to prevent showing dunder members.

    :param x: - the object to be described
    :param cep: - `str` - the name-value seperator
    :param sw: - `str` or `tuple[str]` - argument for `startswith` to check in property name
    :param ew: - `str` or `tuple[str]` - argument for `endswith` to check in property name
    :param P: - callable like `print()`
    """
    for d in dir(x):
        if not (d.startswith(sw) or d.endswith(ew)):
            v = ""
            try:
                v = getattr(x, d)
            except:
                v='?'
            P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def showX(x, cep='\t\t:',P = print) -> None:
    """ Show Object Xtended - describes members of an object using the `dir` call.

    Note: same as `show` but skips `startswith` and `endswith` checks, all property/member are shown.

    :param x: - the object to be described
    :param cep: - `str` - the name-value seperator
    :param P: - callable like `print()`
    """
    for d in dir(x):
        v = ""
        try:
            v = getattr(x, d)
        except:
            v='?'
        P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def about(O, show_object=False, P=print):
    r""" about - shows the `type` and `length` of object.

    > This is used to check `ndarray`, `tensors`, `list`, `tuples` which are usually the 
    output of other functions without having to print the full output which may take up lot of console space.
    
    :param O: the input object 
    :param show_object: `bool` - if True, prints the object itself.
    :param P: - callable like `print()`
    """
    P(f'type: {type(O)}')
    if hasattr(O, '__len__'):
        P(f'len: {len(O)}')
    if hasattr(O, 'shape'):
        P(f'shape: {O.shape}')
    if show_object:
        P(f'object:\n{O}')
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def abouts(Ot, show_object=False, P=print):
    r""" abouts - shows the `type` and `length` of objects in an iterable of objects.

    > This is used to check `ndarray`, `tensors`, `list`, `tuples` which are usually the 
    output of other functions without having to print the full output which may take up lot of console space.
    
    > Same as `about` but takes an iterable of objects`

    :param Ot: the iterable of input object 
    :param show_object: `bool` - if True, prints the object itself.
    :param P: - callable like `print()`
    """
    for t,O in enumerate(Ot):
        P(f'[# {t}]')
        about(O, show_object=show_object, P=P)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class SpecialSymbols:
    r""" A collection of special symbols that can be used to make your verbose look pretty
    """
    CORRECT = '✓'
    INCORRECT = '✗'

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
