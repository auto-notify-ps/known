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
    """ helper function for recP """
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
    """ recursively prints a multi-dim array """
    _recP_(arr, 0, -1, '', show_dim, P) 
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="", sep="|", end="") -> str:
    """ returns a string representation of an array/list for printing """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep=":\n", caption="") -> str:
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-={}DICT #[{}] : {}{}=-=-=-=-==-=-=-=-={}".format(sep, len(arr), caption, sep, sep)
    for k,v in arr.items():
        res+=str(k) + cep + str(v) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strU(form=["%Y","%m","%d","%H","%M","%S","%f"], start='', sep='', end='') -> str:
    """ formated time stamp based UID ~ default form=["%Y","%m","%d","%H","%M","%S","%f"] """
    return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def uid(year=True, month=True, day=True, hour=True, minute=True, second=True, mirco=True, start='', sep='', end='') -> str:
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
    """ Note: 'sw' can accept tuples """
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
    """ same as showx but shows all members, skip startswith test """
    for d in dir(x):
        v = ""
        try:
            v = getattr(x, d)
        except:
            v='?'
        P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def about(O, show_object=False):
    print(f'type: {type(O)}')
    if hasattr(O, '__len__'):
        print(f'len: {len(O)}')
    if hasattr(O, 'shape'):
        print(f'shape: {O.shape}')
    if show_object:
        print(f'Object:\n{O}')
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class SpecialSymbols:
    CORRECT = '✓'
    INCORRECT = '✗'