#-----------------------------------------------------------------------------------------------------
import datetime
import os.path
from math import floor
#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Aliased functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
now = datetime.datetime.now
fdate = datetime.datetime.strftime
pdate = datetime.datetime.strptime


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Printing functions (works for numpy and torch)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strN(s:str, n:int) -> str: 
    """ repeates a string n-times """
    return ''.join([s for _ in range(n)])
def tabN(n:int) -> str: 
    """ repeats tab n-times """
    return strN('\t', n)
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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Path related functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pjs(*paths) -> str:
    """ Path Joins : joins multiple dirs/files in args using os.path.join """
    res = ''
    for p in paths: res = os.path.join(res, p)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pj(path, sep='/') -> str: 
    """ Path Join : similar to pjs but takes a string instead of individual args """
    return pjs(*path.split(sep))
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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Misc
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class REMAP:
    """ mapping between ranges, works with ndarrays.
        forward: maps an input within Input_Range to output within Output_Range
        backward: maps an input within Output_Range to output within Input_Range
        ** this will work for numpy and torch
        """

    def __init__(self, Input_Range, Output_Range) -> None:
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)

    def set_input_range(self, Range):
        self.input_low, self.input_high = Range
        self.input_delta = self.input_high - self.input_low

    def set_output_range(self, Range):
        self.output_low, self.output_high = Range
        self.output_delta = self.output_high - self.output_low

    def backward(self, X):
        return ((X - self.output_low)*self.input_delta/self.output_delta) + self.input_low

    def forward(self, X):
        return ((X - self.input_low)*self.output_delta/self.input_delta) + self.output_low
    """
    def test(self, num):
        x = np.linspace(self.input_low, self.input_high, num=num)
        y = np.linspace(self.output_low, self.output_high, num=num)

        yt = self.forward(x) #<--- should be y
        xt = self.backward(y) #<----- should be x
        xE = np.sum(np.abs(yt - y))
        yE = np.sum(np.abs(xt - x))

        return xE, yE
    """
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class FAKE:
    """ an object with given members (dict) """

    def __init__(self, **members) -> None:
        for k,v in members.items(): setattr(self, k, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=



#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Shared functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num:int, base:int, digs:int) -> list:
    """ convert base-10 integer (num) to base(base) array of fixed no. of digits (digs) """
    res = [ 0 for _ in range(digs) ]
    q = num
    for i in range(digs): # <-- do not use enumerate plz
        res[i]=q%base
        q = floor(q/base)
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def base2int(num:list, base:int) -> int:
    """ convert array from given base to base-10  --> return integer """
    res = 0
    for i,n in enumerate(num):
        res+=(base**i)*n
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def rcs(arr): # reversed cummulative sum ~ used to calculate 'rewards to go' 
    n = len(arr)
    r = [0 for _ in range(n)] #np.zeros_like(arr)
    for i in reversed(range(n)):
        r[i] = arr[i] + (r[i+1] if i+1 < n else 0)
    return r
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=