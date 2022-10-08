#-----------------------------------------------------------------------------------------------------
import datetime
from math import floor
#-----------------------------------------------------------------------------------------------------
now = datetime.datetime.now
fake = lambda members: type('object', (object,), members)()
#-----------------------------------------------------------------------------------------------------

class REMAP:
    def __init__(self,Input_Range, Mapped_Range) -> None:
        self.input_range(Input_Range)
        self.mapped_range(Mapped_Range)

    def input_range(self, Input_Range):
        self.Li, self.Hi = Input_Range
        self.Di = self.Hi - self.Li
    def mapped_range(self, Mapped_Range):
        self.Lm, self.Hm = Mapped_Range
        self.Dm = self.Hm - self.Lm
    def map2in(self, m):
        return ((m-self.Lm)*self.Di/self.Dm) + self.Li
    def in2map(self, i):
        return ((i-self.Li)*self.Dm/self.Di) + self.Lm



#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Printing functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strA(arr, start="", sep="|", end=""):
    """ returns a string representation of an array/list for printing """
    res=start
    for a in arr:
        res += (str(a) + sep)
    return res + end
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strD(arr, sep="\n", cep=":\n", caption=""):
    """ returns a string representation of a dict object for printing """
    res="=-=-=-=-==-=-=-=-={}DICT #[{}] : {}{}=-=-=-=-==-=-=-=-={}".format(sep, len(arr), caption, sep, sep)
    for k,v in arr.items():
        res+=str(k) + cep + str(v) + sep
    return res + "=-=-=-=-==-=-=-=-="+sep
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def show(x, cep='\t\t:', sw='__', ew='__', P = print):
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
def showX(x, cep='\t\t:',P = print):
    """ same as showx but shows all members, skip startswith test """
    for d in dir(x):
        v = ""
        try:
            v = getattr(x, d)
        except:
            v='?'
        P(d, cep, v)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def strU(form=["%Y","%m","%d","%H","%M","%S","%f"], start='', sep='', end=''):
    """ formated time stamp based UID ~ default form=["%Y","%m","%d","%H","%M","%S","%f"] """
    return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end
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