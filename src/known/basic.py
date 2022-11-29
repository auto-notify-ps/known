#-----------------------------------------------------------------------------------------------------
import datetime
import os.path
from math import floor
#import numpy as np
#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Aliased functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
now = datetime.datetime.now
fdate = datetime.datetime.strftime
pdate = datetime.datetime.strptime
fake = lambda members: type('object', (object,), members)() # a fake object with members (dict)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Path related functions
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def pjs(*paths):
    """ Path Joins : joins multiple dirs/files in args using os.path.join """
    res = ''
    for p in paths: res = os.path.join(res, p)
    return res
def pj(path, sep='/'): 
    """ Path Join : similar to pjs but takes a string instead of individual args """
    return pjs(*path.split(sep))
#-----------------------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Misc
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
class REMAP:
    """ mapping between ranges, works with ndarrays.
        forward: maps an input within Input_Range to output within Output_Range
        backward: maps an input within Output_Range to output within Input_Range
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

class JSON:
    import json # load only when called
    def save(path, data_dict):
        """ saves a dict to disk in json format """
        with open(path, 'w') as f:
            f.write(__class__.json.dumps(data_dict, sort_keys=False, indent=4))
        return path
    def load(path):
        """ returns a dict from a json file """
        data_dict = None
        with open(path, 'r') as f:
            data_dict = __class__.json.loads(f.read())
        return data_dict

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