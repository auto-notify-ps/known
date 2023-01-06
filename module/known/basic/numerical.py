from math import floor, log
import numpy as np

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def ndigs(num:int, base:int):
    """ number of digits required to represent [num] in [base] """
    return 1 + (0 if num==0 else floor(log(num, base)))
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def int2base(num:int, base:int, digs:int) -> list:
    """ convert base-10 integer (num) to base(base) array of fixed no. of digits (digs) """
    if not digs: digs=ndigs(num, base)
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
    for i,n in enumerate(num): res+=(base**i)*n
    return res
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

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

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# numpy related
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def numel(shape):  
    return np.prod(np.array(shape))
def arange(shape, start=0, step=1, dtype=None): 
    return np.arange(start=start, stop=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
