#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic/utils.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'ndigs', 'int2base', 'base2int', 
    'numel', 'arange', 
    'REMAP',
    'graphfromimage'
]
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import numpy as np
from numpy import ndarray
from math import floor, log
from typing import Any, Union, Iterable
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def ndigs(num:int, base:int) -> int:
    r""" 
    Returns the number of digits required to represent a base-10 number in the given base.

    :param num:     base-10 number to be represented
    :param base:    base-n number system
    """
    return 1 + (0 if num==0 else floor(log(num, base)))

def int2base(num:int, base:int, digs:int) -> list:
    r""" 
    Convert base-10 integer to a base-n list of fixed no. of digits 

    :param num:     base-10 number to be represented
    :param base:    base-n number system
    :param digs:    no of digits in the output

    :returns:       represented number as a list of ordinals in base-n number system

    .. seealso::
        :func:`~known.basic.utils.base2int`
    """
    if not digs: digs=ndigs(num, base)
    res = [ 0 for _ in range(digs) ]
    q = num
    for i in range(digs): # <-- do not use enumerate plz
        res[i]=q%base
        q = floor(q/base)
    return res

def base2int(num:Iterable, base:int) -> int:
    """ 
    Convert an iterbale of digits in base-n system to base-10 integer

    :param num:     iterable of base-n digits
    :param base:    base-n number system

    :returns:       represented number as a integer in base-10 number system

    .. seealso::
        :func:`~known.basic.utils.int2base`
    """
    res = 0
    for i,n in enumerate(num): res+=(base**i)*n
    return res

def numel(shape) -> int: 
    r""" Returns the number of elements in an array of given shape. """
    return np.prod(np.array(shape))

def arange(shape, start:int=0, step:int=1, dtype=None) -> ndarray: 
    r""" Similar to ``np.arange`` but reshapes the array to given shape. """
    return np.arange(start=start, stop=start+step*numel(shape), step=step, dtype=dtype).reshape(shape)

class REMAP:
    r""" 
    Provides a mapping between ranges, works with scalars, ndarrays and tensors.

    :param Input_Range:     *FROM* range for ``i2o`` call, *TO* range for ``o2i`` call
    :param Output_Range:    *TO* range for ``i2o`` call, *FROM* range for ``o2i`` call

    .. note::
        * :func:`~known.basic.utils.REMAP.i2o`: maps an input within `Input_Range` to output within `Output_Range`
        * :func:`~known.basic.utils.REMAP.o2i`: maps an input within `Output_Range` to output within `Input_Range`

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

    def o2i(self, X):
        r""" maps ``X`` from ``Output_Range`` to ``Input_Range`` """
        return ((X - self.output_low)*self.input_delta/self.output_delta) + self.input_low

    def i2o(self, X):
        r""" maps ``X`` from ``Input_Range`` to ``Output_Range`` """
        return ((X - self.input_low)*self.output_delta/self.input_delta) + self.output_low

def graphfromimage(img_path:str, pixel_choice:str='first', dtype=None) -> ndarray:
    r""" 
    Covert an image to an array (1-Dimensional)

    :param img_path:        path of input image 
    :param pixel_choice:    choose from ``[ 'first', 'last', 'mid', 'mean' ]``

    :returns: 1-D numpy array containing the data points

    .. note:: 
        * This is used to generate synthetic data in 1-Dimension. 
            The width of the image is the number of points (x-axis),
            while the height of the image is the range of data points, choosen based on their index along y-axis.
    
        * The provided image is opened in grayscale mode.
            All the *black pixels* are considered as data points.
            If there are multiple black points in a column then ``pixel_choice`` argument specifies which pixel to choose.

        * Requires ``opencv-python``

            Input image should be readable using ``cv2.imread``.
            Use ``pip install opencv-python`` to install ``cv2`` package
    """
    try:
        import cv2 # pip install opencv-python
    except:
        print(f'[!] failed to import cv2!')
        return None
    img= cv2.imread(img_path, 0)
    imgmax = img.shape[1]-1
    j = img*0
    j[np.where(img==0)]=1
    pixel_choice = pixel_choice.lower()
    pixel_choice_dict = {
        'first':    (lambda ai: ai[0]),
        'last':     (lambda ai: ai[-1]),
        'mid':      (lambda ai: ai[int(len(ai)/2)]),
        'mean':     (lambda ai: np.mean(ai))
    }
    px = pixel_choice_dict[pixel_choice]
    if dtype is None: dtype=np.float_
    return np.array([ imgmax-px(np.where(j[:,i]==1)[0]) for i in range(j.shape[1]) ], dtype=dtype)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=