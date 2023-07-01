__doc__=r"""
:py:mod:`known/imgu.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [
    'Pix', 'graphfromimage',
]#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#import datetime, os
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import cv2 # pip install opencv-python

class Pix:
    r""" Holds an Image as a pixel array (dtype = np.int8). Channels take value between 0 to 255.
    
    :param shape: (None) or 3-Tuple(width, height, channels)
    :param path: (None) or string, location of image to load
    :param flag: (str) can be ['g', 'c', 'u'] based on keys in cv2flags


    .. note:: cv2 imread flags
        * 'g'   cv2.IMREAD_GRAYSCALE : (H, W)
        * 'c'   cv2.IMREAD_COLOR :     (H, W, 3) Channels=BGR
        * 'u'   cv2.IMREAD_UNCHANGED : can be either of 'g', 'c' or (H, W, 4) Channels=BGRA
    
        
    """
    cv2flags = {  'g' : cv2.IMREAD_GRAYSCALE,  'c' : cv2.IMREAD_COLOR, 'u' : cv2.IMREAD_UNCHANGED, }

    def __init__(self, shape=None, path=None, flag='u') -> None:
        # shape = (width, height, channels)
        self.init_using_shape = None
        do_fill=None
        assert shape or path, f'either shape or path should be provided, path will take preference'
        if path:
            if shape: print(f'[!] shape provided bu will not be used!')

            img = cv2.imread(path, __class__.cv2flags[flag])
            if (img.ndim==2): img = np.expand_dims(img, -1) # this happens in flag='g'
            assert img.ndim==3, f'Image Array must have 3 dimensions but got {img.ndim}'
            h,w,c = img.shape
            shape = (w,h,c)
            do_fill = img
            self.init_using_shape = False
        else:
            assert shape, f'Shape must be provided if path is not provided!'
            assert len(shape)==3, f'(height, width, channels) must be provided but got {shape}'
            self.init_using_shape = True

        self.shape = shape
        self.xrange, self.yrange, self.channels = self.shape
        self.image = np.zeros(shape=(self.xrange, self.yrange, self.channels), dtype=np.uint8)
        self.img = self.image.swapaxes(0,1)[::-1,:,:] #self.image.swapaxes(0,1)[:,::-1,:]
        if do_fill is not None: self.img[:] = img
        
        # for get_color property
        self.review =  (0,0,0) if self.channels<3 else (2, 1, 0, *range(3, self.channels))
    # tuple(reversed(range(self.channels)))
    
    def save(self, path): return cv2.imwrite(path, self.img ) # plt.imshow(pix.img[:,:,pix.review])
    
    def show(self, fr=0.05, grid='both', show=True): 
        fig = plt.figure(figsize=(self.xrange*fr, self.yrange*fr))
        plt.imshow(self.img[:,:,self.review])
        if grid is not None: plt.grid(axis=grid)
        if show: 
            plt.show()
            plt.close()
            fig = None
        else:
            plt.close()
        return fig

    def get_color(self, x, y):  return self.image[x,y,self.review]

    def fill_image(self, vals): 
        self.image[:] = vals
        return self

    def fill_img(self, vals): 
        self.img[:] = vals
        return self
    
    def get_scaled_range(self):
        return int(self.xrange/self.grid_scale), int(self.yrange/self.grid_scale)

    def set_grid_scale(self, grid_scale):
        self.grid_scale = grid_scale
        return self
    
    def draw_empty_grid(self, back_color=255, line_color=0, save=None):
        grid_scale = self.grid_scale
        grid_width = self.xrange * grid_scale
        grid_height = self.yrange * grid_scale
        grid_channels = self.channels
        grid = Pix(shape=(grid_width, grid_height, grid_channels)).fill_image(back_color)
        for x in range(self.xrange):
            grid.image[x*grid_scale, :, :] = line_color # black
        for y in range(self.yrange):
            grid.image[:, y*grid_scale, :] = line_color # black
        if save: grid.save(save)
        return grid

    def get_cell(self, X, Y, skip_border=False):
        incr = self.grid_scale
        sx, sy = X*self.grid_scale, Y*self.grid_scale
        if skip_border:
            sx+=1
            sy+=1
            incr-=1
        cell = self.image[sx:sx + incr, sy:sy + incr, :]
        return cell #(  Pix(shape=cell.shape).fill_image(cell)   if as_pix else   cell  )

    @staticmethod
    def imshow(image): return image.swapaxes(0,1)[::-1,:,(2,1,0)]

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
