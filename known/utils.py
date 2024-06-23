__doc__="""
Utilities
"""


def ImportCustomModule(python_file:str, python_object:str, do_initialize:bool):
    r""" Import a custom module from a python file and optionally initialize it """
    import os, importlib.util
    cpath = os.path.abspath(python_file)
    failed=""
    if os.path.isfile(cpath): 
        try: 
            # from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
            cspec = importlib.util.spec_from_file_location("", cpath)
            cmodule = importlib.util.module_from_spec(cspec)
            cspec.loader.exec_module(cmodule)
            success=True
        except: success=False #exit(f'[!] Could import user-module "{cpath}"')
        if success: 
            if python_object:
                try:
                    cmodule = getattr(cmodule, python_object)
                    if do_initialize:  cmodule = cmodule()
                except:         cmodule, failed = None, f'[!] Could not import object {python_object} from module "{cpath}"'
        else:                   cmodule, failed = None, f'[!] Could not import module "{cpath}"'
    else:                       cmodule, failed = None, f"[!] File Not found @ {cpath}"
    return cmodule, failed

def GraphFromImage(img_path:str, pixel_choice:str='first', dtype=None):
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
    import cv2
    import numpy as np
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


    