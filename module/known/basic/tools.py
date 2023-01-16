__doc__=r"""
===========
Tools
===========

Contains some tools

"""

__all__=[
    'graphfromimage',
]

import numpy as np

def graphfromimage(img_path, pixel_choice='first', dtype=None):
    r""" pixel_choice = [ 'first', 'last', 'mid', 'mean' ] """
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

