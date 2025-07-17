
__doc__=r"""pix module

Basic Image manupulation from the command line, 

```
python -m knwon.pix --help

options:
  -h, --help         show this help message and exit
  --action ACTION    (str) one of the static-methods inside the Actions class, 
                           can be - ['new', 'crop', 'extend', 'flip', 'rotate', 'convert', 'autoconvert']
  --args ARGS        (str) csv args accepted by the specified action - each action takes different args
  --input INPUT      (str) input image-file or a text/json-file containing multiple input image-file names
  --output OUTPUT    (str) output image-file or a text/json-file containing multiple output image-file names
  --files FILES      (str) multiple input image-file names - for custom action -- works only with --io=linux
  --io IO            (str) can be 'text' or 'json' or 'linux' - keep blank to io as 'image' - used if providing 
                           input/output file-names in a text/json file
  --verbose VERBOSE  (int) verbose level - 0 or 1

```

## Functionality

#### 1. Create a new image

* creates new images of given size and color 

* args is (int) 7-tuple (height, width, channels, blue, green, red, alpha)
    * `--args=h,w,c,b,g,r,a`

* fill-color (blue, green, red, alpha) depends on the specified number of channels
    * `--args=h,w,4,b,g,r,a`
    * `--args=h,w,3,b,g,r`
    * `--args=h,w,1,i`                  

Example - creating 32 x 64 image with 4 channels named 'new.jpg'
```
python -m known.pix --action=new --args=32,64,4,100,150,200,170 --output=new.jpg
```

#### 2. Crop an image

* crops an image using bounding box (y, x, h, w) 

* args is (int) 4-tuple (y-coord, x-coord, height, width) indicating a bounding box
    * `--args=y,x,h,w`

Example - 
```
python -m known.pix --action=crop --args=8,16,16,32 --input=new.jpg --output=cropped.jpg
```

#### 3. Extend an image

* extends an image using boundary distance 

* args is (int) 8-tuple (north, south, east, west, blue, green, red, alpha)
    * `--args=n,s,e,w,b,g,r,a`

* fill-color (blue, green, red, alpha) depends on the specified number of channels
    * `--args=n,s,e,w,b,g,r,a`
    * `--args=n,s,e,w,b,g,r`    
    * `--args=n,s,e,w,i`

Example - 
```
python -m known.pix --action=extend --args=10,5,6,12,123,123,0,100 --input=new.jpg --output=extended.jpg
```


#### 4. Flip an image

* flip an image (horizontally, vertically)

* args is (int) 2-tuple (horizontally, vertically) 

    * Flip horizontally          `--args=1,0`
    * Flip vertically            `--args=0,1`
    * Flip corners               `--args=1,1`
    * Flip nothing               `--args=0,0`

Example - 
```
python -m known.pix --action=flip --args=1,1 --input=extended.jpg --output=flipped.jpg
```

#### 5. Rotate an image

* rotate an image (clockwise or couter-clockwise)

* args is (int) 1-tuple (clockwise) 
    
    * Rotate clockwise               `--args=1`
    * Rotate counter-clockwise       `--args=0`

Example - 
```
python -m known.pix --action=rotate --args=1 --input=flipped.jpg --output=rotated.jpg
```

#### 6. Convert an image to a format

* converts between image formats *(as per output)*

* args is not used, target file type is infered from the output file extension

Example - 
```
python -m known.pix --action=convert --input=rotated.jpg --output=rotated.png
```

#### 7. Convert an image to multiple formats

* converts between image formats *(as per args)*

* args is (str) n-tuple specifying the extensions to be converted to

* output filenames are not used, the file-names are taken from input files

* the extensions are added as specified in args

* e.g., Convert png to jpg and webp      `--input=input.png --args=jpg,webp`
the output files will be `input.png` and `input.webp`

Example - 
```
python -m known.pix --action=autoconvert --input=rotated.png --args=jpeg,webp
```
"""

import os
import numpy as np
import cv2

DTYPE =             np.uint8
VALID_CHANNELS =    set([1, 3, 4])
DEFAULT_FILL =      255


# returns a new array
def create(h:int, w:int, c:int):
    # creates a new ndarray of specified height, width and channels with datatype specified in class (no of channels can be 1, 3 or 4)
    assert c in VALID_CHANNELS, f'[{c}] is not a valid number of channels, expecting {VALID_CHANNELS}'
    return np.zeros((h, w, c), dtype=DTYPE) 

def save(pix, path):  return cv2.imwrite(path, pix)

# returns a new array
def load(path): 
    img =  cv2.imread(path, cv2.IMREAD_UNCHANGED) #<--- can throw error
    if img.ndim==3:     pix = img.astype(DTYPE)
    elif img.ndim==2:   pix = np.expand_dims(img.astype(DTYPE), -1)
    else:               raise ValueError(f'expecting 2/3-D array but got {img.ndim}-D')
    return pix

# returns a view of the array
def flip(pix, horizontal=False, vertical=False): return pix[(slice(None, None, -1) if vertical else slice(None, None, None)), (slice(None, None, -1) if horizontal else slice(None, None, None)), :]

# returns a view of the array
def rotate(pix, clockwise=True):  return np.swapaxes(pix, 0, 1)[:, ::-1, :] if clockwise else np.swapaxes(pix, 0, 1)[::-1,:,:]

# in-place method
def fill(pix, i_row:int, i_col:int, n_rows:int, n_cols:int, color:tuple, channel=None): 
    # fills an area (region) of the image on all channels or a particular channel with values provided in color
    # i_row, i_col are the starting row and col
    # n_row, n_col are the number of rows and cols
    # if channel is provided, then color is assumed to be integer
    # otherwise color should be a tuple of the form (bgra in 4 channel), (bgr in 3 channel), (intensity in 1 channel)
    if channel is None: 
        ic = slice(None, None, None) 
        if not color: color = [ DEFAULT_FILL for _ in range(pix.shape[-1]) ]   
        else: color = color[:pix.shape[-1]]
    else:
        ic = channel # abs(int(channel)) % pix.shape[-1]
        if color is None: color = DEFAULT_FILL 

    if (n_rows is ...) or (n_rows is None): 
        if (n_cols is ...) or (n_cols is None):     pix[i_row:,             i_col:,             ic] =  color
        else:                                       pix[i_row:,             i_col:i_col+n_cols, ic] =  color
    else: 
        if (n_cols is ...) or (n_cols is None):     pix[i_row:i_row+n_rows, i_col:,             ic] =  color
        else:                                       pix[i_row:i_row+n_rows, i_col:i_col+n_cols, ic] =  color

# in-place method
def region_fill(pix, start_row:int, start_col:int, n_rows:int, n_cols:int, color:tuple):  fill(pix, start_row, start_col, n_rows, n_cols, color, channel=None)

# in-place method
def region_copy(pix_from, start_row_from, start_col_from, n_rows, n_cols, pix_to, start_row_to, start_col_to): pix_to[start_row_to:start_row_to+n_rows, start_col_to:start_col_to+n_cols,  :] = pix_from[start_row_from:start_row_from+n_rows, start_col_from:start_col_from+n_cols, :]

# returns a new array
def extend(pix, north, south, east, west, filler=None):
    # extends an image on all four sides (specified by number of pixels in north, south, east, west)
    new = create(pix.shape[0] + (abs(north) + abs(south)), pix.shape[1] + (abs(east) + abs(west)), pix.shape[-1])
    region_fill(new, 0, 0, None, None, filler)
    region_copy(
        pix_from = pix,   
        start_row_from = 0,
        start_col_from = 0,           
        n_rows = pix.shape[0],
        n_cols = pix.shape[1],
        pix_to = new,     
        start_row_to = abs(north),
        start_col_to = abs(west))
    return new


class Actions:
    r"""

    Defins the available actions to this module (refer __main__.py)

        python -m known.pix --action=<member-function>

    Note: 
        (blue, green, red, alpha) can take values from 0 to 255
        (channels) can be 4, 3 or 1 - the color tuple depends on the channels in the image
    """
    @staticmethod
    def new(inputs, outputs, args, verbose=0, dry=False):
        
        """ creates new images of given size and color 
        
        args is (int) 7-tuple (height, width, channels, blue, green, red, alpha)

            --args=h,w,c,b,g,r,a
        
        fill-color (blue, green, red, alpha) depends on the specified number of channels

            --args=h,w,4,b,g,r,a
            --args=h,w,3,b,g,r            
            --args=h,w,1,i                    
        """
        if verbose: print(f'⚙ [NEW] {len(outputs)}')
        try:
            args = [int(s) for s in args]
            if verbose: print(f'● Dimension {args[0:3]} with fill-color {args[3:]}')
            ip='new'
            for op in outputs:
                if verbose: print(f'\t● {ip} ⇒ {op}')
                img = create(*args[0:3])
                fill(img, 0, 0, ..., ..., color=args[3:], channel=None)
                if dry: print(f'\n[DRY ~ SAVE]\n\t\n\tIMAGE: {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')
        
    @staticmethod
    def crop(inputs, outputs, args, verbose=0, dry=False):
        """ crops an image using bounding box (y, x, h, w) 

        args is (int) 4-tuple (y-cord, x-coord, height, width) indicating a bounding box
            
            --args=y,x,h,w
        """
        if verbose: print(f'⚙ [CROP] {len(inputs)}')
        try:
            y, x, h, w = [int(s) for s in args]
            if verbose: print(f'● Bounding-Box {y=} {x=} {h=} {w=}')
            for ip,op in zip(inputs,outputs):
                if verbose: print(f'\t● {ip} ⇒ {op}')
                org = load(ip)
                img = create(h, w, org.shape[-1])
                region_copy(org, y, x, h, w, img, 0, 0)
                if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {org.shape} --> {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')

    @staticmethod
    def extend(inputs, outputs, args, verbose=0, dry=False):
        """ extends an image using boundary distance 

        args is (int) 8-tuple (north, south, east, west, blue, green, red, alpha)
            
            --args=n,s,e,w,b,g,r,a

        fill-color (blue, green, red, alpha) depends on the specified number of channels

            --args=n,s,e,w,b,g,r,a
            --args=n,s,e,w,b,g,r    
            --args=n,s,e,w,i
        """
        if verbose: print(f'⚙ [EXTEND] {len(inputs)}')
        try:
            args = [int(s) for s in args]
            north, south, east, west = args[0:4]
            if verbose: print(f'● Directions {north=} {south=} {east=} {west=}')
            for ip,op in zip(inputs,outputs):
                if verbose: print(f'\t● {ip} ⇒ {op}')
                img = extend(load(ip), north, south, east, west, filler=args[4:])
                if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')

    @staticmethod
    def flip(inputs, outputs, args, verbose=0, dry=False):
        """ flip an image (horizontally, vertically)

        args is (int) 2-tuple (horizontally, vertically) 
        
            Flip horizontally          --args=1,0
            Flip vertically            --args=0,1
            Flip corners               --args=1,1
            Flip nothing               --args=0,0
        """
        if verbose: print(f'⚙ [FLIP] {len(inputs)}')
        try:
            h, v = [bool(int(s)) for s in args]
            if verbose: print(f'● Directions {h=} {v=}')
            for ip,op in zip(inputs,outputs):
                if verbose: print(f'\t● {ip} ⇒ {op}')
                org = load(ip)
                img = flip(org, horizontal=h, vertical=v)
                if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {org.shape} --> {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')

    @staticmethod
    def rotate(inputs, outputs, args, verbose=0, dry=False):
        """ rotate an image (clockwise or couter-clockwise)

        args is (int) 1-tuple (clockwise) 
            
            Rotate clockwise               --args=1
            Rotate counter-clockwise       --args=0
        """
        if verbose: print(f'⚙ [ROTATE] {len(inputs)}')
        try:
            c = [bool(int(s)) for s in args][0]
            if verbose: print(f'● Direction clockwise - {c}')
            for ip,op in zip(inputs,outputs):
                if verbose: print(f'\t● {ip} ⇒ {op}')
                org = load(ip)
                img = rotate(org, clockwise=c)
                if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {org.shape} --> {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')
    
    @staticmethod
    def convert(inputs, outputs, args, verbose=0, dry=False):
        """ converts an image (as per output)
        
        args is not used, target file type is infered from the output file-name (extension)
            
        e.g., Convert png to jpg      --input=input.png --output=output.jpg
        """
        
        if verbose: print(f'⚙ [CONVERT] {len(inputs)}')
        try:
            for ip,op in zip(inputs,outputs): 
                if verbose: print(f'\t● {ip} ⇒ {op}')
                img = load(ip)
                if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {img.shape}\n\tOUTPUT: {op}\n')
                else: save(img, op)
            if verbose: print(f'✓ Success!')
        except:
            if verbose: print(f'✗ Failed!')

    @staticmethod
    def autoconvert(inputs, outputs, args, verbose=0, dry=False):
        """ converts an image (as per args)
        
        args is (str) n-tuple specifying the extensions to be converted to
        output filenames are not used, the file-names are taken from input files
        the extensions are added as specified in args

        e.g., Convert png to jpg and webp      --input=input.png --args=jpg,webp
        the output files will be `input.jpg` and `input.webp`

        """
        if verbose: print(f'⚙ [AUTOCONVERT] {len(inputs)}')
        try:
            for ip in inputs: 
                dirname, filename = os.path.dirname(ip), os.path.basename(ip)
                i = filename.rfind('.')
                name = filename if i<0 else filename[0:i]
                for ext in args:
                    op = os.path.join(dirname, f'{name}.{ext}')
                    if verbose: print(f'\t● {ip} ⇒ {op}')
                    img = load(ip)
                    if dry: print(f'\n[DRY ~ SAVE]\n\tINPUT: {ip}\n\tIMAGE: {img.shape}\n\tOUTPUT: {op}\n')
                    else: save(img, op)
            if verbose: print(f'✓ Success!')
            return True
        except:
            if verbose: print(f'✗ Failed!')
            return False

    @staticmethod
    def multiconvert(inputs, outputs, args, verbose=0, dry=False):
        """ converts all the images in a directory (as per args)
        
        input is a directory name
        args is csv source extension and target extension ending with .

        e.g., Convert png and bmp to jpg and webp      --input=/home/user --args=.,png,bmp,jpg.,webp.

        """
        args = set(args)
        remove_after_convert = False
        if "." in args: 
            remove_after_convert=True
            args.remove(".")
        tgt_ext = tuple(set([e[:-1] for e in args if e.endswith('.')]))
        src_ext = tuple(set([e      for e in args if not e.endswith('.')]))
        
        if verbose: print(f'⚙ [MULTICONVERT] {inputs}')
        for in_dir in inputs:
            input_files = [os.path.join(in_dir, f) for f in os.listdir(in_dir) if f'{f}'.lower().endswith(src_ext)]
            res = __class__.autoconvert(
                inputs= input_files,
                outputs=None,
                args=tgt_ext,
                verbose=verbose,
                dry=dry,
            )
            if res:
                if verbose: print(f'✓ [MULTICONVERT] Success for {in_dir}')
                if remove_after_convert: 
                    if verbose: print(f'\t ~ Deleting {len(input_files)} Files...')
                    if dry:
                        for f in input_files: print(f'Delete file {f}')
                    else: 
                        for f in input_files: os.remove(f)
            else:
                if verbose: print(f'✗ [MULTICONVERT] Failed for {in_dir}')
