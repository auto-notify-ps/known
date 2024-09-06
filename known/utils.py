__doc__=""" Helper Functions and Utils """

import os 

def ParseLinuxFiles(F, check=False): # parses --files="%F"
    Fl = [fi.strip() for fi in F.split("'/")]
    Fr = [os.path.abspath(f'/{fl[:-1]}'.replace("'\\''","'")) for fl in Fl if fl] 
    if check: Fr = [fr for fr in Fr if os.path.exists(fr)]
    return Fr

def ConfigParser(dict_to_object=True):
    r""" A replacement for usual argparse method of passing arguments to python scripts

    The problems are:
        1. A large number of arguments makes it messy to run everytime
        2. Arguments can be passed as premitive types only - we want to use any python object like ndarray, tensors and dataframes too

    The soultion is - use config files instead of arguments
        > A config file is a python scripts that contains config objects
        > This config file can be passed to a script like `python script.py --config=/path/to/config`
        > There may be multiple configs inside a single config file, hence the specific config to choose must be provided by `--member`
        > Since its a python script that will be imported on call so that we can create objects dynamically


    Example usage:

    [1] In the script file (`/path/to/script.py`) that has to be run, enter the following code

    ```python

        from known.utils import ConfigParser
        args = ConfigParser() # these are the args

    ```

    [2] In the config file ('/path/to/config.py'), enter the following code

    ```python

        import numpy as np
        import os, datetime

        class MyArgs:
            # -------------------------------------------------------------------------------------------------
            # when defining args in a class like this, it will executed as soon as this file is imported
            # -------------------------------------------------------------------------------------------------
            path =      f'{__file__}'
            cwd =       os.getcwd()
            name =      "Monty Python"
            intarg =    19
            floatarg =  99.99
            listarg =   [1, 2.0, "3"]
            tuplearg =  (listarg, intarg, floatarg)
            dictarg =   dict(place="Earth", position=(1.09, 2.908))
            nparg =     np.arange(intarg)
            timearg =   datetime.datetime.now()
            # -------------------------------------------------------------------------------------------------

        def MyArgc():
            # -------------------------------------------------------------------------------------------------
            # when defining args in a function like this, it should not accept any arguments
            # -------------------------------------------------------------------------------------------------
            return dict (
                path =      f'{__file__}',
                cwd =       os.getcwd(),
                name =      "Monty Python",
                intarg =    19,
                floatarg =  99.99,
                listarg =   [1, 2.0, "3"],
                dictarg =   dict(place="Earth", position=(1.09, 2.908)),
                nparg =     np.arange(45),
                timearg =   datetime.datetime.now(),
            )
            # -------------------------------------------------------------------------------------------------

    ```

    [3] Call the script file with following arguments:

        [3.1] either use a class that itself is a config object (class `MyArgs` in this example)

            python /path/to/script.py --config=/path/to/config.py:MyArgs

        # note the semi-colon (:) to indicate that MyArgs is the user config object

        [3.2] or use a callable function that returns a config object or a dict (function `MyArgc` in this example)

            python /path/to/script.py --config=/path/to/config.py:MyArgc!

        # note the exclamation mark (!) to indicate that MyArgc should be called (with no args)

    [NOTE]

        # the `dict_to_object` argument is only valid when user returns a dict (instead of class or object or callable)
        # if user returns anything other than a dict, it will be passed as it is - value of `dict_to_object` will not matter
        
        # if `dict_to_object==True`,    converts a dict (returned by user) to an object (this object is an instance of class 'Fake') 
        # otherwise,                    it will pass the dict as it is, which can be accessed by keys only

        # this provides flexibility to users as they can define arguments and access them in both possible ways
        # a noteable case is when the arguments names are not valid variable names in python  
        # access the dict using `fake.__dict__` from a `fake` object
        
    """

    import os, sys, importlib, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='')
    parsed = parser.parse_args()
    

    config_string = f'{parsed.config}'
    if not config_string: raise RuntimeError(f'Configuration is missing!')
    # first check if it ends in '!'
    provided_called = config_string.endswith("!")
    if provided_called: config_string = config_string[:-1]
    if not config_string: raise RuntimeError(f'Configuration is incomplete! {config_string}')
    # now find the callable by reading backward untill ':' is encountered
    split_at = config_string.rfind(":")
    if split_at<0: raise RuntimeError(f'Configuration is incomplete - member (:) not provided! {config_string}')
    CONFIG_MEMBER = config_string[split_at+1:]
    provided_config = config_string[:split_at]

    if not CONFIG_MEMBER:                               raise NameError(f'Member name is missing!')
    if not os.path.isfile(provided_config):             raise FileNotFoundError(f'Config Module File not found at [{provided_config}]')
    if not provided_config.lower().endswith('.py'):     raise RuntimeError(f'Expecting python script (.py) but got {provided_config}')
    #-----------------------------------------------------------------------------------------
    # ==> read configurations
    #-----------------------------------------------------------------------------------------
    CONFIG_FILE_PATH = os.path.abspath(provided_config)
    CONFIG_DIR_PATH, CONFIG_MODULE_NAME = os.path.dirname(CONFIG_FILE_PATH), os.path.basename(CONFIG_FILE_PATH)[:-3]
    if CONFIG_DIR_PATH not in sys.path: sys.path.append(CONFIG_DIR_PATH) 
    #<---- appends directory of config file to sys.path ->> this will add it to sys.path untill logout
    try:    CONFIG_MODULE = importlib.import_module(CONFIG_MODULE_NAME)
    except: raise ModuleNotFoundError(f'Could import module "{CONFIG_MODULE_NAME}" at file "{CONFIG_FILE_PATH}"')
    try:    CONFIG_OBJECT = getattr(CONFIG_MODULE, CONFIG_MEMBER)
    except: raise RuntimeError(f'Could not find {CONFIG_MODULE_NAME}.{CONFIG_MEMBER}')
    if provided_called:
        try:    CONFIG_OBJECT = CONFIG_OBJECT()
        except: raise RuntimeError(f'Error calling {CONFIG_MODULE_NAME}.{CONFIG_MEMBER} - is the config correct?')

    if dict_to_object and isinstance(CONFIG_OBJECT, dict):
        class Fake:
            def __len__(self): return len(self.__dict__)
            def __contains__(self, x): return x in self.__dict__
            def _get_kwargs(self): return self.__dict__.items() # mimic inbuilt ArgumentParser
            def _get_args(self): return self.__dict__.values()  # mimic inbuilt ArgumentParser
            def __init__(self, **kwargs):
                for k,v in kwargs.items(): setattr(self, k, v)

        try:    CONFIG_OBJECT = Fake(**CONFIG_OBJECT)
        except: raise RuntimeError(f'Could not create python-object from dict {CONFIG_MODULE_NAME}.{CONFIG_MEMBER}.{CONFIG_OBJECT} - are the arguments named properly?')

    return CONFIG_OBJECT #<---- this can return a dict as well (set the argument dict_to_object = False )

def ImportCustomModule(python_file:str, python_object:str='', do_initialize:bool=False):
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

