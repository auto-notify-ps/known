#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/basic.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__all__ = [ 'Fake', 'Infinity', 'Kio', 'Remap',  'BaseConvert', 'IndexedDict', 'Fuzz', 'Mailer', 'Verbose', 'Symbols', 'Table']
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
from typing import Any, Union, Iterable #, Callable #, BinaryIO, cast, Dict, Optional, Type, Tuple, IO
import os, platform, datetime, smtplib, mimetypes
from math import floor, log, ceil
from zipfile import ZipFile
from email.message import EmailMessage
from collections import UserDict
from io import BytesIO

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Fake: # a fake object
    def __init__(self, **attributes):
        for name,value in attributes.items(): setattr(self, name, value)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Infinity: 
    # emulates infinity with comparision operators < <=, >, >=, ==
    #   +inf    =  Infinity(1)
    #   -inf    =  Infinity(-1)
    # emulates Empty and Universal sets with 'in' keyword
    #   Universal Set (everything) = Infinity(1)
    #   Null/Empty Set (nothing) = Infinity(-1)

    def __init__(self, sign=1) -> None: self.sign = (sign>=0) # has positive
    
    def __gt__(self, other): return self.sign       # +inf is greater than everything / -inf is greater than nothing
    def __ge__(self, other): return self.sign       

    def __lt__(self, other): return not self.sign   # -inf is less than everything    / +inf is less than nothing
    def __le__(self, other): return not self.sign

    def __eq__(self, other): return False           # inf is not equal to anything, not even itself

    def __contains__(self, x): return self.sign     # universal set contains everything (always true), empty set contains nothing (always false)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Kio:
    r""" provides input/out methods for loading saving python objects using json and pickle """

    import json, pickle
    IOFLAG = dict(
        json=   (json,      ''     ), 
        pickle= (pickle,    'b'    ),
        )

    @staticmethod
    def get_ioas(): return list(__class__.IOFLAG.keys())

    @staticmethod
    def save_buffer(o:Any, ioas:str, seek0=False) -> None:
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        buffer = BytesIO()
        s_module.dump(o, buffer)
        if seek0: buffer.seek(0) # prepares for reading
        return buffer

    @staticmethod
    def load_buffer(buffer:BytesIO, ioas:str, seek0=True): 
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        if seek0: buffer.seek(0) # prepares for reading
        return s_module.load(buffer)

    @staticmethod
    def save_file(o:Any, path:str, ioas:str, **kwargs):
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        with open(path, f'w{s_flag}') as f: s_module.dump(o, f, **kwargs)
        return path
    @staticmethod
    def load_file(path:str, ioas:str):
        assert ioas in __class__.IOFLAG, f'key error {ioas}'
        s_module, s_flag = __class__.IOFLAG[ioas]
        with open(path, f'r{s_flag}') as f: o = s_module.load(f)
        return o
    
    @staticmethod
    def save_as_json(o:Any, path:str, **kwargs):    return __class__.save_file(o, path, 'json', **kwargs)
    @staticmethod
    def load_as_json(path:str):                     return __class__.load_file(path, 'json')

    @staticmethod
    def save_as_pickle(o:Any, path:str, **kwargs):  return __class__.save_file(o, path, 'pickle', **kwargs)
    @staticmethod
    def load_as_pickle(path:str):                   return __class__.load_file(path, 'pickle')

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Remap:
    r""" 
    Provides a mapping between ranges, works with scalars, ndarrays and tensors.

    :param Input_Range:     *FROM* range for ``forward`` call, *TO* range for ``backward`` call
    :param Output_Range:    *TO* range for ``forward`` call, *FROM* range for ``forward`` call
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

    def backward(self, X):
        r""" maps ``X`` from ``Output_Range`` to ``Input_Range`` """
        return ((X - self.output_low)*self.input_delta/self.output_delta) + self.input_low

    def forward(self, X):
        r""" maps ``X`` from ``Input_Range`` to ``Output_Range`` """
        return ((X - self.input_low)*self.output_delta/self.input_delta) + self.output_low

    def __call__(self, X, backward=False):
        return self.backward(X) if backward else self.forward(X)
    
    def swap_range(self):
        Input_Range, Output_Range = (self.output_low, self.output_high), (self.input_low, self.input_high)
        self.set_input_range(Input_Range)
        self.set_output_range(Output_Range)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class BaseConvert:
    
    r""" Number System Conversion 
    
    A number is abstract concept that has many representations using sets of symbols

    A base-n number system uses a set of n digits to represent any number
    This is called the representation of the number

    Given one representation, we only need to convert to another

    """

    @staticmethod
    def zeros(n): return [0 for _ in range(n)]

    @staticmethod
    def convert(digits, base_from, base_to, reversed=True):
        r""" convers from one base to another 
        
        :param digits:      iterable of digits in base ```base_from```. NOTE: digits are Natural Numbers starting at 0. base 'b' will have digits between [0, b-1]
        :param base_from:   int - the base to convert from
        :param base_to:     int - the base to convert to
        :param reversed:    bool - if True, digits are assumed in reverse (human readable left to right)
                            e.g. if reversed is True then binary digits iterable [1,0,0] will represent [4] in decimal otherwise it will represent [1] in decimal
        """

        digits_from =  [int(abs(d)) for d in digits] # convert to int data-type
        if reversed: digits_from = digits_from[::-1]
        ndigits_from = len(digits_from)
        mult_from = [base_from**i for i in range(ndigits_from)]
        repr_from = sum([ui*vi for ui,vi in zip(digits_from,mult_from, strict=True)]) #dot(digits_from , mult_from)

        #ndc = base_from**ndigits_from
        ndigits_to = ceil(log(repr_from,base_to))
        digits_to =  __class__.zeros(ndigits_to) 
        n = int(repr_from)
        for d in range(ndigits_to):
            digits_to[d] = n%base_to
            n=n//base_to

        if reversed: digits_to = digits_to[::-1]
        return tuple(digits_to)


    @staticmethod
    def ndigits(num:int, base:int): return ceil(log(num,base))

    @staticmethod
    def int2base(num:int, base:int, digs:int) -> list:
        r""" 
        Convert base-10 integer to a base-n list of fixed no. of digits 

        :param num:     base-10 number to be represented
        :param base:    base-n number system
        :param digs:    no of digits in the output

        :returns:       represented number as a list of ordinals in base-n number system

        .. seealso::
            :func:`~known.basic.base2int`
        """
        
        ndigits = digs if digs else ceil(log(num,base)) 
        digits =  __class__.zeros(ndigits)
        n = num
        for d in range(ndigits):
            digits[d] = n%base
            n=n//base
        return digits

    @staticmethod
    def base2int(num:Iterable, base:int) -> int:
        """ 
        Convert an iterbale of digits in base-n system to base-10 integer

        :param num:     iterable of base-n digits
        :param base:    base-n number system

        :returns:       represented number as a integer in base-10 number system

        .. seealso::
            :func:`~known.basic.int2base`
        """
        res = 0
        for i,n in enumerate(num): res+=(base**i)*n
        return int(res)


    SYM_BIN = { f'{i}':i for i in range(2) }
    SYM_OCT = { f'{i}':i for i in range(8) }
    SYM_DEC = { f'{i}':i for i in range(10) }
    SYM_HEX = {**SYM_DEC , **{ s:(i+10) for i,s in enumerate(('A', 'B', 'C', 'D', 'E', 'F'))}}
    
    @staticmethod
    def n_syms(n): return { f'{i}':i for i in range(n) }

    @staticmethod
    def to_base_10(syms:dict, num:str):
        b = len(syms)
        l = [ syms[n] for n in num[::-1] ]
        return __class__.base2int(l, b)

    @staticmethod
    def from_base_10(syms:dict, num:int, joiner='', ndigs=None):
        base = len(syms)
        #print(f'----{num=} {type(num)}, {base=}, {type(base)}')
        if not ndigs: ndigs = (1 + (0 if num==0 else floor(log(num, base))))  # __class__.ndigs(num, base)
        ss = tuple(syms.keys())
        S = [ ss[i]  for i in __class__.int2base(num, base, ndigs) ]
        return joiner.join(S[::-1])


    @staticmethod
    def int2hex(num:int, joiner=''): return __class__.from_base_10(__class__.SYM_HEX, num, joiner)
  
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class IndexedDict(UserDict):
    r""" Implements an Indexed dict where values can be addressed using both index(int) and keys(str) """

    def __init__(self, **members) -> None:
        self.names = []
        super().__init__(*[], **members)
    
    def keys(self): return enumerate(self.names, 0) # for i,k in self.keys()

    def items(self): return enumerate(self.data.items(), 0) # for i,(k,v) in self.items()

    def __len__(self): return len(self.data)

    def __getitem__(self, name): 
        if isinstance(name, int): name = self.names[name]
        if name in self.data: 
            return self.data[name]
        else:
            raise KeyError(name)

    def __setitem__(self, name, item): 
        if isinstance(name, int): name = self.names[name]
        if name not in self.data: self.names.append(name)
        self.data[name] = item

    def __delitem__(self, name): 
        index = None
        if isinstance(name, int):  
            index = name
            name = self.names[name]
        if name in self.data: 
            del self.names[self.names.index(name) if index is None else index]
            del self.data[name]

    def __iter__(self): return iter(self.names)

    def __contains__(self, name): return name in self.data

    # Now, add the methods in dicts but not in MutableMapping

    def __repr__(self) -> str:
        return f'{__class__} :: {len(self)} Members'
    
    def __str__(self) -> str:
        items = ''
        for i,k in enumerate(self):
            items += f'[{i}] \t {k} : {self[i]}\n'
        return f'{__class__} :: {len(self)} Members\n{items}'
    
    def __copy__(self):
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        inst.__dict__["names"] = self.__dict__["names"].copy()
        return inst

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Fuzz:
    r""" file system and zipping """

    @staticmethod
    def SplitFileName(f:str):
        r"""splits a file-name into name.ext 
        Note: make sure to pass os.path.basename() to this function
        Retutns: 2-tuple (name, ext)
            `name` is always a string
            `ext` can be None or a string (None means that there is no "." in file name)
        """
        i = f.rfind('.')
        return (f, None) if i<0 else (f[0:i], f[i+1:])

    @staticmethod
    def ExistInfo(path):
        """ returns 4-tuple (exists,isdir,isfile,islink) """
        return os.path.exists(path), os.path.isdir(path), os.path.isfile(path), os.path.islink(path)

    
    @staticmethod
    def PathInfo(path):
        """ returns information about file-name, path and its types """
        abspath = os.path.abspath(path)
        dirname, filename = os.path.dirname(abspath), os.path.basename(abspath)
        exists,isdir,isfile,islink = __class__.ExistInfo(abspath)
        name, ext = __class__.SplitFileName(f'{filename}')
        return dict(
            abspath=abspath, 
            filename=filename, 
            dirname=dirname, 
            exists=exists, 
            isdir=isdir, 
            isfile=isfile, 
            islink=islink, 
            name=name, ext=ext)
    
    @staticmethod
    def RenameFile(path, new_name, keep_ext=False):
        """ rename a file with or without changing its extension """
        dirname, filename = os.path.dirname(path), os.path.basename(path)
        _, ext = __class__.SplitFileName(f'{filename}')
        if keep_ext and (ext is not None):  name_ = f'{new_name}.{ext}'
        else:                               name_ = f'{new_name}'
        return os.path.join(dirname, name_)
    
    @staticmethod
    def RenameFileExt(path, new_ext):
        """ change a file extension without renaming it """
        dirname, filename = os.path.dirname(path), os.path.basename(path)
        name, _ = __class__.SplitFileName(f'{filename}')
        return os.path.join(dirname, f'{name}.{new_ext}')

    


    @staticmethod
    def ZipFiles(files:list, zip_path:str=None, **kwargs):
        r""" zips all (only files) in the list of file paths and saves at 'zip_path' """
        if isinstance(files, str): files= [f'{files}']
        if not files: return # no files provided to zip
        if not zip_path : zip_path = f'{files[0]}.zip' # if zip path not provided take it to be the first file
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  # append .zip to the end of path

        zipped = 0
        with ZipFile(zip_path, 'w', **kwargs) as zip_object:
            for path in files:
                if not os.path.isfile(path): continue
                zip_object.write(f'{path}')
                zipped+=1
        return zipped, zip_path

    @staticmethod
    def ZipFolders(folders:Union[str,list], zip_path:str=None, **kwargs):  
        r""" zip multiple folders into a single zip file 
        to zip a single folder with the same zip name - provide folder as a string and keep zip_path as none
        """    
        if isinstance(folders, str): folders= [f'{folders}']
        if not folders: return None, None# no folders provided to zip
        if not zip_path : zip_path = f'{folders[0]}.zip' # if zip path not provided take it to be the first folder
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  # append .zip to the end of path
        all_files = []
        for folder in folders:
            for root, directories, files in os.walk(folder): all_files.extend([os.path.join(root, filename) for filename in files])
        return __class__.ZipFiles(all_files, f'{zip_path}', **kwargs)
    
    @staticmethod
    def GetAllFilesInfo(directory):
        r""" recursively list all files in a folder along with size and extension 

        eg - path is /home/user/name.ext
        returns list of 6-tuples (
            file-name,                      # name.ext
            file-name-without-extension,    # name
            file-extension,                 # .ext          --> note it includes '.'
            file-dir,                       # /home/user
            file-path,                      # /home/user/name.ext
            file-size,                      # size in bytes
        ) 
            
        """
        file_paths = []
        for root, _, files in os.walk(directory):
            for file_name in files:
                # join the two strings in order to form the full filepath.
                i = file_name.rfind('.')
                if i<0:     file_name_we, file_ext = file_name, '' # extension = '' means "." was not in filename at all
                else:       file_name_we, file_ext = file_name[0:i], file_name[i:] # includes .
                file_path = os.path.abspath(os.path.join(root, file_name))
                #file_paths.append((file_name, file_name_we, file_ext, root, file_path, file_abs, os.stat(file_path).st_size))
                # eg - path is /home/user/name.ext
                file_paths.append((
                        file_name,                      # name.ext
                        file_name_we,                   # name
                        file_ext,                       # .ext          --> note it includes '.'
                        os.path.dirname(file_path),     # /home/user
                        file_path,                      # /home/user/name.ext
                        os.stat(file_path).st_size,     # size in bytes
                    ))
        return file_paths 

    @staticmethod
    def MatchFiles(directory, policy):
        """ Matches all files (recursive) in a directory against a policy

        Policy:
            # policy should be function that takes 6-tuples retuned by `GetAllFilesInfo` method - fn, fnwe, fe, fd, fp, fs 
            # and returns if this file matches that policy - return bool - match or not
            # policy = lambda fn, fnwe, fe, fd, fp, fs : bool

        Example usage:

        -> get all png and jpg files in /home/user folder that are larger than 15KB

        matched_files = MatchFiles(
                            directory = '/home/user', 
                            policy = lambda fn, fnwe, fe, fd, fp, fs: ((fe.lower() in set(['.jpg', '.png'])) and (fs > 15*1024)),
                        )
        # or
        matched_files = MatchFiles(
                            directory = '/home/user', 
                            policy = lambda *args: ((args[2].lower() in set(['.jpg', '.png'])) and (args[-1] > 15*1024)),
                        )
        """
        allfiles = __class__.GetAllFilesInfo(directory) # can del this later
        return [allfiles[i] for i,(fn, fnwe, fe, fd, fp, fs) in enumerate(allfiles) if policy(fn, fnwe, fe, fd, fp, fs)]


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Mailer:
    r""" Use a g-mail account to send mail. 

    .. warning:: security concern
        You should enable 2-factor-auth in gmail and generate an app-password instead of using your gmail password.
        Visit (https://myaccount.google.com/apppasswords) to generate app-password.
        Usually, these type of emails are treated as spam by google, so they must be marked 'not spam' at least once.
        It is recomended to create a seperate gmail account for sending mails.

    Usage: 
    - use the `Mailer.send` method to send emails
            username:str        the username that is send the email, 
            password:str        the password of sender, 
            subject:str         the subject line, 
            rx:str              reciver email addresses - comma seperated, 
            cc:str              carbon copy email addresses - comma seperated,
            bcc:str             blind carbon copy email addresses - comma seperated, 
            content:str         the content - body of email      
            signature:str       the signature to put at the bottom, 
            attached:list       a list of files to be attached =None by default - a blank list, 
            url:str             the smtp server url (=None by default) - takes from class variable `DEFAULT_URL`, 
            port:Union[int,str] the smtp server port (=None by default) - takes from class variable `DEFAULT_PORT`, 
            tls:bool            a boolean value indicating if `starttls()` should be called on `smpt` (=None by default) - takes from class variable `DEFAULT_TLS`, 
            verbose             (=True by default)

    - otherwise use `compose_mail` to return an `EmailMessage` object and use `send_mail` to send it later
        
    Changes: 

        version 0.0.20
        - moved zipping functionality inside the class, can now be used as a stand alone class
        - removed callable login() function, instead directly requires username and password as strings
    """
    
    DEFAULT_CTYPE =     'application/octet-stream'  
    DEFAULT_URL =       'smtp.gmail.com'
    DEFAULT_PORT =      '587'
    DEFAULT_TLS =       True

    @staticmethod
    def global_alias(prefix=''): return f'{prefix}{os.getlogin()} @ {platform.node()}:{platform.system()}.{platform.release()}'

    @staticmethod # helper method
    def _get_files(directory):
        r""" recursively list all files in a folder """
        file_paths = []
        for root, directories, files in os.walk(directory): file_paths.extend([os.path.join(root, filename) for filename in files])
        return file_paths   

    @staticmethod # helper method
    def _zip_files(zip_path:str, files, **kwargs):
        r""" zips all (only files) in the list of file paths and saves at 'zip_path' """
        zipped = 0
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'
        with ZipFile(zip_path, 'w', **kwargs) as zip_object:
            for path in files:
                if not os.path.isfile(path): continue
                zip_object.write(f'{path}')
                zipped+=1
        return zipped, zip_path
        
    @staticmethod # helper method
    def _get_mimes(files):
        r""" gets mimetype info all files in a list """
        if isinstance(files, str): files=[f'{files}']
        res = []
        for path in files:
            if not os.path.isfile(path): continue
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None: ctype = __class__.DEFAULT_CTYPE  
            maintype, subtype = ctype.split('/', 1)
            res.append( (path, maintype, subtype) )
        return res

    @staticmethod 
    def compose_mail(subject:str, rx:str, cc:str, bcc:str, content:str, signature:str, attached:list=None, verbose=True):
        r""" compose an e-mail msg to send later
        
        :param subject:     subject
        :param rx:          csv recipent email address
        :param cc:          csv cc email address
        :param bcc:         csv bcc email address
        :param content:     main content
        :param attached:    list of attached files - is a 2-tuple (attachment_type:str, files:tuple )

        # attach all files in the list ::   ('',                ('file1.xyz', 'file2.xyz'))
        # zip all the files in the list ::  ('zipname.zip',     ('file1.xyz', 'file2.xyz'))
        """
        
        msg = EmailMessage()

        # set subject
        msg['Subject'] = f'{subject}'
        if verbose: print(f'SUBJECT: {subject}')

        # set to
        msg['To'] = rx
        if verbose: print(f'TO: {rx}')

        if cc: msg['Cc'] = cc
        if verbose: print(f'CC: {cc}')

        if bcc: msg['Bcc'] = bcc
        if verbose: print(f'BCC: {bcc}')

        # set content
        body = content + signature
        msg.set_content(body)
        if verbose: print(f'MESSAGE: #[{len(body)}] chars.')

        default_attached = []

        attached = [] if attached is None else attached
        assert isinstance(attached, (list, tuple)), f'Expecting a list or tuple of attached files but got {type(attached)}'
        for (attach_type, attach_args) in attached:
            if verbose: print(f' ... processing attachement :: {attach_type} :: {attach_args}')

            all_files = []
            for path in attach_args:
                if os.path.isdir(path):     all_files.extend(__class__._get_files(path))
                elif os.path.isfile(path):  all_files.append(path)
                else:
                    if verbose: print(f'[!] Invalid Path :: {path}, skipped...')

            if not attach_type:  default_attached.extend(__class__._get_mimes(all_files)) # attach individually
            else: # make zip
                zipped, zip_path=__class__._zip_files(attach_type, all_files)
                if verbose: print(f'\t --> zipped {zipped} items @ {zip_path} ')
                if zipped>0: default_attached.extend(__class__._get_mimes(zip_path))
                else:
                    if verbose: print(f'[!] [{zip_path}] is empty, will not be attched!' )
                    try:
                        os.remove(zip_path)
                        if verbose: print(f'[!] [{zip_path}] was removed.' )
                    except:
                        if verbose: print(f'[!] [{zip_path}] could not be removed.' ) 
                

        # set attached ( name, main_type, sub_type), if sub_type is none, auto-infers using imghdr
        for file_name,main_type,sub_type in default_attached:
            if verbose: print(f'[+] Attaching file [{main_type}/{sub_type}] :: [{file_name}]')
            with open (file_name, 'rb') as f: file_data = f.read()
            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def send_mail(username:str, password:str, msg:EmailMessage, url:str, port:Union[int,str], tls:bool, verbose=True):
        r""" send a msg using url:port with provided credentials, calls `starttls` is `tls` is True """
        if verbose: print(f'[*] Sending Email from {username}')
        msg['From'] = f'{username}' # set from
        if url is None:     url =   __class__.DEFAULT_URL
        if port is None:    port =  __class__.DEFAULT_PORT
        if tls is None:     tls =    __class__.DEFAULT_TLS
        if verbose: print(f'[~] using smtp server: {url}:{port}/{tls}')
        with smtplib.SMTP(f'{url}', int(port)) as smpt: 
            if tls: smpt.starttls()
            smpt.login(username, password) 
            smpt.ehlo()
            smpt.send_message(msg)
        if verbose: print(f'[*] Sent!')

    @staticmethod
    def send(
            username:str, 
            password:str, 
            subject:str, 
            rx:str, 
            cc:str, 
            bcc:str, 
            content:str, 
            signature:str, 
            attached:list       =None, 
            url:str             =None, 
            port:Union[int,str] =None, 
            tls:bool            =None, 
            verbose             =True
        ): __class__.send_mail(username, password, __class__.compose_mail(subject, rx, cc, bcc, content, signature, attached, verbose), url, port, tls, verbose)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Verbose:
    r""" Contains shorthand helper functions for printing outputs and representing objects as strings."""
    
    #-----------------------------------------------------------------------
    """ SECTION: HRS - Human Readable String  for sizes"""
    #-----------------------------------------------------------------------
    HRS_MAPPER = dict(BB=2**0, KB=2**10, MB=2**20, GB=2**30, TB=2**40) # 2 chars for keys

    @staticmethod
    def hrs2bytes(hrsize:str): return round(float(hrsize[:-2])*__class__.HRS_MAPPER.get(hrsize[-2:].upper(), 0))
    @staticmethod
    def bytes2hrs(size:int, unit:str, roundoff=2): return f"{round(size/(__class__.HRS_MAPPER[unit]),roundoff)}{unit}"

    @staticmethod
    def bytes2bb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'BB', roundoff)
    @staticmethod
    def bytes2kb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'KB', roundoff)
    @staticmethod
    def bytes2mb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'MB', roundoff)
    @staticmethod
    def bytes2gb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'GB', roundoff)
    @staticmethod
    def bytes2tb(size:int, roundoff=2): return __class__.bytes2hrs(size, 'TB', roundoff)

    @staticmethod
    def bytes2auto(size:int, roundoff=2):
        if      size<__class__.HRS_MAPPER["KB"]: return __class__.bytes2bb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["MB"]: return __class__.bytes2kb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["GB"]: return __class__.bytes2mb(size, roundoff)
        elif    size<__class__.HRS_MAPPER["TB"]: return __class__.bytes2gb(size, roundoff)
        else                                   : return __class__.bytes2tb(size, roundoff)
    #-----------------------------------------------------------------------


    #-----------------------------------------------------------------------
    """ SECTION: StrX - human readable string representation of objects """
    #-----------------------------------------------------------------------

    DEFAULT_DATE_FORMAT = ["%Y","%m","%d","%H","%M","%S","%f"] #  Default date format for :func:`~known.basic.Verbose.strU` 
    DASHED_LINE = "=-=-=-=-==-=-=-=-="

    @staticmethod
    def strN(s:str, n:int) -> str:  
        r""" Repeates a string n-times """
        return ''.join([s for _ in range(n)])

    @staticmethod
    def _recP_(a, level, index, pindex, tabchar='\t', show_dim=False):
        # helper function for recP - do not use directly
        if index<0: index=''
        dimstr = ('* ' if level<1 else f'*{level-1} ') if show_dim else ''
        pindex = f'{pindex}{index}'
        if len(a.shape)==0:
            print(f'{__class__.strN(tabchar, level)}[ {dimstr}@{pindex}\t {a} ]') 
        else:
            print(f'{__class__.strN(tabchar, level)}[ {dimstr}@{pindex} #{a.shape[0]}')
            for i,s in enumerate(a):
                __class__._recP_(s, level+1, i, pindex, tabchar, show_dim)
            print(f'{__class__.strN(tabchar, level)}]')

    @staticmethod
    def recP(arr:Iterable, show_dim:bool=False) -> None: 
        r"""
        Recursive Print - print an iterable recursively with added indentation.

        :param arr:         any iterable with ``shape`` property.
        :param show_dim:    if `True`, prints the dimension at the start of each item
        """
        __class__._recP_(arr, 0, -1, '', '\t', show_dim)
    
    @staticmethod
    def strA_(arr:Iterable, start:str="", sep:str="|", end:str="") -> str:
        r"""
        String Array - returns a string representation of an iterable for printing.
        
        :param arr:     input iterable
        :param start:   string prefix
        :param sep:     item seperator
        :param end:     string postfix
        """
        res=start
        for a in arr: res += (str(a) + sep)
        return res + end

    @staticmethod
    def strA(arr:Iterable, start:str="", sep:str="|", end:str="") -> None: print(__class__.strA_(arr, start, sep, end))
    
    @staticmethod
    def strD_(arr:Iterable, sep:str="\n", cep:str=":\n", caption:str="") -> str:
        r"""
        String Dict - returns a string representation of a dict object for printing.
        
        :param arr:     input dict
        :param sep:     item seperator
        :param cep:     key-value seperator
        :param caption: heading at the top
        """
        res=f"=-=-=-=-==-=-=-=-={sep}DICT #[{len(arr)}] : {caption}{sep}{__class__.DASHED_LINE}{sep}"
        for k,v in arr.items(): res+=str(k) + cep + str(v) + sep
        return f"{res}{__class__.DASHED_LINE}{sep}"

    @staticmethod
    def strD(arr:Iterable, sep:str="\n", cep:str=":\n", caption:str="") -> None: print(__class__.strD_(arr, sep, cep, caption))

    @staticmethod
    def strU(form:Union[None, Iterable[str]], start:str='', sep:str='', end:str='') -> str:
        r""" 
        String UID - returns a formated string of current timestamp.

        :param form: the format of timestamp, If `None`, uses the default :data:`~known.basic.Verbose.DEFAULT_DATE_FORMAT`.
            Can be selected from a sub-set of ``["%Y","%m","%d","%H","%M","%S","%f"]``.
            
        :param start: UID prefix
        :param sep: UID seperator
        :param end: UID postfix

        .. seealso::
            :func:`~known.basic.uid`
        """
        if not form: form = __class__.DEFAULT_DATE_FORMAT
        return start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end

    @staticmethod
    def now(year:bool=True, month:bool=True, day:bool=True, 
            hour:bool=True, minute:bool=True, second:bool=True, mirco:bool=True, 
            start:str='', sep:str='', end:str='') -> str:
        r""" Unique Identifier - useful in generating unique identifiers based on current timestamp. 
        Helpful in generating unique filenames based on timestamps. 
        
        .. seealso::
            :func:`~known.basic.Verbose.strU`
        """
        form = []
        if year:    form.append("%Y")
        if month:   form.append("%m")
        if day:     form.append("%d")
        if hour:    form.append("%H")
        if minute:  form.append("%M")
        if second:  form.append("%S")
        if mirco:   form.append("%f")
        assert (form), 'format should not be empty!'
        return (start + datetime.datetime.strftime(datetime.datetime.now(), sep.join(form)) + end)


    #-----------------------------------------------------------------------
    """ SECTION: show/info - human readable information about pbjects """
    #-----------------------------------------------------------------------

    DOCSTR_FORM = lambda x: f'\t!docstr:\n! - - - - - - - - - - - - - - - - -\n{x}\n- - - - - - - - - - - - - - - - - !'

    @staticmethod
    def show_(x:Any, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
        res = ""
        for d in dir(x):
            if not (d.startswith(sw) or d.endswith(ew)):
                v = ""
                try:
                    v = getattr(x, d)
                except:
                    v='?'
                res+=f'({d} {cep} {v}{sep}'
        return res

    @staticmethod
    def show(x:Any, cep:str='\t\t:', sep="\n", sw:str='__', ew:str='__') -> None:
        r"""
        Show Object - describes members of an object using the ``dir`` call.

        :param x:       the object to be described
        :param cep:     the name-value seperator
        :param sw:      argument for ``startswith`` to check in member name
        :param ew:      argument for ``endswith`` to check in member name

        .. note:: ``string.startswith`` and ``string.endswith`` checks are performed on each member of the object 
            and only matching member are displayed. This is usually done to prevent showing dunder members.
        
        .. seealso::
            :func:`~known.basic.Verbose.showX`
        """
        print(__class__.show_(x, cep=cep, sw=sw, ew=ew))

    @staticmethod
    def dir(x:Any, doc=False, filter:str='', sew=('__','__')):
        """ Calls ```dir``` on given argument and lists the name and types of non-dunder members.

        :param filter: csv string of types to filter out like `type,function,module`, keep blank for no filter
        :param doc: shows docstring ```__doc``` 
            If ```doc``` is True, show all member's ```__doc__```.
            If ```doc``` is False, does not show any ```__doc__```. 
            If ```doc``` is a string, show ```__doc__``` of specific types only given by csv string.

        :param sew: 2-Tuple (start:str, end:str) - excludes member names that start and end with specific chars, 
            used to exclude dunder methods by default
        """
        #if self_doc: print( f'{type(x)}\n{x.__doc__}\n' )
        if sew: sw, ew = f'{sew[0]}', f'{sew[1]}'
        doc_is_specified = (isinstance(doc, str) and bool(doc))
        if doc_is_specified: doc_match =[ t for t in doc.replace(' ','').split(',') if t ]
        if filter: filter_match =[ t for t in filter.replace(' ','').split(',') if t ]
        counter=1
        for k in dir(x):
            if sew:
                if (k.startswith(sw) and k.endswith(ew)): continue
            m = getattr(x,k)
            n = str(type(m)).split("'")[1]
            if filter:
                if not (n in filter_match):  continue
            s = f'[{counter}] {k} :: {n}'#.encode('utf-16')

            if doc:
                if doc_is_specified:
                    if n in doc_match: 
                        d = __class__.DOCSTR_FORM(m.__doc__)
                    else:
                        d=''
                else:
                    d = __class__.DOCSTR_FORM(m.__doc__)
            else:
                d = ''
            counter+=1
            print(f'{s}{d}')

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Symbols:
        CORRECT =       '✓'
        INCORRECT =     '✗'
        ALPHA =         'α'
        BETA =          'β'
        GAMMA =         'γ'
        DELTA =         'δ'
        EPSILON =       'ε'
        ZETA =          'ζ'
        ETA =           'η'
        THETA =         'θ'
        KAPPA =         'κ'
        LAMBDA =        'λ'
        MU =            'μ' 
        XI =            'ξ'
        PI =            'π'
        ROH =           'ρ'
        SIGMA =         'σ'
        PHI =           'φ'
        PSI =           'Ψ'
        TAU =           'τ'
        OMEGA =         'Ω'
        TRI =           'Δ'
        DOT=            '●'
        SUN=            '⚙'
        ARROW1=         '↦'
        ARROW2=         '⇒'
        ARROW3=         '↪'

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=




class Table:

    @staticmethod
    def CreateData(*columns):
        data = {None:[f'{col}' for col in columns]} # this is to make sure that col names are always on top
        return data

    @staticmethod
    def Create(columns:tuple, primary_key:str, cell_delimiter=',', record_delimiter='\n'):
        # should be called on a new object after init\
        table = __class__()
        table.data = __class__.CreateData(*columns)
        table.pk = primary_key
        table.pkat = table.data[None].index(table.pk)
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def ImportData(path, key_at, cell_delimiter, record_delimiter): 
        with open(path, 'r', encoding='utf-8') as f: 
            s = f.read()
            lines = s.split(record_delimiter)
            cols = lines[0].split(cell_delimiter) #<--- only if None:cols was added as a first entry (using Create method)
            data = {None:cols}
            if isinstance(key_at, str): key_at = cols.index(key_at)
            assert key_at>=0,f'Invlaid key {key_at}'
            for line in lines[1:]:
                if line:
                    cells = line.split(cell_delimiter)
                    data[f'{cells[key_at]}'] = cells
        return data
    
    @staticmethod
    def Import(path, key_at, cell_delimiter=',', record_delimiter='\n'): 
        table = __class__()
        table.data = __class__.ImportData(path, key_at, cell_delimiter, record_delimiter)
        if isinstance(key_at, str): key_at = table[None].index(key_at)
        table.pk = table.data[None][key_at]
        table.pkat = key_at
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def ExportData(data, path, cell_delimiter, record_delimiter): 
        with open(path, 'w', encoding='utf-8') as f: 
            for v in data.values(): f.write(cell_delimiter.join(v)+record_delimiter)

    @staticmethod
    def Export(table, path): 
        __class__.ExportData(table.data, path, table.cell_delimiter, table.record_delimiter)

    # get row as dict
    def __call__(self, key): return {k:v for k,v in zip(self[None], self[key])}

    # get row as it is (list)
    def __getitem__(self, key): return self.data[key]

    # set row based on if its a dict or a list (note: key is irrelavant here)
    def __setitem__(self, key, row):
        assert len(row) == len(self[None]), f'Rows are expected to have length {len(self[None])} but got {len(row)}'
        if isinstance(row, dict):
            key = row[self.pk]
            if key is not None: self.data[f'{key}'] = [row[r] for r in self[None]]
        else: 
            key = row[self.pkat]
            if key is not None: self.data[f'{key}'] = list(row)

    # del row based on key
    def __delitem__(self, key):
        if key is not None: del self.data[key]

    def __contains__(self, key): return key in self.data

    # quick export > file
    def __gt__(self, other):__class__.ExportData(self.data, f'{other}', self.cell_delimiter, self.record_delimiter)

    # quick import < file
    def __lt__(self, other): self.data = __class__.ImportData(f'{other}', self.pkat, self.cell_delimiter, self.record_delimiter)

    # total number of rows
    def __len__(self): return len(self.data)-1




#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

