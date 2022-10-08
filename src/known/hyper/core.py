#-----------------------------------------------------------------------------------------------------
# mdlog/md.py
#-----------------------------------------------------------------------------------------------------
import os.path, sys
from os import makedirs
#-----------------------------------------------------------------------------------------------------

class LOGGER:
    """ base class for file-based logger """

    def __init__(self, log_dir, log_file, log_extension):
        """
            log_dir         : [str]   directory to create new log file at
            log_file        : [str]   name of new log file
            log_extension   : [str]   extension for file (dont add dot)
            
            Note: By default, the escmode is False, does not escape any special chars
        """
        self.log_dir = log_dir              
        self.log_file = log_file + '.' + log_extension
        self.log_path = os.path.join(self.log_dir, self.log_file)
        self.iomode=''      #<--- no iomode means closeed
        self.escmode({}) #<---- manually set the escape mode after init if its required

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """ Logger File Handles """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    def open(self, mode='w'):
        """ open log file handle """
        try:
            assert (mode=='a' or mode=='w')
        except AssertionError:
            print(f'[Error]: Log file can be only opened in append(a) or write(w) mode.')
        self.iomode = mode
        makedirs(self.log_dir, exist_ok=True)
        self.f = open(self.log_path, mode)

    def close(self):
        """ close log file handle """
        self.f.close()
        self.iomode=''
        del self.f

    def loc(self, file):
        """ returns relative loaction of a file wrt log file 
            - this is useful to linking local files as URLs
            - 'uri' function will auto-convert to relative path if its 'loc' arg is True
        """
        return os.path.relpath( file , self.log_dir )

    def info(self, p=print):
        """ short info about file handle"""
        p('[Logging ~ Mode:[{}] @File:[{}] @Path:[{}]'.format(self.iomode, self.log_file, self.log_path))

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """  IO and Escaping """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    def escmode(self, esc):
        """ enable or disable escaping, sets the 'set of escape charaters' in self.esc 
            - if 'esc' evaluates to False, then escaping is disabled 
            - otherwise, it should be a tuple, all chars in the 'esc' tuple are escaped """
        self.esc = esc
        self.do_esc = True if self.esc else False
        self.write = self.write_do_esc if self.do_esc else self.write_no_esc

    def escape(self, msg):
        """ escapes all instances of chars in self.esc tuple """
        m = str(msg)
        for k,v in self.esc.items():
            m = m.replace(k, v)
        return m

    def escaper(self, escdict):
        """ returns a context-manager for temporary escaping special chars - see the ESCAPER class """
        return ESCAPER(self, escdict)

    def write_no_esc(self, *msg):
        """ write msg without escaping """
        for m in msg:
            self.f.write(str(m))

    def write_do_esc(self, *msg):  
        """ write msg with escaping - escapes all chars that are currently in self.esc tuple """
        emsg = map(self.escape, msg)
        for m in emsg:
            self.f.write(m)

    def ascape(s): # attribute-escape (for mu only)
        t = str(s)
        for k,v in {'"' : '&quot;', "'" : '&#39;'}:
            t = t.replace(k,v)
        return t


    # std-output-redirect
    # NOTE: Child classes should implement codeblock open and close
    def rdr_(self, as_code=True, dual=False): 
        """ redirects std-out(console output) to log file 
        Args:
            as_code: if True, opens a code block before and after redirecting
            dual:    if True, prints the std-output on consile as well
        """
        self.rdr_as_code = as_code
        if self.rdr_as_code:
            self.codeblock_() #<---- must implement
        self.xf = sys.stdout
        sys.stdout = (FAKER(self) if dual else self.f)
    def _rdr(self):
        """ stop redirecting from std-output """
        sys.stdout = self.xf
        if self.rdr_as_code:
            self._codeblock() #<---- must implement
        del self.rdr_as_code
    def codeblock_(self):
        raise NotImplementedError(f'Should implement in inherited class')
    def _codeblock(self):
        raise NotImplementedError(f'Should implement in inherited class')

    # NOTE: use self.write(*msg) to put chars to file directly 
    # NewLines - Generic (file write)
    def nl(self):
        """ [new-line] put newline to file """
        self.f.write('\n')
    def nl2(self):
        """ [new-line-2] put 2 newlines to file """
        self.f.write('\n\n')


class FAKER:
    """ implements a fake-handle for dual output - mainly implements write method """
    def __init__(self, parent) -> None:
        self.parent = parent
    def write(self, *args):
        self.parent.f.write(*args)
        self.parent.xf.write(*args) #<--- temporary 'xf' which is sys.stdout


class ESCAPER:
    """ context manager for toggling character escaping while logging,
        > escaping requires extra computing and is avoided by default 
        > user can switch on escaping specific charecters using this context manager """
    def __init__(self, logger, escdict) -> None:
        """ *log_tup : tuples like (log, esc) """
        self.logger = logger 
        self.escdict = escdict
        self.logger.pesc = {}

    def __enter__(self):
        self.logger.pesc.update(**self.logger.esc)
        self.logger.escmode(self.escdict)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.escmode(self.logger.pesc)
        del self.logger.pesc
        return True

    """ Predefined escape char sets """
    MD_ESC = lambda *keys: {k:'\\'+k for k in keys}

    MD_FORM_ESC =      MD_ESC('`', '*', '_', '~'  ) #{k:'\\'+k for k in (  '`', '*', '_', '~'  )}
    MD_LINK_ESC =      MD_ESC( '{', '}', '[', ']', '(', ')', '!') #{k:'\\'+k for k in ( '{', '}', '[', ']', '(', ')', '!')} 
    MD_BLOCK_ESC =     MD_ESC( '+', '-', '|', '>' ) #{k:'\\'+k for k in ( '+', '-', '|', '>' )}
    MD_ALL_ESC =       {**MD_FORM_ESC, **MD_LINK_ESC, **MD_BLOCK_ESC}
    MD_NO_ESC =        {} #<--- anything evaluating to false is no_esc

    MU_NBSP_ESC={' ' : '&nbsp;'}
    MU_TAG_ESC = {
    '<' : '&lt;', 
    '>' : '&gt;', 
    }
    MU_STD_ESC = {
    '&' : '&amp;',
    '<' : '&lt;', 
    '>' : '&gt;', 
    }
    MU_MU_ESC = {
    '&' : '&amp;',
    '<' : '&lt;', 
    '>' : '&gt;', 
    ' ' : '&nbsp;'
    }
#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
