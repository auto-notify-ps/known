
import os
from .basic import Mailer

__all__ = ['Notifier']


class Notifier(Mailer):
    r"""

    # NOTE: to read sender credentials from env-variables start the script using custom env as:
    # env "USERNAME=username@gmail.com" "PASSWORD=???? ???? ???? ????" python run.py

    # NOTE: Operator preceedence 
    # * @ / // %
    # + -
    # << >>
    # & ^ | 

    """
    def __init__(self, joiner='\n'):
        self.joiner = joiner
        self.subject, self.to, self.cc, self.username, self.password  = '', '', '', '', ''
        self.content, self.attached = [], set([])
        
    def write(self, *lines): 
        self.content.extend(lines)
        return self

    def attach(self, *files):
        self.attached = self.attached.union(set(files))
        return self
    
    # Level 1 --------------------------------------------------------------

    def __truediv__(self, username):     # username/FROM     nf / 'from_mail@gmail.com'
        self.username = username  
        return self
    
    def __floordiv__(self, password):    # password         nf // 'pass word'
        self.password = password 
        return self
    
    def __mul__(self, subject):  # subject      nf * "subject"
        self.subject = subject   
        return self

    def __matmul__(self, to_csv):  # TO         nf @ 'to_mail@gmail.com,...'
        self.to = to_csv  
        return self
    
    def __mod__(self, cc_csv):      # CC       nf % 'cc_mail@gmail.com,...'
        self.cc = cc_csv
        return self

    # Level 2 --------------------------------------------------------------

    def __sub__(self, content):   # body        nf - "content"
        self.write(content)        
        return self
    
    def __add__(self, file):     # attachement      nf + "a.txt"
        self.attach(file)        
        return self

    # Level 3 --------------------------------------------------------------

    def __invert__(self): return self.Compose(
            From=self.username,
            Subject=self.subject,
            To= self.to,
            Cc= self.cc,
            Body=self.joiner.join(self.content),
            Attached=tuple(self.attached),
        )

    def __and__(self, username):
        self.username = username  
        return self

    def __xor__(self, password):
        self.password = password 
        return self

    def __or__(self, other):
        if other: self._status = self()
        else: self._status = False
        return self

    # Level 4 --------------------------------------------------------------

    def __call__(self, msg=None): return self.Send(
        msg = (msg if msg else ~self),
        username=( self.username if self.username else os.environ.get('USERNAME', '') ),
        password=( self.password if self.password else os.environ.get('PASSWORD', '') ),
        )

    #--------------------------------------------------------------



