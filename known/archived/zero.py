__doc__="""Nothing useful here - Contains temporary/old code snippets, classes, functions etc"""







#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# class Fancy(Mailer):
#     r""" A fancy email notifier that uses operators to send mail in one line

#     example:
#     (__class__() \
#         / 'from@gmail.com' \
#         // 'password' \
#         * 'subject' \
#         @ 'to@gmail.com' \
#         % 'cc@gmail.com' \
#         - 'Email Msg Body' \
#         + 'attachment.txt')()

#     # NOTE: to read sender credentials from env-variables start the script using custom env as:
#     # env "USERNAME=username@gmail.com" "PASSWORD=???? ???? ???? ????" python run.py

#     # NOTE: the "joiner" arg specifies the char used to join list items in body

#     # NOTE: Operator preceedence 
#     # * @ / // %
#     # + -
#     # << >>
#     # & ^ | 

#     """
#     def __init__(self, joiner='\n'): 
#         self.joiner = joiner
#         self.subject, self.to, self.cc, self.username, self.password  = '', '', '', '', ''
#         self.content, self.attached = [], set([])
#         self._status = False
        
#     def write(self, *lines): 
#         self.content.extend(lines)
#         return self

#     def attach(self, *files):
#         self.attached = self.attached.union(set(files))
#         return self
    
#     # Level 1 --------------------------------------------------------------

#     def __truediv__(self, username):     # username/FROM     nf / 'from_mail@gmail.com'
#         self.username = username  
#         return self
    
#     def __floordiv__(self, password):    # password         nf // 'pass word'
#         self.password = password 
#         return self
    
#     def __mul__(self, subject):  # subject      nf * "subject"
#         self.subject = subject   
#         return self

#     def __matmul__(self, to_csv):  # TO         nf @ 'to_mail@gmail.com,...'
#         self.to = to_csv  
#         return self
    
#     def __mod__(self, cc_csv):      # CC       nf % 'cc_mail@gmail.com,...'
#         self.cc = cc_csv
#         return self

#     # Level 2 --------------------------------------------------------------

#     def __sub__(self, content):   # body        nf - "content"
#         self.write(content)        
#         return self
    
#     def __add__(self, file):     # attachement      nf + "a.txt"
#         self.attach(file)        
#         return self

#     # Level 3 (SPECIAL CASES ONLY) -----------------------------------------

#     def __invert__(self): return self.Compose(
#             From=self.username,
#             Subject=self.subject,
#             To= self.to,
#             Cc= self.cc,
#             Body=self.joiner.join(self.content),
#             Attached=tuple(self.attached),
#         ) # composing ~nf

#     def __and__(self, username): # set username     nf & "username"
#         self.username = username  
#         return self

#     def __xor__(self, password): # set password     nf ^ "password"
#         self.password = password 
#         return self

#     def __or__(self, other):    # send mail         nf | 1
#         if other: self._status = self()
#         else: self._status = False
#         return self

#     def __bool__(self): return self._status

#     # Level 4 --------------------------------------------------------------

#     def __call__(self, msg=None): return self.Send(
#         msg = (msg if msg else ~self),
#         username=( self.username if self.username else os.environ.get('USERNAME', '') ),
#         password=( self.password if self.password else os.environ.get('PASSWORD', '') ),
#         )

#     #--------------------------------------------------------------









# class AutoFetcher:
#     r""" Fetches emails from a folder in gmail and transfers them to a Queue, 
#         then processes this Queue using user-defined Callback 
#         A callback can be defined in the child class
#         def Callback(self, From, To, Cc, Bcc, Subject, Body, Attachements, Alias, **kwargs): return ...
#         """

#     def __init__(self):
#         self.mailboxes =                        {} # mailboxes to monitor
#         self.Q =                                [] # processing queue

#     def Create(self, alias, username, password):
#         self.mailboxes[alias] = Mailbox().Setup(username=username, password=password, alias=alias)
#         return self

#     def Fetch(self, alias, folder, save="", delete=False):
#         # get all messages from given folder (and flag as well)
#         mailbox = self.mailboxes[alias]
#         try:    _= mailbox.Login()
#         except: return False, f'Cannot open connection'
        
#         if mailbox.imap.state != 'AUTH': return False, f'Cannot authorize connection'

#         reason, rstatus = mailbox.OpenFolder(folder)
#         if not rstatus: return False, f'Cannot open {folder}: {reason}'

#         criteria = ['ALL'] if delete else ['UNFLAGGED']

#         mstatus, mcount = mailbox.GetMessageList(criteria=criteria)
#         if mstatus != 'OK': return False, f'Cannot get message list from {folder}: {reason}'
        
#         for i in range(mcount): self.Q.append(mailbox.GetMessage(i, save=save, seen=False, flag=not delete, delete=delete))        
#         mailbox.CloseFolder()
#         mailbox.Logout()
#         return True, f'Added {mcount} tasks'

#     def Work(self, popat=0):
#         while self.Q: status = self.Callback(**self.Q.pop(popat))
            
#     def Callback(self, **kwargs): return ...
#     #def Callback(self, From, To, Cc, Bcc, Subject, Body, Attachements, Alias, **kwargs): return ...

# #=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
