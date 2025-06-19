__doc__=r"""automation tools"""

import os
from ..basic import Table, Mailer, Symbols
import subprocess


import os
import smtplib, imaplib, email
from email.message import EmailMessage
from email.header import decode_header
import time



class feMailer(Mailer):
    r""" A fancy email notifier that uses operators to send mail in one line

    example:
    (__class__() \
        / 'from@gmail.com' \
        // 'password' \
        * 'subject' \
        @ 'to@gmail.com' \
        % 'cc@gmail.com' \
        - 'Email Msg Body' \
        + 'attachment.txt')()

    # NOTE: to read sender credentials from env-variables start the script using custom env as:
    # env "USERNAME=username@gmail.com" "PASSWORD=???? ???? ???? ????" python run.py

    # NOTE: the "joiner" arg specifies the char used to join list items in body

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
        self._status = False
        
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

    # Level 3 (SPECIAL CASES ONLY) -----------------------------------------

    def __invert__(self): return self.Compose(
            From=self.username,
            Subject=self.subject,
            To= self.to,
            Cc= self.cc,
            Body=self.joiner.join(self.content),
            Attached=tuple(self.attached),
        ) # composing ~nf

    def __and__(self, username): # set username     nf & "username"
        self.username = username  
        return self

    def __xor__(self, password): # set password     nf ^ "password"
        self.password = password 
        return self

    def __or__(self, other):    # send mail         nf | 1
        if other: self._status = self()
        else: self._status = False
        return self

    def __bool__(self): return self._status

    # Level 4 --------------------------------------------------------------

    def __call__(self, msg=None): return self.Send(
        msg = (msg if msg else ~self),
        username=( self.username if self.username else os.environ.get('USERNAME', '') ),
        password=( self.password if self.password else os.environ.get('PASSWORD', '') ),
        )

    #--------------------------------------------------------------



class Notifier:
    r""" Email Message Service based on gmail"""

    # -------------------------------------------------------------------------------------

    def Setup(self,
        username, 
        password,
        server='imap.gmail.com',
        port='993',
    ):
        self.username=f'{username}'
        self.password=f'{password}'
        self.server=server
        self.port=port
        self.Reset()
        
    def Reset(self):
        self.folders = None # after self.ListFolders
        self.folder = None  # after self.OpenFolder
        self.messages = None # after self.GetMessageList 
        self.imap = None 


    def Login(self):
        try:    self.imap = imaplib.IMAP4_SSL(self.server, int(self.port) ) #print(f'[{Symbols.CORRECT}] Connection Established to IMAP Server {self.server}:{self.port}')
        except: self.imap = None                                            #print(f'[{Symbols.INCORRECT}] Connection Failed')
        if self.imap:
            if self.imap.state == 'NONAUTH' or self.imap.state == 'LOGOUT':
                self.imap.login(self.username, self.password)
                return True, f'[{Symbols.CORRECT}] [State is {self.imap.state}] {self.username} logged in'
            else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] {self.username} cannot login'
        else: return False, f'[{Symbols.INCORRECT}] IMAP connection not established'


    def ListFolders(self):
        if self.imap.state == 'AUTH':
            status, folders = self.imap.list()
            return status, [folder.decode().split('"')[-2] for folder in folders]
        else:  return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] cannot list folders'

    def OpenFolder(self, folder='INBOX'):
        self.imap.select(folder)
        if self.imap.state=='SELECTED': 
            self.folder=folder
            return True, f'[{Symbols.CORRECT}] [State is {self.imap.state}] selected folder ({folder})'
        else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] cannot select folder ({folder})'


    def GetMessageList(self, criteria=['ALL',],):
        if self.imap.state=='SELECTED':
            bstatus, bmessages = self.imap.search(None, *criteria)
            self.messages = bmessages[0].split()
            return bstatus, f'Found {len(self.messages)} messages'
        else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] cannot get messages'

    def GetMessageInfo(self):
        if self.imap.state=='SELECTED':
            res=[]
            if self.messages is not None:
                for message_id in self.messages:
                    estatus, edata = self.imap.fetch(message_id, 'ENVELOPE')
                    fstatus, fdata = self.imap.fetch(message_id, 'FLAGS')
                    edata=edata[0].decode()
                    fdata=fdata[0].decode()
                    res.append((edata, fdata))
                return True, res
            else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] message list not avaibale, call GetMessageList first'
        else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] cannot open messages'


    def GetMessage(self,
        index,
        save='',
        seen=False,
        delete=False,
    ):
        message_id = self.messages[index]
        status, msg_data = self.imap.fetch(message_id, 'BODY[]') # fetch="(RFC822)"
        # ----------------------------------------------------
        msg = email.message_from_bytes(msg_data[0][1])
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):subject = subject.decode(encoding or "utf-8")
        attachments=[]
        body=""
        save = os.path.abspath(save) if save else None
        #if delete and not save: print(f'Warning: deleting without saving attachements ...')
        for part in msg.walk():
            content_type = part.get_content_type()
            content_dispo = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_dispo:
                body += part.get_payload(decode=True).decode(errors="ignore")            
            if "attachment" in content_dispo:
                filename = part.get_filename()
                if filename and save:
                    decoded_filename, enc = decode_header(filename)[0]
                    if isinstance(decoded_filename, bytes): filename = decoded_filename.decode(enc or "utf-8")
                    filepath = os.path.join(save, filename)
                    with open(filepath, "wb") as f: f.write(part.get_payload(decode=True))
                else: filepath=None
                attachments.append((filename, filepath))
            # ----------------------------------------------------
        res = {
        'From' : msg.get("From"),
        'To' : msg.get("To"),
        'CC' : msg.get("Cc"),
        'BCC' : msg.get("Bcc"),
        'Subject': subject,
        'Body': body,
        'Attachements': attachments,
        }
        # ----------------------------------------------------

        if delete: self.imap.store(message_id, '+FLAGS', '\\Deleted')
        if seen: self.imap.store(message_id, '+FLAGS', '\\Seen')

        return res
    

    def CloseFolder(self):
        if self.imap.state=='SELECTED':
            self.imap.expunge()
            self.imap.close()
            self.folder=None
            self.messages=None
            return True, f'[{Symbols.CORRECT}] [State is {self.imap.state}] closed folder'
        else: return False, f'[{Symbols.INCORRECT}] [State is {self.imap.state}] cannot close any folder'

    def Logout(self): 
        self.imap.logout() # will call self.imap.shutdown()
        self.Reset()
    


    
    # -------------------------------------------------------------------------------------

        
class AutoShell:
    r""" Executes a command or script and returns the output """

    SUBJECT_PREFIX = "[ps]."

    @staticmethod
    def newdb(export=""): 
        db = Table.Create(
                columns=('alias', 'username', 'password'),
                primary_key='alias',
                cell_delimiter=",", 
                record_delimiter='\n',)
        if export: Table.Export(db, os.path.abspath(f'{export}'))
        return db
        
    def __init__(self, alias, dbpath):
        alias=f'{alias}'
        assert alias, f'alias cannot be blank'
        dbpath = os.path.abspath(dbpath)
        assert os.path.isfile(dbpath), f'File Not Found "{dbpath}" \n Use newdb(export=path) to create one'
        
        self.alias=alias
        self.dbpath=dbpath
        self.db = self.newdb()
        self.db < self.dbpath

        self.notifier = Notifier()
        _, username, password = self.db[alias]
        self.notifier.Setup(username=username, password=password)
        self.Q = []
    
    def Fetch(self, folder, delete=False):
        # get all messages from this folder (and delete from inbox as well)
        count_prev = len(self.Q)
        success=self.notifier.Login()
        if success:
            if self.notifier.imap.state == 'AUTH':
                rstatus, reason = self.notifier.OpenFolder(folder)
                if rstatus:
                    mstatus, _ = self.notifier.GetMessageList(criteria=['ALL'])
                    if mstatus == 'OK':
                        if len(self.notifier.messages) > 0:
                            for i,m in enumerate(self.notifier.messages): self.Q.append(self.notifier.GetMessage(i, delete=delete))
                            self.notifier.CloseFolder()
                        else: print(f'No messages in the folder... will try later...')
                    else: print(f'Cannot get message list from folder... {reason}... will try later...')
                else: print(f'Cannot open folder... {reason}... will try later...')
            else: print(f'Cannot authorize connection... will try later...')
        else: print(f'Cannot open connection... will try later...')
        if success: self.notifier.Logout()
        return success, len(self.Q) - count_prev

    def Work(self):
        if not self.Q: return False, None
        T = self.Q.pop(0)
        subject = str(T['Subject'])
        if not subject.startswith(self.SUBJECT_PREFIX): return False, None
        service_name = subject[len(self.SUBJECT_PREFIX):]
        service_call = f'service_{service_name}'
        res = getattr(self, service_call)(T['Body'])
        return True, Mailer.SendMail(
            username=self.notifier.username, password=self.notifier.password,
            subject=f"(Re): {subject}", To=T['From'], Body=res,
        )

    def service_exe(self, body):

        res = ""
        # Run a shell command and capture the output
        result = subprocess.run(["cmd", "-c", body], capture_output=True, text=True)

        # Print stdout and stderr
        res+=f'Output:\n{result.stdout}\n\n'
        res+=f'Errors:\n{result.stdout}\n\n'
        res+=f'Return Code: {result.returncode}\n\n'
        return res



    # [{'From': 'Noti Fly <notifly.ps@gmail.com>',
    #     'To': 'auto.notify.ps@gmail.com',
    #     'CC': None,
    #     'BCC': None,
    #     'Subject': '[ps].s',
    #     'Body': "It's test\r\n\r\nOn Thu, 19 Jun, 2025, 04:08 Noti Fly,\r\n",
    #     'Attachements': []}]

