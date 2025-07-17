__doc__=r"""email tools"""

__all__ = [ 
    'Mailer', 
    'Mailbox', 
    ]

import os
import smtplib, mimetypes, imaplib
from email import message_from_bytes
from email.message import EmailMessage
from email.header import decode_header

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Mailer:
    r"""
    Use gmail account to read and send mails.
    Use `SendMail` static method to send mail
    Use `ReadMail` static method to read mail
    Both these methods require username and password
        > username must be a gmail address from which mail will be sent - recivers can be in any domain
        > password can be either one of:
            > google (account) password
            > app password

    Its recomended to use app password, to generate one visit 
        > https://myaccount.google.com/apppasswords
    
    App passwords requires enabling 2-step verification first, to enable it visit
        > https://myaccount.google.com/signinoptions/twosv

    
    -> on the reciver side - mark incoming mails as'not spam' at least once

    """
    @staticmethod
    def Mimes(files, default_ctype='application/octet-stream'):
        r""" gets mimetype info all files in a list """
        if isinstance(files, str): files=[f'{files}']
        res = []
        for path in files:
            if not os.path.isfile(path): continue
            ctype, encoding = mimetypes.guess_type(path)
            if ctype is None or encoding is not None: ctype = default_ctype
            maintype, subtype = ctype.split('/', 1)
            res.append( (path, maintype, subtype) )
        return res
    
    @staticmethod
    def Compose(From:str, Subject:str, To:str, Cc:str, Body:str, Attached:list=None, html:bool=False):
        msg = EmailMessage()

        # from, subject and recivers
        msg['From'] = f'{From}' 
        msg['Subject'] = f'{Subject}'
        msg['To'] = To
        if Cc: msg['Cc'] = Cc
        msg.set_content(Body)
        if html: msg.add_alternative(Body, subtype='html')
        # attachement
        if Attached is None: Attached = []
        assert isinstance(Attached, (list, tuple)), f'Expecting a list or tuple of attached files but got {type(Attached)}'       
        for file_name,main_type,sub_type in __class__.Mimes(Attached):
            with open (file_name, 'rb') as f: file_data = f.read()
            msg.add_attachment(file_data, 
                                maintype=main_type, 
                                subtype=sub_type, 
                                filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def Send(msg:EmailMessage, username:str, password:str, url='smtp.gmail.com', port='587', tls=True):
        r""" send a msg using url:port with provided credentials, calls `starttls` is `tls` is True """
        if not (username and password): return False
        try:
            with smtplib.SMTP(f'{url}', int(port)) as smtp: 
                if tls: smtp.starttls()
                # assert username == msg['From'] #<--- should be?
                smtp.login(username, password) 
                smtp.ehlo()
                smtp.send_message(msg)
        except: return False
        return True

    @staticmethod
    def SendMail(
        username, 
        password, 
        Subject, 
        To, 
        Cc, 
        Body, 
        Attached:list=None,
        html:bool=False,
        url='smtp.gmail.com', port='587', tls=True,
    ): return __class__.Send(__class__.Compose(username, Subject, To, Cc, Body, Attached=Attached, html=html), 
            username, password, url=url, port=port, tls=tls)

    @staticmethod
    def ReadFolders(
        username, 
        password, 
    ):
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(username, password)
        status, folders = imap.list()
        folders = [folder.decode().split('"')[-2] for folder in folders]
        imap.logout()
        return status, tuple(folders)

    @staticmethod
    def ReadMail(
        username, 
        password, 
        folder='INBOX',
        criteria='ALL',
        fetch='BODY[]',
        save='',
        delete=False,
        force=False,
    ):
        r"""

        #------------------------------------------
        Argument `ReadMail.folder`
        #------------------------------------------
        status, folders = imap.list()
        #------------------------------------------


        #------------------------------------------
        Argument `ReadMail.criteria`
        #------------------------------------------
        "ALL"	                        All messages
        "UNSEEN"	                    Unread messages
        "SEEN"	                        Read messages
        "FROM \"someone@example.com\""	Messages from a specific sender
        "TO \"someone@example.com\""	Sent to specific recipient
        "SUBJECT \"word\""	            Subject contains "word"
        "BODY \"word\""	                Body contains "word"
        "SINCE 01-Jan-2024"	            Messages after this date
        "BEFORE 01-Jun-2025"	        Messages before this date
        #------------------------------------------

        
        #------------------------------------------
        Argument `ReadMail.fetch`
        #------------------------------------------
        "BODY[]"	    Same as "RFC822" — full message
        "BODY[HEADER]"	Only the email headers (no body)
        "BODY[TEXT]"	Only the plain body, no headers
        "BODYSTRUCTURE"	Describes MIME structure (useful for parsing)
        "FLAGS"	        Read/unread flags (e.g. \Seen)
        "ENVELOPE"	    Summary info like subject, from, to, etc.
        #------------------------------------------


        """
        save = os.path.abspath(save) if save else None
        if delete and not save: 
            print(f'Warning: deleting without saving...')
            if not force: raise ZeroDivisionError(f'... set force=True to delete messages without saving')
                
        imap = imaplib.IMAP4_SSL("imap.gmail.com")
        imap.login(username, password)
        # To list other folders use: status, folders = imap.list()
        imap.select(folder)  # folder="INBOX"
        bstatus, bmessages = imap.search(None, criteria)
        messages = bmessages[0].split()
        bmsg=[]

        while messages:
            latest_msg = messages.pop()
            
            # Fetch the email
            status, msg_data = imap.fetch(latest_msg, fetch) # fetch="(RFC822)"
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = message_from_bytes(response_part[1])
                
                    # Decode subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(encoding or "utf-8")
                    
                    attachments=[]
                    # Process email parts
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_dispo = str(part.get("Content-Disposition"))

                        # Get the plain text body
                        if content_type == "text/plain" and "attachment" not in content_dispo:
                            body = part.get_payload(decode=True).decode(errors="ignore")

                        # Handle attachments
                        
                        if "attachment" in content_dispo:
                            filename = part.get_filename()
                            
                            if filename and save:
                                decoded_filename, enc = decode_header(filename)[0]
                                if isinstance(decoded_filename, bytes):
                                    filename = decoded_filename.decode(enc or "utf-8")
                                
                                filepath = os.path.join(save, filename)
                                with open(filepath, "wb") as f:
                                    f.write(part.get_payload(decode=True))
                                print(f"Attachment saved: {filepath}")
                            else:
                                filepath=None
                            
                            attachments.append((filename, filepath))
                    # 
                    bmsg.append( {
                    'From' : msg.get("From"),
                    'To' : msg.get("To"),
                    'Cc' : msg.get("Cc"),
                    'Bcc' : msg.get("Bcc"),
                    'Subject': subject,
                    'Body': body,
                    'Attachements': attachments,
                    })
            #
            if delete: imap.store(latest_msg, '+FLAGS', '\\Deleted')

        #
        if delete: imap.expunge()       
        imap.close()       
        imap.logout()

        return bstatus, bmsg

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Mailbox:
    r""" Email Reader based on imap.gmail """

    def __str__(self):return f'[{self.alias}] ({self.username})'
    def __repr__(self):return str(self)

    def Setup(self,
        username, 
        password,
        alias="",
        server='imap.gmail.com',
        port='993',
    ):
        self.alias=f'{alias}'
        self.username=f'{username}'
        self.password=f'{password}'
        self.server=server
        self.port=port
        self.Reset()
        return self
        
    def Reset(self):
        self.folder = None  # after self.OpenFolder
        self.messages = None # after self.GetMessageList 
        self.imap = None 

    def Login(self):
        self.imap = imaplib.IMAP4_SSL(self.server, int(self.port) )
        self.imap.login(self.username, self.password)
        return self.imap.state, self.imap.welcome

    def ListFolders(self):
        if self.imap.state == 'AUTH':
            status, folders = self.imap.list()
            return status, [folder.decode().split('"')[-2] for folder in folders]
        else:  return f'[State is {self.imap.state}] cannot list folders', None

    def OpenFolder(self, folder='INBOX'):
        self.imap.select(folder)
        if self.imap.state=='SELECTED': 
            self.folder=folder
            return f'[State is {self.imap.state}] selected folder ({folder})', True
        else: return f'[State is {self.imap.state}] cannot select folder ({folder})', False

    def GetMessageList(self, criteria=['ALL',],):
        if self.imap.state=='SELECTED':
            bstatus, bmessages = self.imap.search(None, *criteria)
            self.messages = bmessages[0].split()
            return bstatus, len(self.messages)
        else: return f'[State is {self.imap.state}] cannot get messages', -1

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
            else: return False, f'[State is {self.imap.state}] message list not avaibale, call GetMessageList first'
        else: return False, f'[State is {self.imap.state}] cannot open messages'

    def GetMessage(self,
        index,
        save='',
        seen=False,
        flag=False,
        delete=False,
    ):
        message_id = self.messages[index]
        status, msg_data = self.imap.fetch(message_id, 'BODY[]') # fetch="(RFC822)"
        # ----------------------------------------------------
        msg = message_from_bytes(msg_data[0][1])
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
        'Cc' : msg.get("Cc"),
        'Bcc' : msg.get("Bcc"),
        'Subject': subject,
        'Body': body,
        'Attachements': attachments,
        'Alias' : self.alias,
        }
        # ----------------------------------------------------

        if delete: self.imap.store(message_id, '+FLAGS', '\\Deleted')
        if flag: self.imap.store(message_id, '+FLAGS', '\\Flagged')
        if seen: self.imap.store(message_id, '+FLAGS', '\\Seen')

        return res
    
    def CloseFolder(self):
        if self.imap.state=='SELECTED':
            self.imap.expunge()
            self.imap.close()
            self.folder=None
            self.messages=None
            return True
        else: return False

    def Logout(self): 
        self.imap.logout() # will call self.imap.shutdown()
        self.Reset()
    
    # -------------------------------------------------------------------------------------

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
