#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/mailer/core.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os, smtplib, mimetypes, json # imghdr
from email.message import EmailMessage
from zipfile import ZipFile

__all__=['Mail']

class Mail:
    r""" Encapsulates a g-mail account that can be used to send mail. 
    Requires two-factor authentication with app password (https://myaccount.google.com/apppasswords) 

    :param username: (str) Sender's Email address
    :param password: (callable) when called (with no args), returns the password
    :param subject_fix: (2-Tuple) prefix and postfix for subject line
    :param content_seperator: (str) seperate each line in the email body
    :param signature: (str) Sender's Signature
    :param rx: (str) Recivers, csv string for 'To' field
    :param cc: (str) CarbonCopy, csv string for 'Cc' field
    :param bcc: (str) Back CarbonCopy, csv string for 'Bcc' field
    :param attached: (List of 2-Tuple) attachements for the email

    .. note:: ```rx```, ```cc```, ```bcc```, ```attached``` require default values during initialization. Keep None to keep blank.
        ```add_content```, ```add_rx```, ```add_cc```, ```add_bcc```, ```attach``` methods can be used to modify email. 
        ```send(subject)``` method is used to send email. ```clear``` clears all fields to recompose the email.
    """
    
    DEFAULT_CTYPE = 'application/octet-stream'  

    @staticmethod
    def load_json(path): 
        r""" loads a json file into a dict """
        try:
            with open(path, 'r') as f: d = json.load(f)
        except:
            d={}
        return d

    @staticmethod
    def save_json(path, d): 
        r""" save a dict as a json file, members must be serializable """
        try:
            with open(path, 'w') as f: json.dump(d, f, indent='\t', sort_keys=False)
        except:
            path=None
        return path


    def __init__(self, username, password,
            subject_fix=('',''), content_seperator='\n', signature='', 
            rx=None, cc=None, bcc=None, attached=None) -> None:
        self.username = username
        self.password = password
        self.subject_fix = tuple(subject_fix)
        self.content_seperator = content_seperator
        self.signature=signature
        self.default_rx = [username] if not rx else [f'{r}' for r in rx]
        self.default_cc = [] if not cc else [f'{r}' for r in cc]
        self.default_bcc = [] if not bcc else [f'{r}' for r in bcc]
        self.default_attached = [] if attached is None else attached
        self.clear()

    def clear(self): 
        r""" resets contents, recipents and attachements """
        self.content =  []
        self.rx = [r for r in self.default_rx]
        self.cc = [r for r in self.default_cc]
        self.bcc = [r for r in self.default_bcc]
        self.attached =  [r for r in self.default_attached]

    def add_content(self, *content): self.content.extend(content)
    def add_rx(self, *rx): self.rx.extend(rx)
    def add_cc(self, *cc): self.rx.extend(cc)
    def add_bcc(self, *bcc): self.rx.extend(bcc)
    def attach(self, zip_name:str, *paths): self.attached.append((f'{zip_name}', tuple(paths))) # set zip_name='' to attach individually
    def send(self, subject): 
        __class__.send_mail(self.password(), self.username,
        __class__.compose_mail(
            username = self.username, 
            subject = f'{self.subject_fix[0]}{subject}{self.subject_fix[1]}', 
            rx = f','.join(self.rx), cc = f','.join(self.cc), bcc = f','.join(self.bcc), 
            content = f'{self.content_seperator}'.join(self.content) + ((self.content_seperator+self.signature) if self.signature else ''), 
            attached = self.attached)
            )

    @staticmethod
    def get_mime_types(files):
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
    def zip_files(zip_path:str, files):
        r""" zips all (only files) in the list of paths and saves at 'zip_path' """
        zipped = 0
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'
        with ZipFile(zip_path, 'w') as zip_object:
            for path in files:
                if not os.path.isfile(path): continue
                #zip_object.write(f'{os.path.abspath(path)}')
                zip_object.write(f'{path}')
                zipped+=1
        return zipped, zip_path

    @staticmethod
    def get_all_file_paths(directory):
        r""" recursively list all files in a folder """
        file_paths = []
        # crawling through directory and subdirectories
        for root, directories, files in os.walk(directory):
            for filename in files:
                # join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
        return file_paths   

    @staticmethod
    def zip_folders(zip_path:str, folders):  
        r""" zip multiple folders into a single zip file """    
        if isinstance(folders, str): folders= [f'{folders}']

        if not zip_path : zip_path = f'{folders[0]}.zip'
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  
        all_files = []
        for folder in folders: all_files.extend(__class__.get_all_file_paths(folder))
        return __class__.zip_files(f'{zip_path}', all_files)
    
    @staticmethod
    def zip_folder(folder:str):
        r""" zip a single folder with the same zip file name """     
        return  __class__.zip_files(f'{folder}.zip', __class__.get_all_file_paths(folder))
    
    @staticmethod
    def compose_mail( username:str, subject:str, rx:str, cc:str, bcc:str, content:str, attached, verbose=True):
        r""" compose an e-mail msg to send later
        
        :param username: sender's email address
        :param subject: subject
        :param rx: csv recipent email address
        :param cc: csv cc email address
        :param content: main content
        :param attached: list of attached files - is a 2-tupe (attachment_type, (args...) )
        """
        
        msg = EmailMessage()

        # set subject
        msg['Subject'] = f'{subject}'
        if verbose: print(f'Subject: {subject}')

        # set from
        msg['From'] = f'{username}'
        if verbose: print(f'From: {username}')

        # set to
        msg['To'] = rx
        if verbose: print(f'To: {rx}')

        if cc: msg['Cc'] = cc
        if verbose: print(f'CC: {cc}')

        if bcc: msg['Bcc'] = bcc
        if verbose: print(f'BCC: {bcc}')

        # set content
        msg.set_content(content)
        if verbose: print(f'Content: #[{len(content)}] chars.')

        default_attached = []
            # attach all files in the list :: ('', ('file1.xyz', 'file2.xyz'))
            # zip all the files in the list :: ('zipname.zip', '(file1.xyz', 'file2.xyz'))

        for (attach_type, attach_args) in attached:
            if verbose: print(f'-- processing :: {attach_type}, {attach_args}')

            all_files = []
            for path in attach_args:
                if os.path.isdir(path):
                    all_files.extend(__class__.get_all_file_paths(path))
                elif os.path.isfile(path):
                    all_files.append(path)
                else:
                    if verbose: print(f'[!] Invalid Path :: {path}, skipped...')

            if not attach_type:  # attach individually
                default_attached.extend(__class__.get_mime_types(all_files))
            else: # make zip
                zipped, zip_path=__class__.zip_files(attach_type, all_files)
                if verbose: print(f'\t-- zipped {zipped} items @ {zip_path} ')
                if zipped>0:
                    default_attached.extend(__class__.get_mime_types(zip_path))
                else:
                    if verbose: print(f'[!] [{zip_path}] is empty, will not be attched!' )
                    try:
                        os.remove(zip_path)
                        if verbose: print(f'[!] [{zip_path}] was removed.' )
                    except:
                        if verbose: print(f'[!] [{zip_path}] could not be removed.' ) 
                

        # set attached ( name, main_type, sub_type), if sub_type is none, auto-infers using imghdr
        for file_name,main_type,sub_type in default_attached:
            if verbose: print(f'[+] attaching file [{main_type}/{sub_type}] :: {file_name}...')
            with open (file_name, 'rb') as f: 
                file_data = f.read()
            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def send_mail(password, username, msg, verbose=True):
        r""" send a msg using smtp.gmail.com:587 with provided credentials """
        if verbose: print(f'[*] Sending Email from {username}...')
        with smtplib.SMTP('smtp.gmail.com', 587) as smpt:
            smpt.starttls()
            smpt.login(username, password)
            smpt.ehlo()
            smpt.send_message(msg)
        if verbose: print(f'[*] Sent!')


