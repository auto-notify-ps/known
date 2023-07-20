#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/mailer.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
import os, platform, smtplib, mimetypes, pickle #, json # imghdr
from email.message import EmailMessage
from zipfile import ZipFile

__all__=['Mail']


class Mail:
    r""" Use a g-mail account to send mail. 

    .. warning:: This requires saving your login credentials on machine in pickle format. 
        You should enable 2-factor-auth in gmail and generate an app-password instead of using your gmail password.
        Visit (https://myaccount.google.com/apppasswords) to generate app-password.
        Usually, these type of emails are treated as spam by google, so they must be marked 'not spam' at least once.
        It is recomended to create a seperate gmail account for sending mails.

    :param login_path: the path to login file

    A login file should be created first using ```save_login(path)``` (static) method. This will ask you to enter username and password
    which will be saved at the supplied path. The same path must be used for ```login_path``` argument.

    The usename and password is read from the file and used to login to gmail server every time ```send()``` or ```send_mail()``` is called.

    > use the ```Mail.send()``` static method to send emails.

    """
    
    DEFAULT_CTYPE = 'application/octet-stream'  

    @staticmethod
    def global_alias(prefix=''): return f'{prefix}{os.getlogin()} @ {platform.node()}:{platform.system()}.{platform.release()}'

    @staticmethod
    def zip_files(zip_path:str, files, **kwargs):
        r""" zips all (only files) in the list of file paths and saves at 'zip_path' """
        zipped = 0
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'
        with ZipFile(zip_path, 'w', **kwargs) as zip_object:
            for path in files:
                if not os.path.isfile(path): continue
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
    def zip_folders(zip_path:str, folders, **kwargs):  
        r""" zip multiple folders into a single zip file """    
        if isinstance(folders, str): folders= [f'{folders}']

        if not zip_path : zip_path = f'{folders[0]}.zip'
        if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  
        all_files = []
        for folder in folders: all_files.extend(__class__.get_all_file_paths(folder))
        return __class__.zip_files(f'{zip_path}', all_files, **kwargs)
    
    @staticmethod
    def zip_folder(folder:str, **kwargs):
        r""" zip a single folder with the same zip file name """     
        return  __class__.zip_files(f'{folder}.zip', __class__.get_all_file_paths(folder),  **kwargs)


    def __init__(self, login_path:str, signature=None, verbose=False) -> None: 
        self.verbose=verbose
        self.login_path=login_path
        if not os.path.isfile(login_path): print(f'{__class__} :: path @ [{login_path}] seems invalid, make sure to first create a login file using the "save_login" method.')
        self.signature = self.__class__.global_alias(prefix='\n') if signature is None else signature 
        
    def __call__(self, subject, rx, cc, bcc, content, attached = None): 
        r""" sends an e-mail message 

        :param subject: (str)
        :param rx: Recivers, csv string for 'To' field
        :param cc:  CarbonCopy, csv string for 'Cc' field
        :param bcc:  Blind CarbonCopy, csv string for 'Bcc' field
        :param content: lines to go inside msg body
        :param attached: (List of 2-Tuple) attachements for the email
        
        """
        
        self.__class__.send_mail(lambda : self.__class__.load_login(self.login_path),
        self.__class__.compose_mail(
            subject = f'{subject}', 
            rx = rx, cc = cc, bcc = bcc, 
            content = content + self.signature, 
            attached = attached, 
            verbose=self.verbose,)
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
    def compose_mail( subject:str, rx:str, cc:str, bcc:str, content:str, attached, verbose=True):
        r""" compose an e-mail msg to send later
        
        :param subject: subject
        :param rx: csv recipent email address
        :param cc: csv cc email address
        :param content: main content
        :param attached: list of attached files - is a 2-tupe (attachment_type, (args...) )

        # attach all files in the list :: ('', ('file1.xyz', 'file2.xyz'))
        # zip all the files in the list :: ('zipname.zip', '(file1.xyz', 'file2.xyz'))
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
        msg.set_content(content)
        if verbose: print(f'MESSAGE: #[{len(content)}] chars.')

        default_attached = []

        attached = [] if attached is None else attached
        for (attach_type, attach_args) in attached:
            if verbose: print(f' ... processing attachement :: {attach_type} :: {attach_args}')

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
                if verbose: print(f'\t --> zipped {zipped} items @ {zip_path} ')
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
            if verbose: print(f'[+] Attaching file [{main_type}/{sub_type}] :: [{file_name}]')
            with open (file_name, 'rb') as f: 
                file_data = f.read()
            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=os.path.basename(file_name))

        return msg

    @staticmethod
    def send_mail(login, msg, verbose=True):
        r""" send a msg using smtp.gmail.com:587 with provided credentials """
        username, password = login()
        if verbose: print(f'[*] Sending Email from {username}')
        msg['From'] = f'{username}' # set from
        with smtplib.SMTP('smtp.gmail.com', 587) as smpt:
            smpt.starttls()
            smpt.login(username, password)
            smpt.ehlo()
            smpt.send_message(msg)
        if verbose: print(f'[*] Sent!')


    @staticmethod
    def send(username:str, password:str, subject:str, rx:str, cc:str, bcc:str, content:str, attached, verbose=True):
        login = lambda: (username, password)
        msg = __class__.compose_mail(subject, rx, cc, bcc, content, attached, verbose)
        __class__.send_mail(login, msg, verbose)

    @staticmethod
    def save_pickle(obj, path:str,**kwargs):
        r""" save object to pickle file """
        with open(path, 'wb') as f: pickle.dump(obj, f,**kwargs)

    @staticmethod
    def load_pickle(path:str):
        r""" load pickle file to object """
        with open(path, 'rb') as f: o = pickle.load(f)
        return o

    @staticmethod
    def str2bytes(s:str, encoding:str='raw_unicode_escape')->list: return [i+b+1 for i,b in enumerate(bytearray(s, encoding))] #list(bytearray(s, encoding))

    @staticmethod
    def bytes2str(s, encoding:str='raw_unicode_escape')->str: return bytes.decode(bytes([b-i-1 for i,b in enumerate(s)]), encoding)

    @staticmethod
    def save_login(path):
        r""" save your login credentials as pickle """
        __class__.save_pickle((  __class__.str2bytes(input('Enter Username')), __class__.str2bytes(input('Enter Password'))  ), path)
    
    @staticmethod
    def load_login(path):
        r""" load your login credentials from json """
        login = __class__.load_pickle(path)
        return __class__.bytes2str(login[0]), __class__.bytes2str(login[1])
    #--------------------------------------------------