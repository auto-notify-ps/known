#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
__doc__=r"""
:py:mod:`known/mailer.py`
"""
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os, smtplib, mimetypes # imghdr
from email.message import EmailMessage
from zipfile import ZipFile
from .basic import j2d, d2j
__all__=['MAIL']


class MAIL:
    r""" Encapsulates a g-mail account that can be used to send mail. 
    Requires two-factor authentication with app password (https://myaccount.google.com/apppasswords) 
    

    ::

        from known.mailer import MAIL
        mailer = MAIL.load_config('mail.json')
        mailer.clear()
        mailer.add_content('Line 5', 'Line 6', 'Line 7') # <<- use add_content() to write lines in email
        mailer.add_recipent('someone@gmail.com', 'anyone@gmail.com') #<<- use add_recipent() to add more recipents
        mailer.send(subject='Test Notify')

        # save a config
        d = dict(
        username = 'your_email@gmail.com', 
        password = 'your_password_here',
        subject_fix = ('[Auto.Notify] @ [', ']'), # prefix and postfix for suject decoration
        content_seperator='\n', # seperates each string in contents by this char
        signature = "\nWarm Regards,\n@[Auto.Notify] :: This e-mail was auto-generated.\n", # append this to end of contents
        recipents = ['person1@domain1.com', 'person2@domain2.com', ], # recipent emails
        attached = [
        ( 'ff', ('file1.ext', 'file2.ext', )                       ),  # attaches each file seperately
        ( 'zf', ('zipname.zip', 'file1.ext', 'file2.ext', )        ),  # attaches zip of all files
        ( 'df', ('folder1', 'folder2', )                           ),  # attaches each file seperately, recursively from all folders
        ( 'zd', ('zipname.zip', 'folder1', 'folder2', )            )   # attaches zip of multiple folders],)
        known.d2j(d, 'mail.json')
    
    
    """
    
    DEFAULT_CTYPE = 'application/octet-stream'  
    AttachType_files='ff' # attach all files in the list :: ('ff', ('file1.xyz', 'file2.xyz'))
    AttachType_zipfiles ='zf' # zip all the files in the list, 0th element is the zip name :: ('zf', ('zip1.zip', 'file1.xyz', 'file2.xyz'))
    AttachType_dirfiles='df' # attach all files from the folder recursivly :: ('df', ('folder1', 'folder2'))
    AttachType_zipdirs='zd' # zip the folder rcurvisly :: ('zd', ('folder1', 'folder2'))



    @staticmethod
    def load_config(path): 
        r""" creates a new MAIL object from saved config in json format """
        return __class__(**j2d(path))

    @staticmethod
    def save_config(mail, path='mail.json'):
        r""" saves current MAIL object config in json format """
        d2j(dict(
            username = mail.username,
            password = mail.password,
            subject_fix = mail.subject_fix,
            content_seperator=mail.content_seperator,
            signature = mail.signature,
            recipents = mail.default_recipents ,
            attached = mail.default_attached,
        ), path)


    def __init__(self, username, password, subject_fix=('',''), content_seperator='\n', signature='', recipents=None, attached=None) -> None:
        self.username = username
        self.password = password
        self.subject_fix = tuple(subject_fix)
        self.content_seperator = content_seperator
        self.signature=signature
        self.default_recipents = [username] if not recipents else [f'{r}' for r in recipents]
        self.default_attached = [] if attached is None else attached
        self.clear()

    def clear(self): 
        r""" resets contents, recipents and attachements """
        self.content =  []
        self.recipents = [r for r in self.default_recipents]
        self.attached =  [r for r in self.default_attached]

    def add_content(self, *content): self.content.extend(content)
    def add_recipent(self, *recipent): self.recipents.extend(recipent)

    def send(self, subject): 
        __class__.send_mail(self.password, self.username,
                __class__.compose_mail(
                        username = self.username, 
                        subject = f'{self.subject_fix[0]}{subject}{self.subject_fix[1]}', 
                        recipent = f','.join(self.recipents), 
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
    
        
    #@staticmethod
    #def zip_folder(folder:str):        
    #    return  __class__.zip_files(f'{folder}.zip', __class__.get_all_file_paths(folder))
    

    @staticmethod
    def compose_mail( username:str, subject:str, recipent:str, content:str, attached, verbose=True):
        r""" compose an e-mail msg to send later
        
        :param username: sender's email address
        :param subject: subject
        :param recipent: csv recipent email address
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
        msg['To'] = recipent
        if verbose: print(f'To: {recipent}')

        # set content
        msg.set_content(content)
        if verbose: print(f'Content: #[{len(content)}] chars.')

        default_attached = []
            # files='ff' # attach all files in the list :: ('ff', ('file1.xyz', 'file2.xyz'))
            # zipfiles ='zf' # zip all the files in the list, 0th element is the zip name :: ('zf', ('zipname.zip', 'file1.xyz', 'file2.xyz'))
            # dirfiles='df' # attach all files from the folder recursivly :: ('df', ('folder1', 'folder2'))
            # zipdirs='zd' # zip the folder rcurvisly :: ('zd',  ('zipname.zip', folder, folder))

        for (attach_type, attach_args) in attached:
            if verbose: print(f'-- processing :: {attach_type}, {attach_args}')
            if attach_type == __class__.AttachType_files: #  ('ff', ('file1.xyz', 'file2.xyz'))
                default_attached.extend(__class__.get_mime_types(attach_args))

            elif attach_type == __class__.AttachType_zipfiles: # ('zf', ('zipname.zip', 'file1.xyz', 'file2.xyz'))
                zipped, zip_path=__class__.zip_files(attach_args[0], attach_args[1:])
                if verbose: print(f'\t-- zipped {zipped} items @ {zip_path} ')
                default_attached.extend(__class__.get_mime_types(zip_path))

            elif attach_type == __class__.AttachType_dirfiles: # ('df', ('folder1', 'folder2'))
                all_files=[]
                for folder in attach_args: all_files.extend(__class__.get_all_file_paths(folder))
                default_attached.extend(__class__.get_mime_types(all_files))

            elif attach_type == __class__.AttachType_zipdirs: # ('zd',  ('zipname.zip', folder, folder))
                zipped, zip_path=__class__.zip_folders(attach_args[0], attach_args[1:])
                if verbose: print(f'\t-- zipped {zipped} items @ {zip_path} ')
                default_attached.extend(__class__.get_mime_types(zip_path))
            else:
                if verbose: print(f'[!] AttachType: {attach_type} is invalid' )

        # set attached ( name, main_type, sub_type), if sub_type is none, auto-infers using imghdr
        for file_name,main_type,sub_type in default_attached:
            if verbose: print(f'[+] attaching file [{main_type}/{sub_type}] :: {file_name}...')
            with open (file_name, 'rb') as f: file_data = f.read()
            msg.add_attachment(file_data, maintype=main_type, subtype=sub_type, filename=file_name)

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



