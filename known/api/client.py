import requests, os, getpass
from http import HTTPStatus

class HeaderType: # lists the http headers used to carry special info
    XTAG =      'User-Agent'    # "Used to specify a Tag"
    XTYPE =     'Warning'       # "Used to specify a ContentType"

class RequestContentType: # lists the valid type of requests that clients can send back (specified in HeaderType.XTYPE header field)
    MESG = "MESG" # represents a string
    BYTE = "BYTE" # a stream of bytes
    JSON = "JSON" # a json serializable object
    FORM = "FORM" # a ClientForm with fields and attachements
    # only use either one of data or json in post request (not both)
    # form can only be sent from client to server but not other way

class ResponseContentType: # lists the valid type of response that server can send back to a `send` request (specified in HeaderType.XTYPE header field)
    MESG = "MESG" # represents a string
    BYTE = "BYTE" # a stream of bytes
    JSON = "JSON" # a json serializable object
    ALL = set([MESG, BYTE, JSON])

class StoreType: # lists the valid type of response that server can send back to a `path` reuest (specified in HeaderType.XTYPE header field)
    HOME = "H" # Home view which lists all possible paths in the root directory as a dict of {path:files}
    DIR =  "D" # Directory view returns a json dict as dict(base=<base_path_of_dir>, files=<list_of_files>, folders=<list_of_folders>)
    FILE = "F" # Indicates a files on the server, this must be saved to a location on client
    MSG = "M"  # Indicates that this is a message

class ClientForm:
    r""" Represents a form with fields and attachements that can sent to server using a POST request 
    
    example:

        form = ClientForm(
            name = "my_name",
            age = "100", 
        ).attach(
            alias="friendly_name1", 
            name="original_name1", 
            mime="file_type_hint", 
            handle=<name or handle or buffer>,
        ).attach(...).attach(...)

        the arguments to `__init__` should be string only - other data-type is not supported (it uses forms in POST method)
        the `attach` function returns the object itself so can be chained

        Note - handle can be either one of:
        # file name as string,  `f="/home/user/abc.txt"` - will open the file and close it after sending
        # open file handle,     `f=open("/home/user/abc.txt", 'r')` - will be automatically closed after sending
        # bytes buffer,         `f=io.BytesIO(); f.write(<something>);` - will seek to position-0 and close after sending
    """

    def __init__(self, **kwargs):
        self.data = {f'{k}':f'{v}' for k,v in kwargs.items()}
        self.attached={}
        self.files={}
    
    def attach(self, alias, name, mime, handle): 
        # handle can be either a file-path or a BytesIO object
        self.attached[alias] = dict(name=name, handle=handle, mime=mime, htype=isinstance(handle, str))
        return self

    def clear(self, data=True, files=True):
        if data: self.data.clear()
        if files: self.files.clear()

    def open(self):
        self.files.clear()
        for alias,info in self.attached.items():
            try:
                handle = open(info['handle'], 'rb') if info['htype'] else info['handle']
                handle.seek(0)
                self.files[alias] = (info['name'], handle, info['mime'])
            except: pass

    def close(self):
        for _, h, _ in self.files.values(): h.close()

class ClientResponse:
    r""" represents the response from a `send` request """
    truesym = '✔️'
    falsesym = '❌'
    def __init__(self, status_ok, xtype, xtag, xresponse):
        self.ok = status_ok
        
        self.type = xtype
        self.tag = xtag
        self.response = xresponse

    def __bool__(self): return self.ok
    def __str__(self): return f'[{self.truesym if self.ok else self.falsesym}] | Type={self.type} | Tag={self.tag} | Content={type(self.response)}'
    def __call__(self): return self.response

class Client:
    r""" HTTP Client Class - Represents a client that will access the API 
    The main methods to send data are:
    `send_` methods 
    `path_` methods 
    both type of methods return a `ClientResponse` object
    """

    def __init__(self, uid:str='', server:str='localhost:8080'):
        self.uid = uid if uid else getpass.getuser()
        assert len(self.uid)>0, f'Invalid uid {type(self.uid)}::[{self.uid}]'
        self.server = server
        self.url = f'http://{self.server}/'
        self.store = f'http://{self.server}/store/'
        self.timeout = None # (float or tuple) – How many seconds to wait for the server to send data - can be (connect timeout, read timeout) tuple.
        self.allow_redirects = False # we keep this False, only server will respond
        #self.params = None  # this is added to url, so make sure to pass strings only - both keys and values

    def __str__(self): return self.uid

    def check(self): # verify connection 
        # make a simple get request - the api should respond with ok
        try:        is_ok = requests.get(self.url, timeout=self.timeout).ok 
        except:     is_ok = False
        return      is_ok

    def send(self, xcontent:object, xtype:RequestContentType,  xstream:bool=False):
        # xtype is RequestContentType
        if xtype==RequestContentType.MESG: 
            xjson, xdata, xfiles = None, f'{xcontent}'.format('utf-8'), None
        elif xtype==RequestContentType.BYTE: 
            assert type(xcontent) is bytes, f'Expecting bytes but got {type(xcontent)}'
            xjson, xdata, xfiles = None, xcontent, None
        elif xtype==RequestContentType.FORM: 
            assert type(xcontent) is ClientForm
            xjson, xdata, xfiles = None, xcontent.data, xcontent.files
            xcontent.open()
        elif xtype==RequestContentType.JSON: 
            xjson, xdata, xfiles = xcontent, None, None
        else:               
            raise TypeError(f'Type "{xtype}" is not a valid content type') # xtype must be in ClientContentType

        # make a request to server ClientResponse
        try:
            response = requests.post(
                url=            self.url, allow_redirects=self.allow_redirects,  timeout=self.timeout,  #params=self.params,
                headers=        {HeaderType.XTYPE:xtype, HeaderType.XTAG:self.uid},
                stream=         xstream,
                json=           xjson,
                data=           xdata,
                files=          xfiles,
            )
        except: response = None
        finally:
            if xtype==RequestContentType.FORM: xcontent.close()
        return self.handle_response(response, xstream)

    def handle_response(self, response, xstream):
        # handle the response
        # NOTE: the `response` object contains the `request` object that we sent in response.request 
        # headers are sent always (independent of stream=True/False)
        if response is None:  status_ok, xtype, xtag, xresponse = False, None, None, None
        else:
            status_code =   response.status_code
            status_ok =     response.ok
            xtag =          response.headers.get(HeaderType.XTAG)
            xtype =         response.headers.get(HeaderType.XTYPE) # response content type

            if   status_code == HTTPStatus.OK: 
                if   xtype==ResponseContentType.MESG:       xresponse = response.content.decode('utf-8')
                elif xtype==ResponseContentType.BYTE:       xresponse = response.content
                elif xtype==ResponseContentType.JSON:       xresponse = response.json()
                else:                                       xresponse = None      
            elif status_code == HTTPStatus.NOT_ACCEPTABLE:  xresponse = None  
            elif status_code == HTTPStatus.NOT_FOUND:       xresponse = None  
            else:                                           xresponse = None   # this should not happen
            #if xstream: pass
            #else:        pass
            response.close()
        return ClientResponse(status_ok, xtype, xtag, xresponse)

    """ send_ methods """

    def send_mesg(self, message:str): return self.send(f'{message}', RequestContentType.MESG, xstream=False )
    def send_json(self, json_object:object): return self.send(json_object, RequestContentType.JSON, xstream=False )
    def send_form(self, client_form:ClientForm): return self.send(client_form, RequestContentType.FORM, xstream=False )
    def send_byte(self, byte_data:bytes): return self.send(byte_data, RequestContentType.BYTE, xstream=False )

    """ path_ methods """

    def path_get(self, path=None, save=None):
        r""" Query the store to get files and folders 
        
        `path`:         <str> the path on the server to get from. 
                        If path is a file, it will download the file and save it at the path provided in header `User-Agent` (it provides a filename)
                        If path is a folder, it will return a dict of listing dict(root=?, files=?, folders=?) `localhost:8080/store/path/to/folder`
                        if path is empty string "", gets the listing from root folder `localhost:8080/store/`
                        If path is None, does a directory listing at top level `localhost:8080/store`
                        
        `save`:        <str> (optional) the local path to save an incoming file, 
                        If None, uses the header `User-Agent` (not required for listing directory - only for file get)

        """
        try:    response = requests.get( url = ( self.store[:-1] if path is None else os.path.join(self.store, path) ), timeout = self.timeout, headers={HeaderType.XTAG: self.uid})
        except: return ClientResponse(False, None, None, "Exception in path_get.get method")
        uok = response.ok 
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ures=None
        if uok:
            if  utype ==  StoreType.HOME or utype == StoreType.DIR: ures = response.json()
            elif utype == StoreType.FILE: 
                if not save: save = utag
                if not save: ures = f"Invalid save location [{save}]"
                else:
                    try:
                        with open(save, 'wb') as f: f.write(response.content)
                    except: ures = f"Error Saving incoming file at {save}"
            else:       ures = f"Response type {utype} is unexpected for this request"
        else:           ures = f"Response not ok \n\n{response.text}" 
        response.close()
        return ClientResponse(uok, utype, utag, ures)
    def path_set(self, path, item=None):
        r""" Put files and folders on the server
        
        `path`:         <str> the path on the server to set at. 
        `item`:         the local path of a file to send (only when sending files not folders)
                        If item is a file, it will create a file on the server at `path` (stream file to server)
                        If item is None, it will create a folder at `path`
                        if item is anything else, error will be thrown

        """
        if item is None: 
            try:response = requests.put(url=os.path.join(self.store, path), timeout=self.timeout, headers={HeaderType.XTAG: self.uid})
            except: return ClientResponse(False, None, None, "Exception in path_set.put method")
        elif os.path.isfile(item):
            try:
                with open(item, 'rb') as f:
                    response = requests.post(url=os.path.join(self.store, path), data=f, timeout=self.timeout, headers={HeaderType.XTAG: self.uid})
            except: return ClientResponse(False, None, None, "Exception in path_set.post method")
        else: raise FileNotFoundError(f'cannot find path {item}')
        uok = response.ok
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ures = ""
        if uok:
            if utype ==  StoreType.MSG: ures = f'{response.text}'
            else:                       ures = f"Response type {utype} is unexpected for this request"
        else:                           ures = f"Response not ok \n\n{response.text}"  
        response.close()
        return ClientResponse(uok, utype, utag, ures)
    def path_del(self, path, recursive=False):
        r""" Delete files and folders from the server
        
        `path`:         <str> the path on the server to delete. 
                        If path is a file on the server, it will be deleted
                        If path is a folder on the server, it will be deleted only if its empty (set recurvie=True for recursive delete)
        """
        # only this request uses the XTYPE header to indicate if directory has to be recurviely deleted or not
        try: response = requests.delete(url= os.path.join(self.store, path), timeout=self.timeout, headers={HeaderType.XTAG: self.uid, HeaderType.XTYPE: f'{int(recursive)}'})
        except: return ClientResponse(False, None, None, "Exception in path_del.delete method")
        uok = response.ok
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ures = ""
        if uok:
            if utype ==  StoreType.MSG: ures = f'{response.text}'
            else:                       ures = f"Response type {utype} is unexpected for this request"
        else:                           ures = f"Response not ok \n\n{response.text}" 
        response.close()
        return ClientResponse(uok, utype, utag, ures)

