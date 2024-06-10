

import requests, os
from http import HTTPStatus



class HeaderType:
    XTAG =      'User-Agent' # "Used to specify a Tag"
    XTYPE =     'Warning'  # "Used to specify a ContentType"

class ContentType:
    MESG = "MESG" # represents a string
    BYTE = "BYTE" # a stream of bytes
    JSON = "JSON" # a json serializable object
    FORM = "FORM" # a ClientForm with fields and attachements
    ALL = set([MESG, BYTE, JSON, FORM])

    # only use either one of data or json in post request (not both)
    # form can only be sent from client to server but not other way

class StoreType:
    VIEW = "V"
    DIR =  "D"
    FILE = "F"
    MSG = "M"

class ClientForm:
    r""" Represents a form with fields and attachements that can sent to server using a POST request """

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

class Client:
    r""" HTTP Client Class - Represents a client that will access the API """

    # ClientContentType = dict(
    #     BYTE='use the data field - can put any binary data here as bytes',
    #     FORM='use the data field for key-value pairs',
    #     JSON='use the get_json() method, puts a json-serializable object',
    # ) # with every request, the client this in the xtype (Warning) header

    def __init__(self, server='localhost:8080'):
        self.server = server
        self.url = f'http://{self.server}/'
        self.store = f'http://{self.server}/store/'
        self.timeout = 10.0 # # (float or tuple) – How many seconds to wait for the server to send data - can be (connect timeout, read timeout) tuple.
        self.allow_redirects = False # we keep this False, only server will respond
        self.params = None  # this is added to url, so make sure to pass strings only - both keys and values

    def check(self): # verify connection 
        # make a simple get request - the api should respond with ok
        try:        is_ok = requests.get(self.url, timeout=self.timeout).ok 
        except:     is_ok = False
        return      is_ok



    def send(self, xcontent, xtype,  xtag='', xstream=False):
        # xtype is <str> 'MESG' 'BYTE', 'FORM', 'JSON'
        if xtype==ContentType.MESG: 
            xjson, xdata, xfiles = None, f'{xcontent}'.format('utf-8'), None
        elif xtype==ContentType.BYTE: 
            assert type(xcontent) is bytes, f'Expecting bytes but got {type(xcontent)}'
            xjson, xdata, xfiles = None, xcontent, None
        elif xtype==ContentType.FORM: 
            assert type(xcontent) is ClientForm
            xjson, xdata, xfiles = None, xcontent.data, xcontent.files
            xcontent.open()
        elif xtype==ContentType.JSON: xjson, xdata, xfiles = xcontent, None, None
        else:               raise TypeError(f'Type "{xtype}" is not a valid content type') # xtype must be in ClientContentType

        # make a request to server
        #print(f'\n[SENDING]\n{xtype=}\t{xtag=}\n{xjson=}\n{xdata=}\n{xfiles=}')

        response = requests.post(
            url=            self.url, allow_redirects=self.allow_redirects,  timeout=self.timeout,  params=self.params,
            headers=        {'User-Agent':xtype, 'Warning':xtag}, # https://en.wikipedia.org/wiki/List_of_HTTP_header_fields
            stream=         xstream,      # (optional) if False, the response content will be immediately downloaded

            json=           xjson,         # (optional) A JSON serializable Python object to send in the body of the Request - works only when no form data and files
            # OR                        # either use (json) or (data + files)
            data=           xdata,         # (optional) Dictionary, list of tuples, bytes, or file-like object to send in the body of the Request.
            files=          xfiles,         # (optional) Dictionary of 'name': file-like-objects (or {'name': file-tuple}) for multipart encoding upload. 
                                        #   file-tuple can be a 
                                        #       2-tuple ('filename', fileobj), 
                                        #       3-tuple ('filename', fileobj, 'content_type')
                                        #       4-tuple ('filename', fileobj, 'content_type', custom_headers), 
                                        # ... where 'content_type' is a string defining the content type of the given file and 
                                        # ... custom_headers a dict-like object containing additional headers to add for the file.
        )
        if xtype==ContentType.FORM: xcontent.close()
        return self.handle_response(response, xstream)

    def handle_response(self, response, streamed):
        # handle the response
        
        # NOTE: the `response` object contains the `request` object that we sent,
        # response.request

        # If we want to access the headers the server sent back to us, we do this:
        # response.headers 
        # headers are sent always (independent of stream=True/False)

        status_code = response.status_code
        status_ok = response.ok
        xtag = response.headers.get(HeaderType.XTAG)
        xtype = response.headers.get(HeaderType.XTYPE)

        if   status_code == HTTPStatus.OK: 
            if   xtype==ContentType.MESG: xresponse = response.content.decode('utf-8')
            elif xtype==ContentType.BYTE: xresponse = response.content
            elif xtype==ContentType.FORM: xresponse = None # this should not be used
            elif xtype==ContentType.JSON: xresponse = response.json()
            else:               xresponse = None      
        elif status_code == HTTPStatus.NOT_ACCEPTABLE:  xresponse = None  
        elif status_code == HTTPStatus.NOT_FOUND:       xresponse = None  
        else:                                           xresponse = None   # this should not happen

        #if streamed: pass
        #else:        pass
        
        response.close()
        #f'[{"✳️" if status_ok else "❌"}]::{status_code}::{xtag=}::{xhint=}\n👉\n{res}\n👈\n{content}'
        return status_ok, xtype, xtag, xresponse


    def store_get(self, path=None, save=None):
        r""" Query the store to get files and folders 
        
        `path`:         <str> the path on the server to get from. 
                        If path is a file, it will download the file and save it at the path provided in header `User-Agent` (it provides a filename)
                        If path is a folder, it will return a dict of listing dict(root=?, files=?, folders=?) `localhost:8080/store/path/to/folder`
                        if path is empty string "", gets the listing from root folder `localhost:8080/store/`
                        If path is None, does a directory listing at top level `localhost:8080/store`
                        
        `save`:        <str> (optional) the local path to save an incoming file, 
                        If None, uses the header `User-Agent` (not required for listing directory - only for file get)

        """        
        response = requests.get( url = ( self.store[:-1] if path is None else os.path.join(self.store, path) ), timeout = self.timeout)
        uok = response.ok
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ureason = ""
        if uok:
            if  utype ==  StoreType.VIEW or utype == StoreType.DIR: res = response.json()
            elif utype == StoreType.FILE: 
                if not save: save = utag
                try:
                    with open(save, 'wb') as f: f.write(response.content)
                except: ureason = f"Error Saving incoming file at {save}"
            else:       ureason = f"Response type {utype} is unexpected for this request"
        else:           ureason = f"Response not ok"  
        response.close()
        return uok, ureason

    def store_set(self, path, item=None):
        r""" Put files and folders on the server
        
        `path`:         <str> the path on the server to set at. 
        `item`:         the local path of a file to send (only when sending files not folders)
                        If item is a file, it will create a file on the server at `path` (stream file to server)
                        If item is None, it will create a folder at `path`
                        if item is anything else, error will be thrown

        """
        if item is None: response = requests.put(url=os.path.join(self.store, path), timeout=self.timeout)
        elif os.path.isfile(item):
            with open(item, 'rb') as f:
                response = requests.post(url=os.path.join(self.store, path), data=f, timeout=self.timeout)
        else: raise FileNotFoundError(f'cannot find path {item}')
        uok = response.ok
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ureason = ""
        if uok:
            if utype ==  StoreType.MSG: ureason = f'{response.text}'
            else:                       ureason = f"Response type {utype} is unexpected for this request"
        else:                           ureason = f"Response not ok"  
        response.close()
        return uok, ureason

    def store_del(self, path, recursive=False):
        r""" Delete files and folders from the server
        
        `path`:         <str> the path on the server to delete. 
                        If path is a file on the server, it will be deleted
                        If path is a folder on the server, it will be deleted only if its empty (set recurvie=True for recursive delete)
        """
        # only this request uses the XTYPE header to indicate if directory has to be recurviely deleted or not
        response = requests.delete(url= os.path.join(self.store, path), timeout=self.timeout, headers={HeaderType.XTYPE: f'{int(recursive)}'})
        uok = response.ok
        utype = response.headers.get(HeaderType.XTYPE)
        utag = response.headers.get(HeaderType.XTAG)
        ureason = ""
        if uok:
            if utype ==  StoreType.MSG: ureason = f'{response.text}'
            else:                       ureason = f"Response type {utype} is unexpected for this request"
        else:                           ureason = f"Response not ok"  
        response.close()
        return uok, ureason






