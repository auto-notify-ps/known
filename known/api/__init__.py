
__doc__=f""" 
-------------------------------------------------------------
Flask-based web api 
-------------------------------------------------------------
python -m pip install Flask waitress requests
python -m known.api --dir=/home/user --config=api
"""

DefaultArgs = {
    'dir'     : dict(   type=str, default='',    help="path of workspace directory"   ),
    'config'  : dict(   type=str, default='api', help="name of config-file"           ),
}

class HeaderType: # lists the http headers used to carry special info
    # headers are interpreted different 
    #   based on `request` and `response` object
    #   based on `api` and `store` access
    XTAG =      'known-api-tag'         # "Used to specify a Tag"
    XTYPE =     'known-api-type'        # "Used to specify a ContentType"


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
    ALL = {str:MESG, dict:JSON, list:JSON, tuple:JSON, bytes:BYTE}
    #ALL = set(list(MAP.keys()))

class StoreType: # lists the valid type of response that server can send back to a `path` reuest (specified in HeaderType.XTYPE header field)
    HOME = "H" # Home view which lists all possible paths in the root directory as a dict of {path:files}
    DIR =  "D" # Directory view returns a json dict as dict(base=<base_path_of_dir>, files=<list_of_files>, folders=<list_of_folders>)
    FILE = "F" # Indicates a files on the server, this must be saved to a location on client
    MSG = "M"  # Indicates that this is a message

