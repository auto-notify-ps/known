

import requests

class Client:
    r""" Client class for making api calls (http reuqests)

    :param server: <str> the address of api server
    :param port:   <str> the port of api server


    To send json data, use: Client.send_json(json_object)
    ---> The <json_object> can be a dict, list or string

    To send buffers, use: Client.send_buffer(**buffers):
    ---> The <buffers> are named buffers provided as keyword arguments like {buffer-name : buffer-object}
    ---> If using buffer, make sure to buffer.seek(0) before sending 

    To send files, use: Client.send_file(*handles):
    ---> The <handles> is a list of file names


    The response contains 3 things - msg, mtype, mcode
    msg :   base object sent by server - a string, list or dict
    mtype:  type of msg object - can be 'text', 'json', 'bytes' or 'error'  (error has text type msg)
    mcode:  status code - can be True or False (true when HTTP response was recieved, false otherwise)
    """
    
    class MTypes:
        ERROR = -1
        NONE = 0
        TEXT = 1
        JSON = 2
        BYTES = 3
        _rev = {
            -1 : 'ERROR',
            0 : 'NONE',
            1 : 'TEXT',
            2 : 'JSON',
            3 : 'BYTES',

        }
        def code(c): return __class__._rev.get(int(c), None)

    def __init__(self, server='localhost', port='8080'): 
        self.server = f'http://{server}:{port}/' # http://localhost:8080/
    
    def handle_response(self, response):
        self.response = response
        if self.response.ok:
            ct = self.response.headers['Content-Type']
            if ct.startswith('text'):                       msg, mtype = self.response.text, __class__.MTypes.TEXT
            elif ct.startswith('application/json'):         msg, mtype = self.response.json(), __class__.MTypes.JSON
            else:                                           msg, mtype = self.response.content, __class__.MTypes.BYTES
        else:                                               msg, mtype = (f'[{self.response.status_code}]'), __class__.MTypes.NONE
        return msg, mtype, True


    def send_args(self, **kwargs): 
        try:                    msg, mtype, mcode = self.handle_response(requests.post(url=self.server,data=kwargs))
        except:                 msg, mtype, mcode = f"[SEND_ERROR:ARGS] cannot send ARGS to {self.server}", __class__.MTypes.ERROR, False
        return                  msg, mtype, mcode

    def send_json(self, json_object): 
        try:                    msg, mtype, mcode = self.handle_response(requests.post(url=self.server,json=json_object))
        except:                 msg, mtype, mcode = f"[SEND_ERROR:JSON] cannot send JSON to {self.server}", __class__.MTypes.ERROR, False
        return                  msg, mtype, mcode

    def send_buffers_by_name(self, seek0, buffers:dict):
        try:
            assert (len(buffers)>0) 
            if seek0: 
                for b in buffers.values(): b.seek(0)
            msg, mtype, mcode = self.handle_response(requests.post(url=self.server,files=buffers))
        except AssertionError:  msg, mtype, mcode = f"[SEND_ERROR:BUFFER] NO BUFFERS provided", __class__.MTypes.ERROR, False
        except:                 msg, mtype, mcode = f"[SEND_ERROR:BUFFER] cannot send BUFFERS to {self.server}", __class__.MTypes.ERROR, False
        return                  msg, mtype, mcode


    def send_files_by_handles(self, handles:list): return self.send_buffers_by_name(False, {h.name:h for h in handles})

    def send_files_by_named_handels(self, handles:dict): return self.send_buffers_by_name(False, handles)

    def send_files_by_path(self, paths:list):
        handles = []
        try:
            handles += [open(path, 'r') for path in paths]
            msg, mtype, mcode = self.send_files_by_handles(handles)
        except: msg, mtype, mcode = f"[SEND_ERROR:FILE] Files could not be read", __class__.MTypes.ERROR, False
        finally:
            for h in handles: 
                try:h.close()
                except:pass
        return msg, mtype, mcode

    def send_files_by_named_paths(self, paths:dict): 
        handles = {}
        try:
            handles += {name:open(path, 'r') for name,path in paths.items()}
            msg, mtype, mcode = self.send_files_by_named_handels(handles)
        except: msg, mtype, mcode = f"[SEND_ERROR:FILE] Files could not be read", __class__.MTypes.ERROR, False
        finally:
            for hk, hv in handles.items():
                try: hv.close()
                except: pass
        return   msg, mtype, mcode


    def close(self):
        try: self.response.close()
        except:pass

HelpModule=\
f"""

# global scope functions
# python -m known.api --user=user.py
def handle_args(host, args): return ""
def handle_json(host, data): return ""
def handle_file(host, files): return ""


class UserModuleInstance:
    # creates an instance of this class, init is called
    # python -m known.api --user=user.py --object=UserModuleInstance --callable=1
    def __init__(self): pass
    def handle_args(self, host, args): return ""
    def handle_json(self, host, data): return ""
    def handle_file(self, host, files): return ""

class UserModuleStatic:
    # no instance created, assumed to be a simple object
    # python -m known.api --user=user.py --object=UserModuleStatic --callable=0
    def handle_args(host, args): return ""
    def handle_json(host, data): return ""
    def handle_file(host, files): return ""




"""
# ACCEPTED 		        : 202
# ALREADY_REPORTED 		: 208
# BAD_GATEWAY 		    : 502
# BAD_REQUEST 		    : 400  #<------- used - indicates use of other request type than GET or POST
# CONFLICT 		        : 409
# CONTINUE 		        : 100
# CREATED 		        : 201
# EARLY_HINTS 		    : 103
# EXPECTATION_FAILED 		: 417
# FAILED_DEPENDENCY 		: 424
# FORBIDDEN 		        : 403
# FOUND 		            : 302
# GATEWAY_TIMEOUT 		: 504
# GONE 		            : 410
# HTTP_VERSION_NOT_SUPPORTED 		: 505
# IM_A_TEAPOT 		    : 418
# IM_USED 		        : 226
# INSUFFICIENT_STORAGE 		: 507
# INTERNAL_SERVER_ERROR 		: 500
# LENGTH_REQUIRED 		: 411
# LOCKED 		: 423
# LOOP_DETECTED 		: 508
# METHOD_NOT_ALLOWED 		: 405
# MISDIRECTED_REQUEST 		: 421
# MOVED_PERMANENTLY 		: 301
# MULTIPLE_CHOICES 		: 300
# MULTI_STATUS 		: 207
# NETWORK_AUTHENTICATION_REQUIRED 		: 511
# NON_AUTHORITATIVE_INFORMATION 		: 203
# NOT_ACCEPTABLE 		: 406
# NOT_EXTENDED 		: 510
# NOT_FOUND 		: 404 #<------ used - indicates the return_object was not of correct type (str, dict, list)
# NOT_IMPLEMENTED 		: 501
# NOT_MODIFIED 		: 304
# NO_CONTENT 		: 204
# OK 		: 200
# PARTIAL_CONTENT 		: 206
# PAYMENT_REQUIRED 		: 402
# PERMANENT_REDIRECT 		: 308
# PRECONDITION_FAILED 		: 412
# PRECONDITION_REQUIRED 		: 428
# PROCESSING 		: 102
# PROXY_AUTHENTICATION_REQUIRED 		: 407
# REQUESTED_RANGE_NOT_SATISFIABLE 		: 416
# REQUEST_ENTITY_TOO_LARGE 		: 413
# REQUEST_HEADER_FIELDS_TOO_LARGE 		: 431
# REQUEST_TIMEOUT 		: 408
# REQUEST_URI_TOO_LONG 		: 414
# RESET_CONTENT 		: 205
# SEE_OTHER 		: 303
# SERVICE_UNAVAILABLE 		: 503
# SWITCHING_PROTOCOLS 		: 101
# TEMPORARY_REDIRECT 		: 307
# TOO_EARLY 		: 425
# TOO_MANY_REQUESTS 		: 429
# UNAUTHORIZED 		: 401
# UNAVAILABLE_FOR_LEGAL_REASONS 		: 451
# UNPROCESSABLE_ENTITY 		: 422
# UNSUPPORTED_MEDIA_TYPE 		: 415
# UPGRADE_REQUIRED 		: 426
# USE_PROXY 		: 305
# VARIANT_ALSO_NEGOTIATES 		: 506



# #def query(url, **params): return requests.get(url, params) # works for string key-value pairs only
# #response=query(server, name='Nelson', age='31')
# #vb.show(response)


# # # use any one of data or json
# # response = requests.post(
# #     url=server,
# #     data={'key': 'value'}, # A dictionary, list of tuples, bytes or a file object to send to the specified url
# #     )
# # vb.show(response)


# """

# """ GENERIC METHOD CALLS

# # response = requests.post(
# #     url=server,
# #     data={key: value}, # A dictionary, list of tuples, bytes or a file object to send to the specified url
# #     json={key: value}, # A JSON object to send to the specified url
# #     files={key: value}, # File handles (opened)
# #     timeout=0.0,
# #     )
# # response = requests.get(
# #     url=server,
# #     params={key: value}, # A dictionary, list of tuples, bytes or a file object to send to the specified url
# #     timeout=0.0,
# #     )
# """
