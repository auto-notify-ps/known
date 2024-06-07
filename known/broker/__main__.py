__doc__="""
RabbitMQ for Queue based messaging
"""

#-----------------------------------------------------------------------------------------
from sys import exit
if __name__!='__main__': exit(f'[!] can not import {__name__}.{__file__}')
#-----------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------
# imports ----------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
import os, argparse, datetime, sys, importlib, importlib.util

#PYDIR = os.path.dirname(__file__) # script directory of __main__.py
try:
    import pika
except: exit(f'[!] The required packages missing:\tpika\n  ⇒ pip install pika')

class DefaultUserModule:
    def handle_msg(self, body):        print(f"handle_args({body})")

parser = argparse.ArgumentParser()
parser.add_argument('--user', type=str, default='', help="path of main user module file")
parser.add_argument('--object', type=str, default='', help="path of main user module")
parser.add_argument('--callable', type=int, default=0, help="if true, calls the class to create a new instance (with no args)")
parser.add_argument('--handle_msg', type=str, default='handle_msg', help="path of main user module file")
parser.add_argument('--host', type=str, default='', help="host ip-address")
parser.add_argument('--queue', type=str, default='default', help="queue name")
parsed = parser.parse_args()

if not parsed.host: exit(f'Invalid host address')
if not parsed.queue: exit(f'Invalid queue name')
if parsed.user:
    # user-module
    USER_MODULE_FILE_PATH = os.path.abspath(parsed.user)
    print(f'↪ Loading user-module from {USER_MODULE_FILE_PATH}')
    if not os.path.isfile(USER_MODULE_FILE_PATH): exit(f'Invalid user-module file @ {USER_MODULE_FILE_PATH}')
    try: 
        # from https://stackoverflow.com/questions/67631/how-can-i-import-a-module-dynamically-given-the-full-path
        user_module_spec = importlib.util.spec_from_file_location("", USER_MODULE_FILE_PATH)
        user_module = importlib.util.module_from_spec(user_module_spec)
        user_module_spec.loader.exec_module(user_module)
        print(f'↪ Imported user-module from {user_module.__file__}')
    except: exit(f'[!] Could import user-module "{USER_MODULE_FILE_PATH}"')
    if parsed.object:
        try:
            user_module = getattr(user_module, parsed.object)
            if bool(parsed.callable): user_module = user_module()
        except:
            exit(f'Could not load object {parsed.object}')
    USER_HANDLE_MSG = parsed.handle_msg
else:
    print(f'↪ [!] user-module not defined, using default.')
    USER_HANDLE_MSG = 'handle_msg'
    user_module = DefaultUserModule()


if not hasattr(user_module, USER_HANDLE_MSG): exit(f'[!] MSG Handler Method not found {user_module}.{USER_HANDLE_MSG}')
#from known.basic import Verbose as vb
# ------------------------------------------------------------------------------------------
# application setting and instance
# ------------------------------------------------------------------------------------------

# establish connection
try:
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=parsed.host))
    channel = connection.channel()
    channel.queue_declare(queue=parsed.queue)
except: exit(f'Connection cannot be established with {parsed.host}@{parsed.queue}')
def callback(ch, method, properties, body): return getattr(user_module, USER_HANDLE_MSG)(body)
channel.basic_consume(queue=parsed.queue, on_message_callback=callback, auto_ack=True)
start_time = datetime.datetime.now()
print('◉ start server @ [{}]'.format(start_time))
try: channel.start_consuming()
except KeyboardInterrupt:
    try: sys.exit(0)
    except SystemExit: os._exit(0)

#<-------------------DO NOT WRITE ANY CODE AFTER THIS
end_time = datetime.datetime.now()
print('◉ stop server @ [{}]'.format(end_time))
print('◉ server up-time was [{}]'.format(end_time - start_time))


