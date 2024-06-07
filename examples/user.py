class UserModule:
    # reponse must be a string, list or dict

    def __init__(self): pass

    def handle_args(self, host, args):
        print(f'\n--------------> Incoming args data from {host}\n{args}')
        return "Got Args Data"

    def handle_json(self, host, data):
        print(f'\n--------------> Incoming json data from {host}\n{data}')
        return "Got Json Data"
    
    def handle_file(self, host, files):
        print(f'\n--------------> Incoming file data from {host}\n{files}')
        return "Got Files Data"

    def handle_msg(self, body):
        print(f'\n--------------> Incoming msg data from broker\n{body}')
        #return "Got Broker Msg Data"