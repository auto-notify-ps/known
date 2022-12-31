
class JSON:
    try:
        import json
    except:
        print(f'class: [store.JSON] requires package: [json] which is not available.')
        
    def save(path, data_dict):
        """ saves a dict to disk in json format """
        with open(path, 'w') as f:
            f.write(__class__.json.dumps(data_dict, sort_keys=False, indent=4))
        return path
    def load(path):
        """ returns a dict from a json file """
        data_dict = None
        with open(path, 'r') as f:
            data_dict = __class__.json.loads(f.read())
        return data_dict
