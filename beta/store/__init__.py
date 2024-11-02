import os
from zipfile import ZipFile

def ZipFiles(files:list, zip_path:str=None, **kwargs):
    r""" zips all (only files) in the list of file paths and saves at 'zip_path' """
    if isinstance(files, str): files= [f'{files}']
    if not files: return # no files provided to zip
    if not zip_path : zip_path = f'{files[0]}.zip' # if zip path not provided take it to be the first file
    if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  # append .zip to the end of path

    zipped = 0
    with ZipFile(zip_path, 'w', **kwargs) as zip_object:
        for path in files:
            if not os.path.isfile(path): continue
            zip_object.write(f'{path}')
            zipped+=1
    return zipped, zip_path

def ZipFolders(folders, zip_path:str=None, **kwargs):  
    r""" zip multiple folders into a single zip file 
    to zip a single folder with the same zip name - provide folder as a string and keep zip_path as none
    """    
    if isinstance(folders, str): folders= [f'{folders}']
    if not folders: return None, None# no folders provided to zip
    if not zip_path : zip_path = f'{folders[0]}.zip' # if zip path not provided take it to be the first folder
    if not zip_path.lower().endswith('.zip'): zip_path = f'{zip_path}.zip'  # append .zip to the end of path
    all_files = []
    for folder in folders:
        for root, directories, files in os.walk(folder): all_files.extend([os.path.join(root, filename) for filename in files])
    return ZipFiles(all_files, f'{zip_path}', **kwargs)

class Table:

    @staticmethod
    def CreateData(*columns):
        data = {None:[f'{col}' for col in columns]} # this is to make sure that col names are always on top
        return data

    @staticmethod
    def Create(columns:tuple, primary_key:str, cell_delimiter=',', record_delimiter='\n'):
        # should be called on a new object after init\
        table = __class__()
        table.data = __class__.CreateData(*columns)
        table.pk = primary_key
        table.pkat = table.data[None].index(table.pk)
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def ImportData(path, key_at, cell_delimiter, record_delimiter): 
        with open(path, 'r', encoding='utf-8') as f: 
            s = f.read()
            lines = s.split(record_delimiter)
            cols = lines[0].split(cell_delimiter) #<--- only if None:cols was added as a first entry (using Create method)
            data = {None:cols}
            if isinstance(key_at, str): key_at = cols.index(key_at)
            assert key_at>=0,f'Invlaid key {key_at}'
            for line in lines[1:]:
                if line:
                    cells = line.split(cell_delimiter)
                    data[f'{cells[key_at]}'] = cells
        return data
    
    @staticmethod
    def Import(path, key_at, cell_delimiter=',', record_delimiter='\n'): 
        table = __class__()
        table.data = __class__.ImportData(path, key_at, cell_delimiter, record_delimiter)
        if isinstance(key_at, str): key_at = table[None].index(key_at)
        table.pk = table.data[None][key_at]
        table.pkat = key_at
        table.cell_delimiter, table.record_delimiter = cell_delimiter, record_delimiter
        return table


    @staticmethod
    def ExportData(data, path, cell_delimiter, record_delimiter): 
        with open(path, 'w', encoding='utf-8') as f: 
            for v in data.values(): f.write(cell_delimiter.join(v)+record_delimiter)

    @staticmethod
    def Export(table, path): 
        __class__.ExportData(table.data, path, table.cell_delimiter, table.record_delimiter)

    # get row as dict
    def __call__(self, key): return {k:v for k,v in zip(self[None], self[key])}

    # get row as it is (list)
    def __getitem__(self, key): return self.data[key]

    # set row based on if its a dict or a list (note: key is irrelavant here)
    def __setitem__(self, key, row):
        assert len(row) == len(self[None]), f'Rows are expected to have length {len(self[None])} but got {len(row)}'
        if isinstance(row, dict):
            key = row[self.pk]
            if key is not None: self.data[f'{key}'] = [row[r] for r in self[None]]
        else: 
            key = row[self.pkat]
            if key is not None: self.data[f'{key}'] = list(row)

    # del row based on key
    def __delitem__(self, key):
        if key is not None: del self.data[key]

    def __contains__(self, key): return key in self.data

    # quick export > file
    def __gt__(self, other):__class__.ExportData(self.data, f'{other}', self.cell_delimiter, self.record_delimiter)

    # quick import < file
    def __lt__(self, other): self.data = __class__.ImportData(f'{other}', self.pkat, self.cell_delimiter, self.record_delimiter)

    # total number of rows
    def __len__(self): return len(self.data)-1



