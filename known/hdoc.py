__doc__=r"""
:py:mod:`known/hdoc.py`
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Block:
    r""" Represents an HTML Block that has:
        -> tag              an HTML tag (string) 
        -> attributes       a dict of HTML attributes (dict)
        -> members          a list of member HTML blocks (list)
        -> close            a bool value indicating if the tag should be closed (bool)
                            - not all tag require closing
                            - only a closed tag can have members
    """

    def __init__(self, tag): self.tag, self.attributes, self.members, self.close = f'{tag}', {}, [], True

    def __len__(self): return len(self.members)

    def __getitem__(self, index): return self.members[index]

    def __bool__(self): return self.close

    def __invert__(self): # ~ operator to toggle 'close' property
        self.close = not self.close
        return self
    
    # to add members and set attributes
    def __call__(self, *args, **kwargs): 
        r""" Call to add members and set attributes
        `args`      represent the members (HTML Blocks) to be appended to self.members list
        `kwargs`    represent the attributes to set on this HTML block
        """
        if self.close: # add members only if it is not an inline tag i.e., it is closed tag
            for other in args:
                if other is not None: self.members.append(other)

        for attribute,value in kwargs.items(): # add attributes to this block
            if attribute in self.attributes:
                if value is None:   del self.attributes[attribute] # remove that attribute
                else:               self.attributes[attribute] = f'{value}' # overwite that attribute
            else:
                if value is not None: self.attributes[attribute] = f'{value}' # wite that attribute
        return self

    # to add members 
    def __add__(self, other): 
        r""" alternative way to add members (appends) """
        if (self.close) and (other is not None): 
            self.members.append(other)
        return self
    
    # to set one member - supposed to be used on tags like <title> that have only one member that is usually text
    def __mul__(self, other): 
        r""" alternative way to add members (sets) """
        self.members.clear()
        return self + other

    # to render code
    def __str__(self): # returns the string repsentation - html code
        atr = (' ' + ' '.join([f'{k.lower()}="{v}"' for k,v in self.attributes.items()])) if self.attributes else '' # tag attributes are converted to lowercase
        tag_open =  f'<{self.tag}{atr}>'
        members =   ''.join([ f'{m}' for m in self.members ])
        tag_close = f'</{self.tag}>' if self.close else ''
        return f'{tag_open}{members}{tag_close}'

    # human readable representation
    def __repr__(self) -> str: return f'[{self.tag}] {("<>" if self.close else "--")} :: {len(self.members)} member(s), {len(self.attributes)} attribute(s)'

    # to save rendered code
    def __gt__(self, other):
        with open(other, 'w') as f: f.write(f'{self}') # dumps code to a file - will overwite existing file

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

"""
CSS can be added to HTML documents in 3 ways:

Inline              -  by using the style attribute inside HTML elements <---------------- Highest Priority
Internal (style)    -  by using a <style> element in the <head> section
External (css)      -  by using a <link> element to link to an external CSS file

The most common way to add CSS, is to keep the styles in external CSS files.
"""

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# returns an Inline style string from attributes stored in a dict, 
def GetStyle(d): return (';'.join([f'{k}:{v}' for k,v in d.items()]) + ";")

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

import os

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

class Document(Block):
    r""" Inherits from Block - encapsulates an HTML page as a block object
    # All the paths must be relative to the base directory where document is saved
    """

    def __init__(self, path, title, style, css) -> None:
        super().__init__('html')
        self(
                Block('head')( #0
                    ~Block('meta')(charset="UTF-8"), #00
                    ~Block('link')(rel='stylesheet', href=f'{css}'), #01
                    Block('style')(f'{style}'), #02
                    Block('title')(f'{title}'), #03
                ),
                Block('body') #1
            ) 
        if not path.lower().endswith(".html"): path = path + ".html"
        self.FilePath = os.path.abspath(path)
        self.DirPath = os.path.dirname(self.path)
        #self.filename = os.path.basename(self.path)
        #self.filealias = self.filename[0:-5]
        #self.fileext = self.filename[-4:]


    @property
    def css(self): return self[0][1].attributes['href']
    @css.setter
    def css(self, value): self[0][1](href=f'{value}')

    @property
    def style(self): return self[0][2][0]
    @style.setter
    def style(self, value): self[0][2]*value

    @property
    def title(self): return self[0][3][0]
    @title.setter
    def title(self, value): self[0][3]*value

    @property
    def body(self): return self[1]

    # get the relative path wrt to the directory of the self.path
    def __mod__(self, path): return os.path.relpath(path, self.DirPath)

    def save(self): self > self.path


#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

r"""
Known blocks

>> divisons     Sections - documents is body made of multiple divs
>> heading      
>> paragraph
>> code/pre blocks
>> lists
>> multimedia   images/audio/video
>> tables
>> 
"""