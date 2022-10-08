#-----------------------------------------------------------------------------------------------------
# mdlog/mu.py
#-----------------------------------------------------------------------------------------------------
from .core import LOGGER
#-----------------------------------------------------------------------------------------------------

class MarkUpLogger(LOGGER):

    def __init__(self, log_dir, log_file):
        super().__init__(log_dir, log_file, log_extension='html')



    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """ Default codeblock creation for output redirection """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    def codeblock_(self):
        self.tag_('pre') 
        self.tag_('code', nl=True)
    def _codeblock(self):
        self._tag('code') 
        self._tag('pre', nl=True)


    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """ Markup Elements """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    

    # markup can use direct tags for lists, uris and images
    def tag_(self, tag, nl=False, **atr):
        if atr:
            a= ' ' + ' '.join([ '{}="{}"'.format(_name, self.ascape(_value))  for _name, _value in atr.items() ])
        else:
            a= ''
        self.f.write('<' + tag + a + '>')
        self.f.write('\n') if nl else None
    def _tag(self, tag, nl=False):
        self.f.write('</' + tag +'>')
        self.f.write('\n') if nl else None

    # Strings/Chars (Lines and Paragraphs)
    def ln_(self, *msg, sep=' '):
        """ [line] put all items in input to file - seperated by sep, dont end the block """
        for m in msg:
            self.write(m)
            self.f.write(sep)
    def _ln(self, *msg, sep=' '):
        """ [line] put all items in input to file - seperated by sep, dont end the block """
        for m in msg[0:-1]:
            self.write(m)
            self.f.write(sep)
        self.write(msg[-1])

    # headings
    def h(self, n, msg, **atr):
        """ insert heading of size 'n' : <h1> to <h6> """
        self.tag_('h'+str(n), **atr)
        self.write(msg)
        self._tag('h'+str(n), nl=True)

    def hr(self):
        """ insert horizontal rule : <hr> """
        self.f.write('<hr>\n')

    # Lists
    TAG_OLS, TAG_ULS, TAG_LIS = 'ol', 'ul', 'li'
    #--------------------------------------------------------------------------
    def ll(self, L, order=False):
        """ writes items of list L - no nesting """
        tag = self.TAG_OLS if order else self.TAG_ULS
        self.tag_(tag, nl=True)
        for l in L: # for i,l in enumerate(L):
            self.tag_(self.TAG_LIS)
            self.write(l)
            self._tag(self.TAG_LIS, nl=True)
        self._tag(tag, nl=True)
    def ll2(self, H, L, outer_order=False, inner_order=True):
        """ writes items of 2 List with H containing parent and L containing sublist """
        outer_tag = self.TAG_OLS if outer_order else self.TAG_ULS
        #inner_tag = self.OLS if inner_order else self.ULS
        self.tag_(outer_tag, nl=True)
        for h,l in zip(H,L):
            self.tag_(self.TAG_LIS)
            self.write(h)
            self._tag(self.TAG_LIS)
            self.ll(l, order=inner_order)
        self._tag(outer_tag, nl=True)
    def lld(self, D, outer_order=False, inner_order=True):
        """ writes items of 2 List with D.keys containing parent and D.values containing sublist """
        self.ll2(list(D.keys()),list(D.values()), outer_order, inner_order)

    
    TAG_TABLE, TAG_TR, TAG_TH, TAG_TD = 'table', 'tr', 'th', 'td'
    # Tables (from iterables)
    def _dump_header(self, header):
        """ helper method, dont use directly """
        self.tag_(self.TAG_TR, nl=True)
        for h in header:
            self.tag_(self.TAG_TH)
            self.write(h)
            self._tag(self.TAG_TH)
        self._tag(self.TAG_TR, nl=True)
    #--------------------------------------------------------------------------
    def t_(self, header):
        """ opens a table """
        self.tag_(self.TAG_TABLE, nl=True)
        self._dump_header(header)
    def r(self, R):
        """ write full row at once """
        self.r_()
        for i in R:
            self.ri(i)
        self._r()
    def r_(self):
        """ open row - for writing unit by unit """
        self.tag_(self.TAG_TR, nl=True)
    def ri(self, i):
        """ write a unit (row item) """
        self.tag_(self.TAG_TD)
        self.write(i)
        self._tag(self.TAG_TD)
    def _r(self):
        """ close a row """
        self._tag(self.TAG_TR, nl=True)
    def _t(self):
        """ close a table """
        self._tag(self.TAG_TABLE, nl=True)
    #--------------------------------------------------------------------------
    def rt(self, header, R):
        """ [Row table] - table with header and each item in R defining one full Row """
        self.t_(header)
        for i in R:
            self.r( i )
        self._t()
    def mrt(self, header, *R):
        """ [Multi-Row table] - table with header and each item in MR defining one full Row
            - auto generates header if its none """
        header = range(len(R[0])) if header is None else header
        self.rt(header, R)
    def ct(self, header, C):
        """ [Col table] - table with header and each item in C defining one full Col """ 
        self.t_(header)
        rows, cols = len(C[0]), len(C)
        for i in range(rows):
            self.r_()
            for j in range(cols):
                self.ri(C[j][i])
            self._r()
        self._t()
    def mct(self, header, *C):
        """ [Multi-Col table] - table with header and each item in C defining one full Col 
        - auto generates header if its none """
        header = range(len(C)) if (header is None) else header
        self.t_(header)
        for cc in zip(*C):
            self.r(cc)
        self._t()
    def dct(self, D):
        """ [Dict col table] - table with header as D.keys() and each item in D.values() defining one full Col 
        - directly calls self.ct with two args - keys and value - as each column """
        self.ct(list(D.keys()), list(D.values()))
    def drt(self, D, hkey='Key', hval='Val'):
        """ [Dict row table] - table with 2-cols (hkey and hval) from a dict 
        - directly calls self.rt with two args - keys and value - as each column """
        self.rt([hkey, hval], D.items() )

    # preformated-text (from dict)
    def pfd(self, D, caption=""):
        """ [pre-formated Dict] - pre-format text from a dict """
        self.c_()
        self.f.write("=-=-=-=-==-=-=-=-=\n"+caption+"\n=-=-=-=-==-=-=-=-=\n")
        for k,v in D.items():
            self.f.write(str(k) + " : " + str(v) + '\n')
        self.f.write("=-=-=-=-==-=-=-=-=\n")
        self._c()


""" NOTE:
    * Author:           Nelson.S

"""
#-----------------------------------------------------------------------------------------------------
