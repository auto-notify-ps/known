#-----------------------------------------------------------------------------------------------------
# mdlog/md.py
#-----------------------------------------------------------------------------------------------------
from .core import LOGGER
#-----------------------------------------------------------------------------------------------------

class MarkDownLogger(LOGGER):
    
    def __init__(self, log_dir, log_file, uri_title_quote=False):
        super().__init__(log_dir, log_file, log_extension='md')
        self.uri_title_start, self.uri_title_end = \
            (('"', '"') if uri_title_quote else ('(', ')'))

    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """ Default codeblock creation for output redirection """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
    def codeblock_(self):
        self.c_()
    def _codeblock(self):
        self._c(False)




    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~    
    """ Markdown Elements """
    # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 


    # Strings/Chars (Lines and Paragraphs)
    def ln(self, *msg, sep=' ', term=False):
        """ [line] put all items in input to file - seperated by sep, if term=True, ends the block """
        self._ln(*msg, sep=sep) if term else self.ln_(*msg, sep=sep)
    def ln_(self, *msg, sep=' '):
        """ [line] put all items in input to file - seperated by sep, dont end the block """
        for m in msg:
            self.write(m)
            self.f.write(sep)
    def _ln(self, *msg, sep=' '):
        """ [line] put all items in input to file - seperated by sep, and end the block """
        self.ln_(*msg, sep=sep)
        self.f.seek( self.f.tell() - 1 )
        self.f.write('\n\n')

    # Heading 
    HEADINGS = [' ', '# ', '## ', '### ', '#### ', '##### ', '###### ']
    def h(self, n, msg):
        """ insert heading of size 'n' : <h1> to <h6> """
        self.f.write(self.HEADINGS[n])
        self.write(msg)
        self.f.write('\n\n')

    # Horizontal Rule
    HR = '___'
    def hr(self):
        """ insert horizontal rule : <hr> """
        self.f.write(self.HR + '\n\n')

    # Block (Text and Quotes)
    PARAS, QUOTES = '', '>'
    def b_(self, quote=False):
        """ open  text/quote block """
        self.f.write(self.QUOTES if quote else self.PARAS)
    def _b(self):
        """ close text/quote block """
        self.f.write('\n\n')
    #--------------------------------------------------------------------------
    def b(self, *msg, quote=False):
        """ write text/quote block """
        self.f.write(self.QUOTES if quote else self.PARAS)
        self.write(*msg)
        self.f.write('\n\n')

    # Code Block
    CODES = '```'
    def c_(self):
        """ open code block """
        self.f.write(self.CODES + '\n')
    def _c(self, put_nl=True):
        """ close code block, if put_nl is true, inserts a newline before closing block """
        self.f.write('\n') if put_nl else None
        self.f.write(self.CODES + '\n\n')
    #--------------------------------------------------------------------------
    def c(self, *msg, put_nl=True):
        """ write a code block, if put_nl is true, inserts a newline before closing block"""
        self.f.write(self.CODES + '\n')
        self.write(*msg)
        self.f.write( ('\n' if put_nl else '') + self.CODES + '\n\n')

    # URI (urls and images)
    URILINK, URIIMG = '[', '!['
    def uri(self, caption, url, title=None, image=False, inline=False, loc=False):
        """ inserts a url as a link or an image (uses relative path if 'loc' is True) """
        if image:
            self.f.write(self.URIIMG)
            self.f.write(caption)
        else:
            self.f.write(self.URILINK)
            self.write(caption)
        uri = (self.loc(url) if loc else url) #<---- relative path please
        self.f.write(']'+'(' + uri +  ((' ' + self.uri_title_start + title + self.uri_title_end)  if title else '') + ')')
        self.f.write('\n\n') if (not inline) else None

    # Lists
    OLS, ULS = '1. ', '* '
    def ll_(self, order=False):
        """ opens a list """
        self.ostrL = []
        self.ostrL.append(self.OLS if order else self.ULS)
        self.ostrP = 0
        self.ul_tab_ind = 0
        self.ul_pre_str=''
    def _ll(self):
        """ close a list """
        self.f.write('\n')
    def l_(self, order=False):
        """ (+1) indent -> sub-list """
        self.ostrL.append(self.OLS if order else self.ULS)
        self.ostrP += 1
        self.ul_tab_ind+=1
        self.ul_pre_str+='\t'
    def _l(self):
        """ (-1) indent -> super-list """
        del self.ostrL[-1]
        self.ostrP -= 1
        self.ul_tab_ind-=1
        self.ul_pre_str = self.ul_pre_str[0:-1] if (self.ul_tab_ind>0) else ''
    def li_(self):
        """ opens a list item """
        self.f.write(self.ul_pre_str + self.ostrL[self.ostrP])
    def _li(self):
        """ closes a list item """
        self.f.write('\n')
    def li(self, *msg):
        """ short-hand for _li and li_ """
        self.f.write(self.ul_pre_str + self.ostrL[self.ostrP]) 
        self.write(*msg)
        self.f.write('\n')
    #--------------------------------------------------------------------------
    def ll(self, L, order=False, level=0):
        """ writes items of list L - no nesting """
        ostr = self.OLS if order else self.ULS
        lstr = ""
        for _ in range(level):
            lstr+='\t'
        for l in L: # for i,l in enumerate(L):
            self.f.write(lstr + ostr)
            self.write(l)
            self.f.write('\n')
        self.f.write('\n')
    def ll2(self, H, L, outer_order=False, inner_order=True, level=0, in_sep=True):
        """ writes items of 2 List with H containing parent and L containing sublist """
        ostr = self.OLS if outer_order else self.ULS
        lstr = ""
        for _ in range(level):
            lstr+='\t'
        for h,l in zip(H,L):
            self.f.write(lstr + ostr)
            self.write(h)
            self.f.write('\n')
            self.ll(l, order=inner_order, level=level+1)
            self.f.write('\n') if in_sep else None
        self.f.write('\n')
    def lld(self, D, outer_order=False, inner_order=True, level=0, in_sep=True):
        """ writes items of 2 List with D.keys containing parent and D.values containing sublist """
        self.ll2(list(D.keys()),list(D.values()), outer_order, inner_order, level, in_sep)

    # Tables (from iterables)
    TAB_ALIGN_LEFT, TAB_ALIGN_CENTER, TAB_ALIGN_RIGHT = ':------', ':------:', '------:'
    def _dump_header(self, header, align):
        """ helper method, dont use directly """
        self.f.write('|')
        for h in header:
            self.write(h)
            self.f.write('|')
        self.f.write('\n|'+ '|'.join([ a for a in align ]) +'|\n' )
    #--------------------------------------------------------------------------
    def t_(self, header, align):
        """ opens a table """
        if type(align) is str:
            self._dump_header(header, [align for _ in range(len(header))])
        else:
            self._dump_header(header, align)
    def r(self, R):
        """ write full row at once """
        self.f.write('|')
        for i in R:
            self.write(i)
            self.f.write('|')
        self.f.write('\n')
    def r_(self):
        """ open row - for writing unit by unit """
        self.f.write('|')
    def ri(self, i):
        """ write a unit (row item) """
        self.write(i)
        self.f.write('|')
    def _r(self):
        """ close a row """
        self.f.write('|\n')
    def _t(self):
        """ close a table """
        self.f.write('\n')
    #--------------------------------------------------------------------------
    def rt(self, header, align, R):
        """ [Row table] - table with header and each item in R defining one full Row """
        self.t_(header, align)
        for i in R:
            self.r( i )
        self._t()
    def mrt(self, header, align, *R):
        """ [Multi-Row table] - table with header and each item in MR defining one full Row
            - auto generates header if its none """
        header = range(len(R[0])) if header is None else header
        self.rt(header, align, R)
    def ct(self, header, align, C):
        """ [Col table] - table with header and each item in C defining one full Col """ 
        self.t_(header, align)
        rows, cols = len(C[0]), len(C)
        for i in range(rows):
            self.r_()
            for j in range(cols):
                self.ri(C[j][i])
            self._r()
        self._t()
    def mct(self, header, align, *C):
        """ [Multi-Col table] - table with header and each item in C defining one full Col 
        - auto generates header if its none """
        header = range(len(C)) if (header is None) else header
        self.t_(header, align)
        for cc in zip(*C):
            self.r(cc)
        self._t()
    def dct(self, align, D):
        """ [Dict col table] - table with header as D.keys() and each item in D.values() defining one full Col 
        - directly calls self.ct with two args - keys and value - as each column """
        self.ct(list(D.keys()), align, list(D.values()))
    def drt(self, align, D, hkey='Key', hval='Val'):
        """ [Dict row table] - table with 2-cols (hkey and hval) from a dict 
        - directly calls self.rt with two args - keys and value - as each column """
        self.rt([hkey, hval], align, D.items() )

    # preformated-text (from dict)
    def pfd(self, D, caption=""):
        """ [pre-formated Dict] - pre-format text from a dict """
        self.c_()
        self.f.write("=-=-=-=-==-=-=-=-=\n"+caption+"\n=-=-=-=-==-=-=-=-=\n")
        for k,v in D.items():
            self.f.write(str(k) + " : " + str(v) + '\n')
        self.f.write("=-=-=-=-==-=-=-=-=\n")
        self._c()

#-----------------------------------------------------------------------------------------------------
# Foot-Note:
""" NOTE: https://daringfireball.net/projects/markdown/syntax

    Paragraph <p>
        paragraph is simply one or more consecutive lines of text, separated by one or more blank lines. 
        (A blank line is any line that looks like a blank line — a line containing nothing but spaces or tabs is considered blank.) 
        Normal paragraphs should not be indented with spaces or tabs.

    Headings <h?>
        To create an atx-style header, you put 1-6 hash marks (#) at the beginning of the line 
        — the number of hashes equals the resulting HTML header level.
        uses: self._P_PRE_H, self._P_POST_H

    Blockquotes 
        are indicated using email-style ‘>’ angle brackets.

    Block level tags and Inline HTML
        
        For any markup that is not covered by Markdown’s syntax, you simply use HTML itself. 
        There’s no need to preface it or delimit it to indicate that you’re switching from Markdown to HTML; you just use the tags.

        The only restrictions are that block-level HTML elements — e.g. <div>, <table>, <pre>, <p>, etc. 
        — must be separated from surrounding content by blank lines, and the start and end tags of the block should not be indented with tabs or spaces. 
        Markdown is smart enough not to add extra (unwanted) <p> tags around HTML block-level tags.

        When you do want to insert a <br /> break tag using Markdown, you end a line with two or more spaces, then type return.
        Yes, this takes a tad more effort to create a <br />, but a simplistic “every line break is a <br />” rule wouldn’t work for Markdown. 
        Markdown’s email-style blockquoting and multi-paragraph list items work best — and look better — when you format them with hard breaks.

    Lists
        Markdown supports ordered (numbered) and unordered (bulleted) lists.
        Unordered lists use asterisks, pluses, and hyphens — interchangably — as list markers:
        Ordered lists use numbers followed by periods.
        It’s important to note that the actual numbers you use to mark the list have no effect on the HTML output Markdown produces.
        Lists for fixed size iterables
        > size of list is known a priori
        > not suitable for long running loops

    Lists for time consuming long iterations
        > use open, close to specify begin and end of list
        > use  plus, minus for indendation(nesting)
        > use ll_ for wiritng lists
        > lchar is '*' for unordered list and '0' for ordered list

    Links
        This is [an example](http://example.com/ "Title") inline link.
            
        if you’re referring to a local resource on the same server, you can use relative paths:
            See my [About](/about/) page for details.   

        Reference-style links use a second set of square brackets, inside which you place a label of your choosing to identify the link:
            This is [an example][id] reference-style link.
        
        You can optionally use a space to separate the sets of brackets:
            This is [an example] [id] reference-style link.
        Then, anywhere in the document, you define your link label like this, on a line by itself:
            [id]: http://example.com/  "Optional Title Here"
        
        That is:
        > Square brackets containing the link identifier (optionally indented from the left margin using up to three spaces);
        > followed by a colon;
        > followed by one or more spaces (or tabs);
        > followed by the URL for the link;
        > optionally followed by a title attribute for the link, enclosed in double or single quotes, or enclosed in parentheses.
        
        The following three link definitions are equivalent:

            [foo]: http://example.com/  "Optional Title Here"
            [foo]: http://example.com/  'Optional Title Here'
            [foo]: http://example.com/  (Optional Title Here)


        Using header as withing doc link
        The #header-IDs are generated from the content of the header according to the following rules:
            All text is converted to lowercase.
            All non-word text (e.g., punctuation, HTML) is removed.
            All spaces are converted to hyphens.
            Two or more hyphens in a row are converted to one.
            If a header with the same ID has already been generated, a unique incrementing number is appended, starting at 1.


    EMPHASIS
        Markdown treats asterisks (*) and underscores (_) as indicators of emphasis. 
        Text wrapped with one * or _ will be wrapped with an HTML <em> tag; double *’s or _’s will be wrapped with an HTML <strong> tag. 
        E.g., this input:
            *single asterisks*, _single underscores_            for em
            **double asterisks**, __double underscores__        for strong

        these not need extra functions dont 

        CODE
        To indicate a span of code, wrap it with backtick quotes (`). 

        Unlike a pre-formatted code block, a code span indicates code within a normal paragraph. For example:
            Use the `printf()` function.
        will produce:
            <p>Use the <code>printf()</code> function.</p>

        To include a literal backtick character within a code span, you can use multiple backticks as the opening and closing delimiters:
            ``There is a literal backtick (`) here.``
        which will produce this:
            <p><code>There is a literal backtick (`) here.</code></p>
        
        The backtick delimiters surrounding a code span may include spaces — one after the opening, one before the closing. 
        This allows you to place literal backtick characters at the beginning or end of a code span:
            A single backtick in a code span: `` ` ``
            A backtick-delimited string in a code span: `` `foo` ``

    IMAGES
        Admittedly, it’s fairly difficult to devise a “natural” syntax for placing images into a plain text document format.

        Markdown uses an image syntax that is intended to resemble the syntax for links, allowing for two styles: inline and reference.

        Inline image syntax looks like this:

            ![Alt text](/path/to/img.jpg)

            ![Alt text](/path/to/img.jpg "Optional title")

        That is:
        An exclamation mark: !;
        followed by a set of square brackets, containing the alt attribute text for the image;
        followed by a set of parentheses, containing the URL or path to the image, and an optional title attribute enclosed in double or single quotes.
        Reference-style image syntax looks like this:

            ![Alt text][id]
        Where “id” is the name of a defined image reference. Image references are defined using syntax identical to link references:
            [id]: url/to/image  "Optional title attribute"
        As of this writing, Markdown has no syntax for specifying the dimensions of an image; 
        if this is important to you, you can simply use regular HTML <img> tags.


    escapers = ( '\\', '`', '*', '_', '{', '}', '[', ']', '(', ')', '#', '+', '-', '.', '!' )
    Markdown provides backslash escapes for the following characters:

        \   backslash
        `   backtick
        *   asterisk
        _   underscore
        {}  curly braces
        []  square brackets
        ()  parentheses
        #   hash mark
        +   plus sign
        -   minus sign (hyphen)
        .   dot
        !   exclamation mark

"""
""" NOTE:
    * Author:           Nelson.S
"""
#-----------------------------------------------------------------------------------------------------
