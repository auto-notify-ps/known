# known
__doc__=r"""
**known** is a collection of reusable python code. Use pip to install **known**

.. code-block:: console

   $ pip install known
   
The package is frequently updated by adding new functionality, make sure to have the latest version. To check version use

.. code-block:: console

   >>> import known
   >>> known.__version___
   >>> known.ll

:py:mod:`known/__init__.py`
"""
__version__ = '0.0.1'
print(f'known.{__version__} ~use known.ll for modules')

ll = f"""
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Available Modules under package known.{__version__} *

    import known

    # [basic]
    import known.basic as basic
    from known.basic import *
    from known.basic import Verbose as verb

    # [mailer]
    import known.mailer
    from known.mailer import MAIL

    # [hyper]
    import known.hyper
    from known.hyper import *
    from known.hyper import MarkDownLogger, MarkUpLogger

    # [ktorch]
    import known.ktorch as kt

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
from . import basic
from .basic import *



#---------------------------------------------------------
# meta information
#---------------------------------------------------------



#---------------------------------------------------------
