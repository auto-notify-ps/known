# -----------------------------------------------------------------------------------------------------
# import datetime
# import os.path
# from math import floor
import numpy as np
import torch as tt
# -----------------------------------------------------------------------------------------------------

class COMMON_TORCH:

    def numel(shape): 
        """ returns no of total elements (or addresses) in a multi-dim array 
            Note: for torch tensor use Tensor.numel()"""
        return tt.prod(tt.tensor(shape)).item()

    def arange(shape, start=0, step=1, dtype=None): 
        """ returns arange for multi-dimensional array (reshapes) """
        return tt.arange(start=start, end=start+step*__class__.numel(shape), step=step, dtype=dtype).reshape(shape)

    def shares_memory(a, b) -> bool: 
        """ checks if two tensors share same underlying storage
            Note: this is different from Tensor.is_set_to(Tensor) function which checks shape as well"""
        return (a.storage().data_ptr() == b.storage().data_ptr())

class COMMON_NUMPY:

    def numel(shape): 
        """ returns no of total elements (or addresses) in a multi-dim array 
            Note: for torch tensor use Tensor.numel()"""
        return np.prod(np.array(shape))

    def arange(shape, start=0, step=1, dtype=None): 
        """ returns arange for multi-dimensional array (reshapes) """
        return np.arange(start=start, stop=start+step*__class__.numel(shape), step=step, dtype=dtype).reshape(shape)

    def shares_memory(a, b) -> bool: 
        """ checks if two numpy array share same underlying storage, alias for np.shares_memory() """
        return np.shares_memory(a,b)


# -----------------------------------------------------------------------------------------------------
""" MISC """
# -----------------------------------------------------------------------------------------------------


""" EINSUM v/s FOR-LOOPS

NOTE: use einsum-equivalance code

    # using numpy
    import numpy as np 
    x = np.random.rand(1,8,2,8,10)
    y = np.random.rand(8,10,10)
    # generate equivalent code for 
    #   np.einsum('nkctv,kvw->nctw', x, y)
    code = einsum_code_check('np', 'nkctv,kvw->nctw', x=x, y=y)
    print(code)
    

    # using torch
    import torch
    a = torch.rand((3, 3, 2))
    b = torch.rand((4, 2, 3))
    c = torch.rand((3, 4, 5))
    # generate equivalent code for 
    #   torch.einsum('iik,jki,ijl->kilj', a, b, c)
    code = einsum_code_check('torch', 'iik,jki,ijl->kilj', a=a, b=b, c=c)
    print(code)

    # run the printed code
"""

def einsum_code(module:str, equation:str, **doperands):
  """ Generates eisum code using for loops
      Note: 
        (1) module is a string that can be:
            -> 'torch' if using "import torch"
            -> 'np' if using "import numpy as np"
        (2) equation must be specified in explicit mode only - without use of elipses (...)
        (3) doperands is a dictionary of operands """
  # ----------------------------------------------------------
  EQU = '->'  # lhs-rhs seperator, must be specified in explicit mode
  SEP = ','   # input seperator
  assert EQU in equation, f"equation symbol {EQU} must be specified in explicit mode"

  # gets the operands
  operands =      list(doperands.values())
  operands_name = list(doperands.keys())

  # parse and validate the equation
  q = equation.split(EQU)
  assert len(q)==2, \
    f"equation [{q}] must have exactly 2 sides (LHS and RHS) found splits: [{len(q)}]"

  # get lhs and rhs
  lhs, rhs = q[0], q[1]
  assert lhs, f"LHS is empty"
  assert rhs, f"RHS is empty"

  # split lhs to get inputs
  lhss = lhs.split(SEP) 
  assert len(lhss) == len(operands), \
    f"no of input operands mis-match - Expected:[{len(lhss)}] Supplied:[{len(operands)}]"

  # for each of input - check if all its dims are specified in the equation
  for lin, opr in zip(lhss,operands): assert (len(lin)==len(opr.shape)), \
    f'dim mis-match on input [{lin}]. {len(lin)} != {len(opr.shape)}'

  # make symbol table
  symtab = {}
  for i,l in enumerate(lhss):
    for j,s in enumerate(l):
      if s in symtab:
        assert symtab[s] == operands[i].shape[j], \
          f'mis-match for operand ({i}) @ dim ({j}) :: {operands[i].shape[j]} != {symtab[s]} for symbol {s}'
      else:
        symtab[s] = operands[i].shape[j]
  #print(symtab)

  # check rhs
  assert len(rhs)<=len(symtab), \
    f'output {rhs} has more symbols ({len(rhs)}) than expected ({len(symtab)})'
  for k in symtab.keys(): assert rhs.rfind(k)==rhs.find(k), \
    f'symbol [{k}] occurs more than once in output [{rhs}]'

  # all checks are done, we have 'lhss', 'rhs' and 'symtab' to work with

  # find output shape and order of indix symbols
  out_shape = tuple([ symtab[s] for s in rhs ])
  out_index = ''
  for s in rhs: out_index+=(f'{s},')
  out_index = out_index[0:-1]

  # find order of input symbols for each input in lhss
  in_indexL = []
  for l in lhss:
    in_index = ''
    for s in l: in_index+=(f'{s},')
    in_indexL.append(in_index[0:-1])
  
  # make a printable indexed-operand string
  index_str = ''
  for op,il in zip(operands_name, in_indexL):
    index_str += f'{op}[{il}]*'
  index_str=index_str[0:-1]

  # define the einsum function code
  estr = f'def my_einsum():\n'
  tabs = '\t'
  estr += f'{tabs}out = {module}.zeros({out_shape})\n'
  for k,v in symtab.items():
    estr += f'{tabs}for {k} in range({v}):\n'
    tabs+='\t'
  estr += f'{tabs}out[{out_index}] += ({index_str})\n'
  estr += f'\treturn out\n'
  #print(estr)
  # ----------------------------------------------------------
  return estr
  # ----------------------------------------------------------

def einsum_code_check(module:str, equation:str, **doperands):
    """ Generates additional code for checking against inbuilt einsum """
    # ----------------------------------------------------------
    F = einsum_code(module, equation, **doperands)
    F += f'\ndef my_einsum_test():\n'
    F += f'\tM = my_einsum()\n'
    F += f'\tT = {module}.einsum("{equation}"'
    for a in list(doperands.keys()):
      F+= f', {a}'
    F+=')\n'
    F+=f'\tErr = {module}.sum({module}.abs(M-T))\n'
    F+=f'\tprint("\\n{module} einsum :", T.shape, "\\n", T)\n'
    F+=f'\tprint("\\nmy einsum :", M.shape, "\\n", M)\n'
    F+=f'\tprint("\\nerror =", Err)\n'
    F+=f'\nmy_einsum_test()\n'
    # ----------------------------------------------------------
    return F
    # ----------------------------------------------------------



""" ARCHIVE

"""
