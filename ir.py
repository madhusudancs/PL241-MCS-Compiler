# Copyright 2012 Madhusudan C.S. 
#
# This file parser.py is part of PL241-MCS compiler.
#
# PL241-MCS compiler is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PL241-MCS compiler is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PL241-MCS compiler. If not, see <http://www.gnu.org/licenses/>.

"""Generates the Intermediate Representation (IR for short) for PL241.

It is on this Intermediate Representation that we perform all the machine
independent optimizations.

Allowed Intermediate Representation instructions are:

    neg x                 unary minus
    add x y               addition
    sub x y               subtraction
    mul x y               multiplication
    div x y               division
    cmp x y               comparison
    adda x y              add two addresses x und y (used only with arrays)
    load y                load from memory address y
    store y x             store y to memory address x
    move y x              assign x := y
    phi x x1 x2 ...       x := Phi(x1, x2, x3, ...)
    end                   end of program
    bra y                 branch to y
    bne x y               branch to y on x not equal
    beq x y               branch to y on x equal
    ble x y               branch to y on x less or equal
    blt x y               branch to y on x less
    bge x y               branch to y on x greater or equal
    bgt x y               branch to y on x greater
    read                  read
    write                 write
    wln                   writeLn

Conventions followed:
    [a] value loaded from memory at memory address a
    #X X is a literal number
    !FP Frame Pointer register
"""

import copy
import logging

from argparse import ArgumentParser

from parser import Parser

# Module level logger object
LOGGER = logging.getLogger(__name__)


class Instruction(object):
  """Abstraction for all the instruction in the Intermediate Representation.
  """

  global_counter = 0
  local_counter = 0

  def __init__(self, instruction, operand1=None, operand2=None):
    """Constructs the 3-address code for the instruction.

    Args:
      instruction: contains the instruction type for this instruction.
      op1: The first operator for the instruction. This operand is not
          optional, should at least be updated later in case of branch
          instructions using the update method.
      op2: The second operator for the instruction. This operand is optional.
    """
    self.instruction = instruction
    self.operand1 = operand1
    self.operand2 = operand2
    self.label = self.__class__.global_counter
    self.local_label = self.__class__.local_counter
    self.__class__.global_counter += 1
    self.__class__.local_counter += 1

  @classmethod
  def local_reset(cls):
    """Reset the label counter variable.
    """
    cls.local_counter = 0

  def update(self, operand1=None, operand2=None):
    """Update the instruction operands at a later point.

    In some instructions like branch we will not know what the end of the
    taken or fall through branches to assign the branch labels, so in such
    cases we will have to update the operands at a later point.
    """
    if operand1:
      self.operand1 = operand1
    if operand2:
      self.operand2 = operand2


class IntermediateRepresentation(object):
  """Stores the entire program in the Intermediate Representation form.
  """

  BUILT_INS = ['InputNum', 'OutputNum', 'OutputNewLine']

  INSTRUCTION_MAP = {
      '+': ['add'],
      '-': ['sub'],
      '*': ['mul'],
      '/': ['div'],
      '=': ['move'],
      '==': ['cmp', 'beq'],
      '!=': ['cmp', 'bne'],
      '<': ['cmp', 'blt'],
      '<=': ['cmp', 'ble'],
      '>': ['cmp', 'bgt'],
      '>=': ['cmp', 'bge'],
      'mov': ['mov'],
      'store': ['store'],
      'adda': ['adda'],
      'load': ['load'],
      'bra': ['bra'],
      'InputNum': ['read'],
      'OutputNum': ['write'],
      'OutputNewLine': ['writeLn'],
      }

  def __init__(self, parse_tree):
    """Initializes the datastructres required for Intermediate Representation.

    Args:
      parse_tree: The parse tree for which the IR should be generated.
    """
    self.original_parse_tree = copy.deepcopy(parse_tree)

    self.parse_tree = parse_tree
    # We will have an IR per function which we concatenate in the end.
    self.ir = []
    # This holds the IR per function.
    self.function_ir = []
    # This stores the list of function_irs
    self.temp_ir = []

    self.scope_stack = []

  def print_ir(self):
    """Prints the current instructions in the IR.
    """
    for n in i.ir:
      print "%d: %s %s\t\t%s" % (n.label, n.instruction,
                                 n.operand1, n.operand2)

  def push_scope(self, scope):
    """Pushes the scope to the scope stack.

    Args:
      scope: A string that represents the current scope.
    """
    self.scope_stack.append(scope)

  def pop_scope(self):
    """Pops and returns the element from the top of the scope stack.
    """
    return self.scope_stack.pop(len(self.scope_stack) - 1)

  def current_scope(self):
    """Returns the current scope of the program, which is the top of the stack.
    """
    return self.scope_stack[-1]

  def instruction(self, operator, *operands):
    """Builds the instruction for the given arguments.

    Args:
      operator: Determines the type of instruction.
      operands: tuple of operands for the instruction.
    """
    # If there are no operands such instructions update through backpatching.
    # These are also special instructions in the sense that they are passed
    # as the instruction itself as the value to the operator parameter to this
    # function.
    if not operands:
      instruction = Instruction(operator)
      self.function_ir.append(instruction)
      return instruction.label

    operand1 = operands[0]

    for ins, operand2 in zip(
        self.INSTRUCTION_MAP[operator], operands[1:]):
      instruction = Instruction(ins, operand1, operand2)
      self.function_ir.append(instruction)
      operand1 = instruction.label

    return operand1

  def generate(self):
    """Generates the Intermediate Representation from the parse tree.
    """
    # We need to only convert function bodies to IR. The declaration of
    # variables really exist for scope checking. So we can directly skip to
    # generate IR for the first function we encounter.
    for c in self.parse_tree.root.children:
      if c.type == 'abstract' and c.value == 'varDecl':
        continue
      elif c.type == 'keyword' and c.value == 'function':
        self.function(c)
      elif c.type == 'abstract' and c.value == 'funcBody':
        # Initalize the first function IR with the scope label for
        # the function IR.
        scope = 'main'
        self.push_scope(scope)
        self.instruction('.%s' % scope)
        self.funcBody(c.children[0], 'main')

    # Generate the entire IR by concatenating all the functional IRs.
    # First add main to the instruction set.
    self.ir.extend(self.function_ir)
    for ir_set in self.temp_ir:
      self.ir.extend(ir_set)

    # Add the end instruction.
    instruction = Instruction('end')
    self.ir.append(instruction)

  def condition(self, root):
    """Process the condition node.
    """
    op_node1, relational_operator, op_node2 = root.children
    operand1 = self.dfs(op_node1)
    operand2 = self.dfs(op_node2)

    return self.instruction(relational_operator.value, operand1,
                            operand2, None)

  def taken(self, root):
    """Process the taken branch in the branch instructions.
    """
    return self.dfs(root.children[0])

  def fallthrough(self, root):
    """Process the fallthrough branch in the branch instructions.
    """
    self.dfs(root.children[0])
    return self.instruction('bra')

  def abstract(self, root):
    """Process the abstract nodes in the parse tree.
    """
    func = getattr(self, root.value)
    return func(root)

  def formalParam(self, root):
    """Generates the IR for loading formal paramters.
    """
    # The first formal parameter value is after framelength and return
    # label, so advance it by 8 bytes.
    start = self.instruction('+', '!FP', '#8')
    for parameter in root.children:
      instruction_label = self.instruction('load', start)
      instruction_label = self.instruction('mov', instruction_label, '[ret]')
      start = self.instruction('+', start, '#4')

    return instruction_label

  def function(self, root):
    """Process the complete function and generate IR for it.
    """
    ident, formal_param, func_body = root.children

    scope = ident.value

    self.push_scope(scope)

    # Initalize the first function IR with the scope label for the function IR.
    self.instruction('.%s' % (scope))

    start = self.instruction('+', '!FP', '#0')
    fp_label = self.instruction('load', start)
    self.instruction('mov', fp_label, '[framesize]')

    ret = self.instruction('+', start, '#4')
    return_label = self.instruction('load', ret)
    self.instruction('mov', return_label, '[ret]')

    self.formalParam(formal_param)

    stat_seq = func_body.children[-1]
    self.funcBody(stat_seq, ident.value)

    # Indicate the end of current function by adding all the function
    # IR code to the temp_ir list and empty the function_ir to prepare
    # for the next function.
    self.temp_ir.append(self.function_ir)
    self.function_ir = []
    Instruction.local_reset()

    self.pop_scope()

  def funcBody(self, root, scope):
    """Process the body of the function and generate IR for it.
    """
    # Generate IR for the function.
    self.dfs(root)

  def statement(self, root):
    """Processes statement type node.
    """
    for children in root.children:
      result = self.dfs(children)

    return result

  def keyword(self, root):
    """Processes keyword type node.
    """
    # We prefix keyword_ to the methods of keywords because they may clash
    # with Python's keywords
    func = getattr(self, 'keyword_%s' % root.value)
    return func(root)

  def keyword_if(self, root):
    """Process the if statement.
    """
    condition = root.children[0]
    taken = root.children[1]
    fallthrough = root.children[2] if len(root.children) == 3 else None

    condition_result = self.condition(condition)
    fallthrough_result = self.fallthrough(fallthrough)
    self.function_ir[condition_result].update(operand2=fallthrough_result+1)
    taken_result = self.taken(taken)
    self.function_ir[fallthrough_result].update(operand1=taken_result+1)

    return taken_result

  def keyword_call(self, root):
    """Process the call statement.
    """
    func_name = root.children[0]
    arguments = root.children[1:]

    # Advance the frame pointer by the framesize of the calling function
    advance = self.instruction('+', '!FP', '[framesize]')
    # Store it as the current framepointer
    self.instruction('mov', advance, '!FP')

    offset = 0
    # The length of the new function's frame will be number of arguments
    # + the framelength storage + return label storage + return value
    # storage multiplied by size of each storage which is 4
    self.instruction('store', '#%d' % ((len(arguments) + 3) * 4), '!FP')

    # Store the return label in the next storage area.
    storage = self.instruction('+', '!FP', '#4')
    # We do not know the return instruction label yet, so have to
    # be backpatched.
    return_store_result = self.instruction('store', None, storage)

    argument_results = []
    for arg in arguments:
      storage = self.instruction('+', storage, '#4')
      expression_result = self.expression(arg)
      argument_results.append(self.instruction('store', expression_result,
                                               storage))

    if func_name in self.BUILT_INS:
      result = self.instruction(func_name)
    else:
      # Currently a dummy value which is the label of the function is inserted
      # but will later be updated with the actual value when merging IR for
      # each individual functions.
      result = self.instruction('bra', '.%s' % (func_name))

    self.function_ir[return_store_result].update(operand1=result+1)

  def keyword_let(self, root):
    """Process the let statement.
    """
    lvalue = self.designator(root.children[0])
    rvalue = self.expression(root.children[2])

    # FIXME: When lvalue is an array.
    return self.instruction('mov', lvalue, rvalue)

  def keyword_return(self, root):
    """Process the return statement.
    """
    result = self.dfs(root.children[0])

    # Store the result of the return in the memory address corresponding
    # to the return value of this function which is denoted by the function
    # name.
    self.instruction('store', result, '[%s]' % (self.current_scope()))
    result = self.instruction('bra')

    # FIXME: Return to where?
    # Backpatch with dummy value for now.
    self.function_ir[result].update(operand1='[ret]')

    return result

  def designator(self, root):
    """Generate the IR for "designator" nodes.
    """
    #FIXME: Fix the designator for array case.
    return self.dfs(root.children[0])

  def factor(self, root):
    """Generate the IR for "factor" nodes.
    """
    return self.dfs(root.children[0])

  def term(self, root):
    """Generate the IR for "term" nodes.
    """
    operand1 = self.dfs(root.children[0])
    num_children = len(root.children)
    i = 1
    while i < num_children:
      operator = root.children[i]
      operand2 = self.dfs(root.children[i + 1])
      operand1 = self.instruction(operator.value, operand1, operand2)
      i += 2

    return operand1

  def expression(self, root):
    """Process an expression.
    """
    operand1 = self.dfs(root.children[0])
    num_children = len(root.children)
    i = 1
    while i < num_children:
      operator = root.children[i]
      operand2 = self.dfs(root.children[i + 1])
      operand1 = self.instruction(operator.value, operand1, operand2)
      i += 2

    return operand1

  def statSeq(self, root):
    """Process the statSeq type node in the parse tree.
    """
    for statement in root.children:
      result = self.dfs(statement)

    return result

  def ident(self, root):
    """Return identifier as the operator.
    """
    return '%s/%s' % (self.current_scope(), root.value)

  def number(self, root):
    """Returns the number by prefixing # as per convention.
    """
    return '#%s' % (root.value)

  def dfs(self, root):
    """Depth-first search the parse tree and translate node by node.

    Although the only common part it shares with other trees is the depth-first
    recursion, but requires special processing at each step.
    """
    root.compress()

    func = getattr(self, root.type)
    return func(root)


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  try:
    p = Parser(args.file_names[0])
    i = IntermediateRepresentation(p)
    return i
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  i = bootstrap()
  i.generate()
  i.print_ir()
