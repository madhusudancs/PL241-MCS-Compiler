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

  label_counter = 0

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
    self.label = self.__class__.label_counter
    self.__class__.label_counter += 1

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
    operand1 = operands[0]

    for ins, operand2 in zip(self.INSTRUCTION_MAP[operator], operands[1:]):
      self.ir.append(Instruction(i, operand1, operand2))
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
      elif c.type == 'abstract' and c.value == 'statSeq':
        self.funcBody(c, 'main')

    # Generate the entire IR by concatenating all the functional IRs.
    # First add main to the instruction set.
    self.ir.extend(self.function_ir)
    for ir_set in self.temp_ir:
      self.ir.extend(ir_set)


  def abstract(self, root):
    """Process the abstract nodes in the parse tree.
    """
    func = getattr(self, root.value)
    return func(root)

  def function(self, root):
    """Process the complete function and generate IR for it.
    """
    ident, formal_param, func_body = root.children
    stat_seq = func_body.children[-1]
    self.funcBody(stat_seq, ident)

  def funcBody(self, root, scope):
    """Process the body of the function and generate IR for it.
    """
    self.push_scope(scope)

    self.dfs(root)

  def statement(self, root):
    """Processes statement type node.
    """
    for children in root.children:
      self.dfs(children)

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
    pass

  def keyword_call(self, root):
    """Process the call statement.
    """
    pass

  def keyword_let(self, root):
    """Process the let statement.
    """
    pass

  def term(self, root):
    """Generate the IR for "term" nodes.
    """
    operand1 = self.dfs(root.children[0])
    num_children = len(root.children)
    i = 1
    while i < num_children:
      operator = root.children[i]
      operand2 = self.dfs(root.children[i + 1])
      operand1 = self.instruction(operator, operand1, operand2)
      i += 2

    return operand1

  def statSeq(self, root):
    """Process the statSeq type node in the parse tree.
    """
    for statement in root.children:
      self.dfs(statement)

  def ident(self, root):
    """Return identifier as the operator.
    """
    return '%s/%s' % (self.current_scope(), root.value)

  def dfs(self, root):
    """Depth-first search the parse tree and translate node by node.

    Although the only common part it shares with other trees is the depth-first
    recursion, but requires special processing at each step.
    """
    root.compress()

    func = getattr(self, root.type)
    print "Once: %s: %s" % (root.type, root.value)
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

