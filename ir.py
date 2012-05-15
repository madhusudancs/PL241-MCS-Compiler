# Copyright 2012 Madhusudan C.S.
#
# This file ir.py is part of PL241-MCS compiler.
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

    Not in Prof. Franz's specs but makes things easier. He has given the
    liberty to add such things.
    call funcname a b c  calls a function or a procedure. Contains the name
                         of the function it calls and its operands are the
                         arguments to the function. If it is a function, it
                         returns a result, if it is a procedure not result
                         needs to be returned.

    .begin FUNCNAME a b  Denotes the prologue of a function and its formal
                         parameters.
    .end FUNCNAME a b    Denotes the epilogue of a function and its formal
                         parameters.

Conventions followed:
    [a]                value loaded from memory at memory offset (continuously
                       numbered not the memory bytes). Stored as memory object.
    #X                 X is a literal number
    !FP                Frame Pointer register
    .begin_funcname    Beginning of the function
    .end_funcname      Ending of the function
"""


import collections
import logging
import sys

from argparse import ArgumentParser

from datastructures import CFG
from datastructures import CFGNode
from parser import GLOBAL_SCOPE_NAME
from parser import LanguageSyntaxError
from parser import Parser


# Module level logger object
LOGGER = logging.getLogger(__name__)


# Anything that begins with this is not a valid variable in the source
# program and hence can be ignored during SSA construction.
NON_VARIABLE_OPERANDS_STARTSWITH = {
    '#': True,
    '.': True,
    '!': True,
    '[': True,
    }


def is_variable(operand):
  """Returns True if the operand is a variable.

    Args:
      operand: The operand to an that must be checked whether it is a
          variable or not.
  """
  if operand and isinstance(operand, str) and (
      operand[0] not in NON_VARIABLE_OPERANDS_STARTSWITH):
    return True

  return False


def is_variable_or_label(operand):
  """Checks if a given operand is a variable or a LABEL.

  Args:
    operand: The operand that must be checked if it is a variable or label
  """
  if operand:
    if isinstance(operand, str) and (
        operand[0] not in NON_VARIABLE_OPERANDS_STARTSWITH):
      return True
    elif isinstance(operand, int):
      return True

  return False


def is_memory(operand):
  """Checks if a given operand is a memory object.

  Args:
    operand: The operand that must be checked if it is a memory object.
  """
  return operand and isinstance(operand, Memory)


class Memory(object):
  """Represents a memory operand in a generic way.

  IMPORTANT: We don't talk about the memory size here, this is just an
  abstraction to represent memory. The size and other details are decided
  during the code generation phase.
  """

  def __init__(self, name=None, size=None, base=None, offset=None, scope=None):
    """Constructs a placeholder for memory required for the program.

    Args:
      name: Name for the particular memory location. Represents the name of
          the variable
      size: The size of this memory object in multiples of size of integers.
      base: The base memory address for this memory object. None indicates
          the beginning of the function.
      offset: The offset in the multiples of integer sizes from the base
          address where memory is allocated for this memory object.
    """
    # The name for the memory object. Represents the name of the variable.
    self.name = name

    # Until the code generation phase the size required is represented in
    # the multiples of size required to store an integer. Architecture
    # specific details will take care of this.
    self.size = size

    # The base memory address.
    self.base = base

    # This is the memory offset from the base of the stack. This will be
    # used only during code generation.
    self.offset = offset

    # The scope of this memory object.
    self.scope = scope

  def __str__(self):
    """Returns the string representation for this memory enclosed in [].
    """
    mem_str = '%d(%s)' % (
        self.offset if self.offset else 0, self.base if self.base else 0)
    if self.name:
      mem_str += ' [%s]' % (self.name)

    return mem_str


class Immediate(object):
  """Represents a immediate data in a generic way.
  """

  def __init__(self, value):
    """Constructs an object to represent immediate data.

    Args:
      value: The value of the immediate data.
    """
    self.value = value

  def __str__(self):
    """Returns the string representation for this memory enclosed in [].
    """
    return '#%d' % self.value if isinstance(self.value, int) \
        else ('#%s' % self.value)


class Instruction(object):
  """Abstraction for all the instruction in the Intermediate Representation.
  """

  label_counter = 0

  @classmethod
  def reset_counter(cls):
    """Resets the label counter for new IR generation.
    """
    cls.label_counter = 0

  def __init__(self, instruction, operand1=None, operand2=None, *operands):
    """Constructs the 3-address code for the instruction.

    Args:
      instruction: contains the instruction type for this instruction.
      operand1: The first operator for the instruction. This operand is not
          optional, should at least be updated later in case of branch
          instructions using the update method.
      operand2: The second operator for the instruction. This operand is optional.
      operands: A tuple of operands other than first two. We only want this
          in case of phi instructions.
    """
    self.instruction = instruction
    self.operand1 = operand1
    self.operand2 = operand2
    self.operands = operands

    # Stores the result of the instruction.
    # Used only when we start register allocation. While in the IR/SSA form
    # the instruction label itself is used as the result of the instruction.
    # However on real machine this should be stored in a register, so an
    # explicit result is required.
    # This is also introduced instead of using operands for storing the result
    # because, to be consistent with the way 'move' instruction works in our
    # IR move y x => x := y, we will have to place the result as the last
    # operand, but since operands are operand1, operand2 and then any
    # arbitrary number of operands finding the last operand becomes a pain.
    # So easiest way is to store the result separately
    self.result = None

    # These stores the assigned registers for the operands after register
    # allocation. These are introduced to make debugging easier.
    self.assigned_operand1 = None
    self.assigned_operand2 = None
    self.assigned_operands = []
    self.assigned_result = None

    # For function instructions, the function name is stored.
    self.function_name = None

    self.label = self.__class__.label_counter
    self.__class__.label_counter += 1

  def update(self, operand1=None, operand2=None):
    """Update the instruction operands at a later point.

    In some instructions like branch we will not know what the end of the
    taken or fall through branches to assign the branch labels, so in such
    cases we will have to update the operands at a later point.
    """
    if operand1 != None:
      self.operand1 = operand1
    if operand2 != None:
      self.operand2 = operand2

  def is_variable(self, operand):
    """Returns True if the operand is a variable.

    Args:
      operand: The operand to an instruction which must be checked for whether
          it is a variable or not.
    """
    return is_variable(operand)

  def is_variable_or_label(self, operand):
    """Returns True if the operand is a variable or a label of another
    instruction.

    Args:
      operand: The operand to an instruction which must be checked for whether
          it is a variable or a label of another instruction not.
    """
    return is_variable_or_label(operand)

  def is_memory(self, operand):
    """Returns True if the operand is a memory object.

    Args:
      operand: The operand to an instruction which must be checked for whether
          it is a memory object or not.
    """
    return is_memory(operand)

  def __contains__(self, operand):
    """Checks if the given operand belongs to the instruction.

    NOTE: If we are comparing with the Register objects == comparisions
    still work because Register class implements Register.__eq__() method.

    Args:
      operand: The operand that must be checked if it belongs to the class.
    """
    if self.result == operand:
      return True
    elif self.operand1 == operand:
      return True
    elif self.operand2 == operand:
      return True

    for additional_operand in self.operands:
      if operand == additional_operand:
        return True

    return False

  def __str__(self):
    """Prints the current instructions in the IR.
    """
    ir = '%4d: %5s' % (self.label, self.instruction)
    if self.operand1 != None:
      if isinstance(self.operand1, int):
        ir = ir + '%50s' % ('(%d)' % self.operand1)
      elif isinstance(self.operand1, CFGNode):
        ir = ir + '%50s' % (self.operand1.plain_str())
      else:
        ir = ir + '%50s' % (self.operand1)
    if self.operand2 != None:
      if isinstance(self.operand2, int):
        ir = ir + '%50s' % ('(%d)' % self.operand2)
      elif isinstance(self.operand2, CFGNode):
        ir = ir + '%50s' % (self.operand2.plain_str())
      else:
        ir = ir + '%50s' % (self.operand2)

    for operand in self.operands:
      ir = ir + '%50s' % ('(%d)' % (operand) if \
          isinstance(operand, int) else operand)

    return ir

  def __hash__(self):
    return hash('%s %s %s %s' % (
        self.instruction, self.operand1,
        self.operand2, ' '.join(['%s' % (o) for o in self.operands])))


class IntermediateRepresentation(object):
  """Stores the entire program in the Intermediate Representation form.
  """

  BUILT_INS_MAP = {
      'InputNum': 'read',
      'OutputNum': 'write',
      'OutputNewLine': 'writeLn',
      }

  COMPLEMENT_OPERATORS = {
      '==': '!=',
      '!=': '==',
      '<': '>=',
      '>': '<=',
      '<=': '>',
      '>=': '<',
      }

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
      'move': ['move'],
      'store': ['store'],
      'adda': ['adda'],
      'load': ['load'],
      'bra': ['bra'],
      }

  def __init__(self, function_name, parse_tree, local_symbol_table={},
               global_symbol_table={}):
    """Initializes the datastructres required for Intermediate Representation.

    Args:
      function_name: The name of the function that we are parsing.
      parse_tree: The parse tree for which the IR should be generated.
      local_symbol_table: Dictionary containing a symbol table for the
          given function being parsed
      global_symbol_table: Dictionary containing the symbol table for globals.
    """
    self.function_name = function_name

    self.parse_tree = parse_tree

    self.local_symbol_table = local_symbol_table
    self.global_symbol_table = global_symbol_table

    # Indicates whether this IR is being generated for functions or procedures.
    self.type = self.parse_tree.value

    # The instructions for the given function.
    self.ir = None

    self.basic_blocks = None

    # Control Flow Graph for the IR.
    self.cfg = None

    # A reverse map of the CFG nodes from the boundaries of the
    # CFG to the corresponding nodes.
    self.start_node_map = {}
    self.end_node_map = {}

    # Dictionary containing the keys as the CFG nodes that are pointed by
    # instructions.
    self.nodes_pointed_instructions = collections.defaultdict(list)

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
      self.ir.append(instruction)
      return instruction.label

    operand1 = operands[0]

    if len(operands) == 1:
      instruction = Instruction(self.INSTRUCTION_MAP[operator][0], operand1)
      self.ir.append(instruction)
      return instruction.label

    for ins, operand2 in zip(
        self.INSTRUCTION_MAP[operator], operands[1:]):
      instruction = Instruction(ins, operand1, operand2)
      self.ir.append(instruction)
      operand1 = instruction.label

    return operand1

  def generate(self):
    """Generates the Intermediate Representation from the parse tree.

    Since the IR holds a single function, this method directly kicks of the
    generation of IR for the function definition.
    """
    Instruction.reset_counter()
    self.ir = []

    ident, formal_param, func_body = self.parse_tree.children

    formal_parameters = []
    for parameter in formal_param.children:
      formal_parameters.append(self.ident(parameter))

    prologue_instruction = self.instruction('.begin_')
    self.ir[prologue_instruction].operands = formal_parameters
    self.ir[prologue_instruction].function_name = self.function_name

    stat_seq = func_body.children[-1]
    self.funcBody(stat_seq)

    epilogue_instruction = self.instruction('.end_')
    self.ir[epilogue_instruction].function_name = self.function_name

  def rewrite_branch_targets(self):
    """Rewrites all the branch targets from labels to the CFG nodes.

    This relies on the fact that every basic block beginning should be some
    branch target. Doing this gives us some advantage with branch targets
    since removing instructions during optimizations changes the ordering
    and resolving phi instructions after register allocation introduces more
    instructions which all break the ordering in self.ir.ir and branch targets
    may not remain the same. So it is just easier to use the fact about
    basic block nodes in the CFG to store branch targets.
    """
    for instruction in self.ir:
      if instruction.instruction == 'bra':
        node = self.start_node_map[instruction.operand1]
        instruction.operand1 = node
        self.nodes_pointed_instructions[node].append(instruction)
      elif instruction.instruction in ['beq', 'bne', 'blt',
                                       'ble', 'bgt', 'bge']:
        node = self.start_node_map[instruction.operand2]
        instruction.operand2 = node
        self.nodes_pointed_instructions[node].append(instruction)

  def identify_basic_blocks(self):
    """Identifies the basic block for the IR.
    """
    self.basic_blocks = collections.OrderedDict()

    # Entry instruction is always a leader
    leaders = [0]

    targets = collections.defaultdict(list)
    return_targets = {}

    i = 1
    while i < len(self.ir):
      instruction = self.ir[i]
      if instruction.instruction == 'ret':
        return_targets[i] = True
      elif instruction.instruction == 'bra':
        target = instruction.operand1
        leaders.append(target)
        targets[i].append(target)

        leaders.append(i + 1)
      elif instruction.instruction in ['beq', 'bne', 'blt',
                                       'ble', 'bgt', 'bge']:
        leaders.append(instruction.operand2)
        targets[i].append(instruction.operand2)

        leaders.append(i + 1)
        targets[i].append(i+1)
      elif instruction.instruction == '.end_':
        leaders.append(i)

      i += 1


    sorted_leaders = sorted(set(leaders))
    i = 1
    while i < len(sorted_leaders):
      leader = sorted_leaders[i - 1]

      # End if before the next leader
      end = sorted_leaders[i] - 1

      block_dict = {
          'end': end,
          'targets': targets.get(end, []),
          'return_target': return_targets.get(end, False)
          }

      if not (block_dict['targets'] or block_dict['return_target'] or
          self.ir[block_dict['end']].instruction.startswith('.end')):
        block_dict['targets'].append(block_dict['end'] + 1)

      self.basic_blocks[leader] = block_dict
      i += 1

    # For the last block the last instruction is the end of the basic block.
    leader = sorted_leaders[-1]
    self.basic_blocks[leader] = {
        'end': len(self.ir) - 1,
        'return_target': True,
        }

    # Create a basic block that grounds all the nodes in the end.
    self.basic_blocks[len(self.ir) - 1] = {
        'end': len(self.ir) - 1
    }

    return self.basic_blocks

  def build_cfg(self):
    """Build the Control Flow Graph for the IR.
    """
    self.identify_basic_blocks()

    nodes = collections.OrderedDict()

    for leader, leader_dict in self.basic_blocks.iteritems():
      entry = True if self.ir[leader].instruction.startswith('.begin_') \
          else False
      node = CFGNode(value=(leader, leader_dict['end']), entry=entry)
      self.start_node_map[leader] = node
      self.end_node_map[leader_dict['end']] = node
      nodes[leader] = node

    end = len(self.ir) - 1
    nodes[end] = CFGNode(value=(end, end))

    for node in nodes.values():
      basic_block = self.basic_blocks.get(node.value[0], None)
      for target_leader in basic_block.get('targets', []):
        node.append_out_edges(nodes[target_leader])

      return_target = basic_block.get('return_target', False)
      if return_target:
        node.append_out_edges(nodes[end])

    cfg_nodes = nodes.values()

    self.cfg = CFG(cfg_nodes)

    self.rewrite_branch_targets()

    return self.cfg

  def condition(self, root, complement=False):
    """Process the condition node.
    """
    op_node1, relational_operator, op_node2 = root.children
    operand1 = self.dfs(op_node1)
    operand2 = self.dfs(op_node2)

    if complement:
      operator = self.COMPLEMENT_OPERATORS[relational_operator.value]
    else:
      operator = relational_operator.value

    return self.instruction(operator, operand1, operand2, None)

  def taken(self, root):
    """Process the taken branch in the branch instructions.
    """
    return self.dfs(root.children[0])

  def fallthrough(self, root):
    """Process the fallthrough branch in the branch instructions.
    """
    return self.dfs(root.children[0])

  def abstract(self, root):
    """Process the abstract nodes in the parse tree.
    """
    func = getattr(self, root.value)
    return func(root)

  def funcBody(self, root):
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

    condition_result = self.condition(condition, complement=True)
    taken_result = self.taken(taken)

    if fallthrough == None:
      self.ir[condition_result].update(operand2=taken_result+1)
      return taken_result
    else:
      taken_result = self.instruction('bra')
      self.ir[condition_result].update(operand2=taken_result+1)
      fallthrough_result = self.fallthrough(fallthrough)
      self.ir[taken_result].update(operand1=fallthrough_result+1)
      return fallthrough_result

  def keyword_call(self, root):
    """Process the call statement.
    """
    func_node = root.children[0]
    func_name = func_node.value
    arguments = root.children[1:]

    argument_results = []
    for arg in arguments:
      argument_results.append(self.expression(arg))

    call_instruction = self.instruction('call')
    self.ir[call_instruction].function_name = func_name
    self.ir[call_instruction].operands = argument_results

    # All functions return whatever self.instruction method return which
    # returns a label, so let us be consistent
    return call_instruction

  def keyword_let(self, root):
    """Process the let statement.
    """
    lvalue, array_offset = self.designator(root.children[0], lvalue=True)
    rvalue = self.expression(root.children[2])

    if array_offset:
      result = self.instruction('store', rvalue, array_offset)
      self.ir[result].operands = [lvalue]
      return result
    else:
      return self.instruction('move', rvalue, lvalue)

  def keyword_return(self, root):
    """Process the return statement.
    """
    if self.type == 'procedure':
      raise LanguageSyntaxError('A return statement was encountered in the '
          'procedure "%s". Procedures can\'t return values. If you want to '
          'return a value use functions instead' % (self.function_name))

    result = self.expression(root.children[0])

    return_instruction = self.instruction('ret')
    self.ir[return_instruction].operand1 = result
    self.ir[return_instruction].function_name = self.function_name

    # All functions return whatever self.instruction method return which
    # returns a label, so let us be consistent
    return return_instruction

  def keyword_while(self, root):
    """Generate the IR for while statement.
    """
    condition = root.children[0]
    taken = root.children[1]

    condition_result = self.condition(condition, complement=True)
    self.taken(taken)
    # We are branching to condition_result - 1 because condition consists
    # of two instructions, cmp and condition, we need to go back to condition.
    branch_result = self.instruction('bra', condition_result-1)
    self.ir[condition_result].update(operand2=branch_result+1)

    return branch_result

  def designator(self, root, lvalue=False):
    """Generate the IR for "designator" nodes.
    """
    result = self.dfs(root.children[0])
    if len(root.children) <= 1:
      return (result, None) if lvalue else result

    # Else this must be an array.

    if not isinstance(result, Memory):
      array_name = result
      result = Memory(name=array_name, scope=self.function_name)
      self.local_symbol_table[array_name]['memory'] = result

    if result.scope == self.function_name:
      symtab_entry = self.local_symbol_table[result.name]
    elif result.scope == GLOBAL_SCOPE_NAME:
      symtab_entry = self.global_symbol_table[result.name]

    dimensions = symtab_entry['dimensions']
    expression_result = self.expression(root.children[1])
    for i, offset in enumerate(root.children[2:]):
      temp_result = self.instruction('*', expression_result,
                                     Immediate(dimensions[i + 1]))
      offset_result = self.expression(offset)
      expression_result = self.instruction('+', offset_result,
                                           temp_result)

    if lvalue:
      return result, expression_result

    result = self.instruction('load', expression_result, result)
    return result

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
    if root.value in self.local_symbol_table:
      return '%s' % (root.value)
    elif root.value in self.global_symbol_table:
      if 'memory' in self.global_symbol_table[root.value]:
        return self.global_symbol_table[root.value]['memory']
      else:
        memory = Memory(name=root.value, scope=GLOBAL_SCOPE_NAME)
        self.global_symbol_table[root.value]['memory'] = memory
        return memory
    else:
      raise LanguageSyntaxError('Identifier "%s" encountered which is not '
          'declared anywhere.')

  def number(self, root):
    """Returns the number by prefixing # as per convention.
    """
    return Immediate(root.value)

  def dfs(self, root):
    """Depth-first search the parse tree and translate node by node.

    Although the only common part it shares with other trees is the depth-first
    recursion, but requires special processing at each step.
    """
    root.compress()

    func = getattr(self, root.type)
    return func(root)

  def __iter__(self):
    """Implements the bootstrapping part of the Python iterator protocol.
    """
    self.count = 0
    self.len_of_ir = len(self.ir)
    return self

  def next(self):
    """Returns the next element in the instruction for the iterator protocol.
    """
    if self.count < self.len_of_ir:
      instruction = self.ir[self.count]
      self.count += 1
      return instruction
    else:
      raise StopIteration

  def __str__(self):
    """Prints the current instructions in the IR.
    """
    ir = ''
    for instruction in self.ir:
      ir += '%s\n' % (instruction)

    return ir


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-g', '--vcg', metavar="VCG", type=str,
                      nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph output.')
  parser.add_argument('-r', '--ir', metavar="IR", type=str,
                      nargs='?', const=True,
                      help='Generate the Intermediate Representation.')
  parser.add_argument('-t', '--dom', metavar="DominatorTree", type=str,
                      nargs='?', const=True,
                      help='Generate the Dominator Tree VCG output.')

  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  try:
    p = Parser(args.file_names[0])
    ir = IntermediateRepresentation(p)

    ir.generate()
    cfg = ir.build_cfg()
    cfg.compute_dominance_frontiers()

    if args.vcg:
      external_file = isinstance(args.vcg, str)
      vcg_file = open(args.vcg, 'w') if external_file else \
          sys.stdout
      vcg_file.write(cfg.generate_vcg(ir=ir.ir))
      if external_file:
        vcg_file.close()

    if args.ir:
      external_file = isinstance(args.ir, str)
      ir_file = open(args.ir, 'w') if external_file else \
          sys.stdout
      ir_file.write(str(ir))
      if external_file:
        ir_file.close()

    if args.dom:
      external_file = isinstance(args.dom, str)
      dom_file = open(args.dom, 'w') if external_file else \
          sys.stdout
      dom_file.write(str(cfg.generate_dom_vcg()))
      if external_file:
        dom_file.close()

    return ir
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ir = bootstrap()
