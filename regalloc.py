# Copyright 2012 Madhusudan C.S. 
#
# This file regalloc.py is part of PL241-MCS compiler.
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

"""Algorithms to allocate registers for the program.
The architecture used is X86_64.
"""


import collections
import logging
import sys

from argparse import ArgumentParser

from ir import IntermediateRepresentation
from optimizations import cse_cp
from parser import LanguageSyntaxError
from parser import Parser
from ssa import SSA


# Module level logger object
LOGGER = logging.getLogger(__name__)


class RegisterAllocator(object):
  """Allocates the registers for the given SSA form of the source code.
  """

  def __init__(self, ssa):
    """Initializes the allocator with SSA.
    """
    self.ssa = ssa

    # Dictionary whose keys are opernads (or variables) and the values are
    # the registers assigned in the virtual registers space.
    self.variable_register_map = {}

    # Count holding the currently alloted registers in the virtual
    # registers space.
    self.register_count = 0

    # Dictionary of loop header nodes as the keys and the values are the
    # loop footer nodes.
    self.loop_pair = {}

  def register_for_operand(self, operand):
    """Finds an already existing register for the operand or creates a new one.

    Args:
      operand: The operand for which the register must be found.
    """
    if operand in self.variable_register_map:
      return self.variable_register_map[operand]
    else:
      register = 'r%d' % self.register_count
      self.register_count += 1
      self.variable_register_map[operand] = register
      return register

  def allocate_virtual_registers(self):
    """Allocate registers in virtual infinite space of registers.

    This is the most important phase because, we actually explicitly allocate
    some register for the result of the instructions in this phase.
    """
    for instruction in self.ssa.optimized():
      if instruction.instruction.startswith('.begin_'):
        continue
      elif instruction.instruction.startswith('.end_'):
        self.register_count = 0
        self.variable_register_map = {}
        continue
      elif instruction.instruction == 'bra':
        continue
      elif instruction.instruction == 'phi':
        # Phi functions in the original basic blocks should be
        # updated as well
        node = self.ssa.label_nodes[instruction.label]

        original_variable = instruction.operand1.rsplit('_', 1)[0]
        phi_function = node.phi_functions[original_variable]

        # phi instructions are special cases, so handle them separately
        # The first operand of the phi instruction is actually the result
        # of the phi function and the operands from 2 to all other operands
        # are the inputs to the phi function.
        operand1 = instruction.operand1
        if instruction.is_variable_or_label(operand1):
          new_register = self.register_for_operand(operand1)
          instruction.operand1 = new_register

          phi_function['LHS'] = new_register

        operand2 = instruction.operand2
        if instruction.is_variable_or_label(operand2):
          new_register = self.register_for_operand(operand2)
          instruction.operand2 = new_register

          phi_function['RHS'][0] = new_register

        new_operands = []
        for i, operand in enumerate(instruction.operands):
          if instruction.is_variable_or_label(operand):
            new_register = self.register_for_operand(operand)
            new_operands.append(new_register)

            phi_function['RHS'][i + 1] = new_register

        instruction.operands = new_operands

      else:
        if instruction.is_variable_or_label(instruction.operand1):
          instruction.operand1 = self.register_for_operand(
              instruction.operand1)

        # The second operand of the branch instruction should still be a label.
        if instruction.instruction in ['beq', 'bne', 'blt',
                                       'ble', 'bgt', 'bge']:
          continue

        if instruction.is_variable_or_label(instruction.operand2):
          instruction.operand2 = self.register_for_operand(
              instruction.operand2)
        new_operands = []
        for operand in instruction.operands:
          if instruction.is_variable_or_label(operand):
            new_operands.append(self.register_for_operand(operand))
  
        instruction.operands = new_operands

        # After Copy propagation the only move instructions that remain
        # are for the function eplilogue, prologue and at the callee site
        # we need not allocate a register for results of these instructions.
        if instruction.instruction in ['move', 'store']:
          continue

        # Assign a register for the result of the instruction
        register = 'r%d' % self.register_count
        self.register_count += 1
        self.variable_register_map[instruction.label] = register
        instruction.result = register

  def allocate(self):
    """Allocate the registers to the program.
    """
    self.allocate_virtual_registers()
    self.liveness()

  def analyze_basic_block_liveness(self, root):
    """Analyzes the liveness of the variables in the given basic block

    Performs a post-order traversal of the dominator tree for processing
    the liveness, so the traversal is on the dominator tree.

    This code is almost the translation of the pseudo-code from the paper
    titled, "Linear Scan Register Allocation on SSA Form" by Christian Wimmer
    and Michael Franz available at:
    http://www.christianwimmer.at/Publications/Wimmer10a/Wimmer10a.pdf

    Args:
      root: The root of the dominator subtree on which post-order traversal
          should be performed.
    """
    for child in root.out_edges:
      if self.visited.get(child, False):
        self.loop_pair[child] = root
        continue

      self.visited[child] = True

      self.analyze_basic_block_liveness(child)

    # The live variables set in the block where each key is the variable and
    # the value is a two elements first representing the start of the range
    # and second representing the end of the range.
    live = {}
    intervals = {}

    for successor in root.out_edges:
      for variable in successor.live_in.keys():
        # None for first element of the list means it starts from the
        # beginning of the block and None for the second element means
        # it runs until the end of the block. See datastructures.CFGNode
        # __init__() method for matching how things work with live
        # dictionaries.
        live[variable] = True
        intervals[variable] = [None, None]

      for phi_function in successor.phi_functions.values():
        # Get the in-edges of the successor to determine which entry in the
        # phi-functions input corresponds to a given node, since the order
        # in the phi-function's input is same as the order of the in-edges.
        # Look at ssa.SSA.search() method to see why this ordering is
        # preserved.

        # This is adding one more degree to the O(n) polynomial since the
        # index implementation on the Python lists is O(n) and this is within
        # a nested loop. Think of something efficient?
        input_position = successor.in_edges.index(root)

        operand = phi_function['RHS'][input_position]
        if self.is_register(operand):
          live[operand] = True
          intervals[operand] = [None, None]

    # start and end labels of the basic blocks in the SSA CFG which is the
    # other universe of regular IR's CFG.
    start, end = root.value

    # Adjust start ignoring phi instructions, since they are dealt separately
    start -= len(root.phi_functions)

    # Walk the instructions in the reverse order, note -1 for stepping.
    for instruction in self.ssa.optimized(end, start - 1, reversed=True):
      if instruction.instruction == 'phi':
        continue

      if self.is_register(instruction.result):
        if instruction.result not in live:
          # Dead-on-Arrival. I love Dead-on-Arrival stuff, more
          # optimizations! :-P
          intervals[instruction.result] = [instruction.label,
                                           instruction.label]
          # NOTE: pop is not added because it doesn't even exist
        else:
          intervals[instruction.result][0] = instruction.label
          live.pop(instruction.result)

      # We need to process the input operands if and only if they don't
      # really exist, if they already exist, it means that they are used
      # else where after this basic block and hence should stay alive.
      # Only the last use (or the first sighting since we are coming from
      # reverse now should get an lifetime end set.)
      operand1 = instruction.operand1
      if self.is_register(operand1) and operand1 not in live:
        live[instruction.operand1] = True
        intervals[instruction.operand1] = [None, instruction.label]

      operand2 = instruction.operand2
      if self.is_register(operand2) and operand2 not in live:
        live[instruction.operand2] = True
        intervals[instruction.operand2] = [None, instruction.label]

      for operand in instruction.operands:
        if self.is_register(operand) and operand not in live:
          intervals[operand] = [None, instruction.label]
          live[operand] = True

    for phi_function in root.phi_functions.values():
      intervals[phi_function['LHS']][0] = root.value[0]
      live.pop(phi_function['LHS'])

      # Handle loop headers
#      if b is loop header then
#        loopEnd = last block of the loop starting at b
#        for each opd in live do
#          intervals[opd].addRange(b.from, loopEnd.to)

    root.live_in = live
    root.live_intervals = intervals

  def liveness(self):
    """Computes the liveness range for each variable.
    """
    # A temporary dictionary containing the nodes visited as the keys and
    # dummy True as the value. This dictionary must be reset for every
    # traversal.
    self.visited = {}

    # FIXME: Liveness checking for global variables is completely different
    # since they are live until the last call to the function that uses
    # global variables return. So may be it is easier to keep the live
    # ranges alive during the entire range of the program? Too much
    # spilling? Actually Prof. Franz seeded this idea about which I had
    # not thought about till now. "Compile each function independently."

    for dom_tree in self.ssa.cfg.dom_trees:
      self.analyze_basic_block_liveness(dom_tree.other_universe_node)

  def is_register(self, operand):
    """Checks if the given operand is actually a register.
    """
    return True if (operand and isinstance(operand, str) and \
        operand[0] == 'r' and operand[1:].isdigit()) else False

  def str_virtual_register_allocation(self):
    """Gets the text representation of the program after virtual allocation.
    """
    virtual_alloc_str = ''
    for instruction in self.ssa.optimized():
       virtual_alloc_str += '%s\t<- %s\n' % (
           instruction.result if instruction.result else '', instruction)
    return virtual_alloc_str

def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-g', '--vcg', metavar="VCG", type=str,
                      nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph '
                          'output.')
  parser.add_argument('--virtual', metavar="Allocate Virtual Register ",
                      type=str, nargs='?', const=True,
                      help='Allocate registers in the virtual space of '
                          'infinite registers.')
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

    ssa = SSA(ir, cfg)
    ssa.construct()

    ssa = cse_cp(ssa)

    regalloc = RegisterAllocator(ssa)
    regalloc.allocate()

    if args.vcg:
      vcg_file = open(args.vcg, 'w') if isinstance(args.vcg, str) else \
          sys.stdout
      vcg_file.write(ssa.cfg.generate_virtual_reg_vcg(ssa=ssa))
      vcg_file.close()

    if args.virtual:
      external_file = isinstance(args.virtual, str)
      virtual_file = open(args.virtual, 'w') if external_file \
          else sys.stdout
      virtual_file.write(regalloc.str_virtual_register_allocation())
      if external_file:
        virtual_file.close()

    return ssa

  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ssa = bootstrap()
