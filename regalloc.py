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

from datastructures import InterferenceGraph
from datastructures import InterferenceNode
from ir import IntermediateRepresentation
from optimizations import Optimize
from parser import LanguageSyntaxError
from parser import Parser
from ssa import SSA


# Module level logger object
LOGGER = logging.getLogger(__name__)


class RegisterAllocator(object):
  """Allocates the registers for the given SSA form of the source code.
  """

  def __init__(self, ssa, num_registers=8):
    """Initializes the allocator with SSA.

    Args:
      ssa: the SSA object for the program IR in SSA form.
      num_registers: Number of Machine registers available.
    """
    self.ssa = ssa

    self.num_registers = num_registers

    # Dictionary whose keys are opernads (or variables) and the values are
    # the registers assigned in the virtual registers space.
    self.variable_register_map = {}

    # Count holding the currently alloted registers in the virtual
    # registers space.
    self.register_count = 0

    # Dictionary of loop header nodes as the keys and the values are the
    # loop footer nodes.
    self.loop_pair = {}

    # Dictionary containing the live ranges for each register
    self.live_ranges = {}

    # List of interference graphs where each graph corresponds to a function.
    self.interference_graphs = []

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
    # List of (phi instruction, phi function) tuples. Ordering is important,
    # so no dictionary only a list of tuples. Further we only do sequential
    # processing so doesn't make sense to use a dictionary.
    phi_instructions = []

    for instruction in self.ssa.optimized():
      if instruction.instruction.startswith('.begin_'):
        continue
      elif instruction.instruction.startswith('.end_'):
        # Process all the phi-functions in the end of the function because
        # there are phi-functions especially in case of blocks whose operands
        # are defined after this block. And also it may so happen that, that
        # definition is removed because of copy propagation and only some
        # other instructions label remaining. This cannot be determined or
        # fixed before the result of the instruction whose result is the
        # operand for the phi function is computed
        for instruction, phi_function in phi_instructions:
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

        # Reset all the function level datastructures
        self.register_count = 0
        self.variable_register_map = {}
        phi_instructions = []

      elif instruction.instruction == 'bra':
        continue
      elif instruction.instruction == 'phi':
        # phi instructions are special cases, so handle them separately
        # Phi functions in the original basic blocks should be
        # updated as well
        node = self.ssa.label_nodes[instruction.label]

        original_variable = instruction.operand1.rsplit('_', 1)[0]
        phi_function = node.phi_functions[original_variable]

        # The first operand of the phi instruction is actually the result
        # of the phi function and the operands from 2 to all other operands
        # are the inputs to the phi function.
        operand1 = instruction.operand1
        if instruction.is_variable_or_label(operand1):
          new_register = self.register_for_operand(operand1)
          instruction.operand1 = new_register

        phi_function['LHS'] = new_register

        # The operands/RHS of the phi function are handled at the end of the
        # function call. See the comment where it is handled to know why.
        # Just record the phi instruction here, process later
        phi_instructions.append((instruction, phi_function))
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

  def analyze_basic_block_liveness(self, start_node, node):
    """Analyzes the liveness of the variables in the given basic block

    Performs a post-order traversal of the dominator tree for processing
    the liveness, so the traversal is on the dominator tree.

    This code is almost the translation of the pseudo-code from the paper
    titled, "Linear Scan Register Allocation on SSA Form" by Christian Wimmer
    and Michael Franz available at:
    http://www.christianwimmer.at/Publications/Wimmer10a/Wimmer10a.pdf

    Args:
      start: The starting node of the control flow subgraph that stores the
          complete subgraph specific information like liveness of each
          variable etc.
      node: The node of the control flow subgraph on which post-order
          traversal should be performed.
    """
    for child in node.out_edges:
      if self.visited.get(child, False):
        self.loop_pair[child] = node
        continue

      self.visited[child] = True

      self.analyze_basic_block_liveness(start_node, child)

    # The live variables set in the block where each key is the variable and
    # the value is a two elements first representing the start of the range
    # and second representing the end of the range.
    live = {}
    intervals = {}

    # Dictionary containing the in_edge and the operands that
    # should be included only for those predecessors
    include = collections.defaultdict(list)

    for successor in node.out_edges:
      exclude = []
      for successor_in in successor.live_include:
        if successor_in != node:
          exclude.extend(successor.live_include[successor_in])

      successor_live_in = set(successor.live_in.keys()) - set(exclude)

      for variable in successor_live_in:
        # None for first element of the list means it starts from the
        # beginning of the block and None for the second element means
        # it runs until the end of the block. See datastructures.CFGNode
        # __init__() method for matching how things work with live
        # dictionaries.
        live[variable] = True
        intervals[variable] = list(node.value)

      for phi_function in successor.phi_functions.values():
        # Get the in-edges of the successor to determine which entry in the
        # phi-functions input corresponds to a given node, since the order
        # in the phi-function's input is same as the order of the in-edges.
        # Look at ssa.SSA.search() method to see why this ordering is
        # preserved.

        # This is adding one more degree to the O(n) polynomial since the
        # index implementation on the Python lists is O(n) and this is within
        # a nested loop. Think of something efficient?
        input_position = successor.in_edges.index(node)

        operand = phi_function['RHS'][input_position]
        if self.is_register(operand):
          live[operand] = True
          intervals[operand] = list(node.value)

    # start and end labels of the basic blocks in the SSA CFG which is the
    # other universe of regular IR's CFG.
    start, end = node.value

    # Adjust start ignoring phi instructions, since they are dealt separately
    start -= len(node.phi_functions)

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
        intervals[instruction.operand1] = [node.value[0], instruction.label]

      operand2 = instruction.operand2
      if self.is_register(operand2) and operand2 not in live:
        live[instruction.operand2] = True
        intervals[instruction.operand2] = [node.value[0], instruction.label]

      for operand in instruction.operands:
        if self.is_register(operand) and operand not in live:
          intervals[operand] = [node.value[0], instruction.label]
          live[operand] = True

    for phi_function in node.phi_functions.values():
      intervals[phi_function['LHS']][0] = node.value[0]
      live.pop(phi_function['LHS'])

    for phi_function in node.phi_functions.values():
      for i, operand in enumerate(phi_function['RHS']):
        include[node.in_edges[i]].append(operand)
        if operand in intervals:
          intervals[operand][1] = phi_function['instruction'].label
        elif self.is_register(operand):
          live[operand] = True
          intervals[operand] = [node.value[0],
                                phi_function['instruction'].label]

    node.live_in = live
    node.live_include = include

    for operand in intervals:
      if operand in start_node.live_intervals:
        # Merge the intervals
        start_node.live_intervals[operand] = [
            min(intervals[operand][0], start_node.live_intervals[operand][0]),
            max(intervals[operand][1], start_node.live_intervals[operand][1]),
            ]
      else:
        # Add a new interval
        start_node.live_intervals[operand] = intervals[operand]

  def liveness(self, start):
    """Computes the liveness range for each variable.

    Args:
      start: the starting node of the subgraph on which liveness must be
          computed.
    """
    # FIXME: Liveness checking for global variables is completely different
    # since they are live until the last call to the function that uses
    # global variables return. So may be it is easier to keep the live
    # ranges alive during the entire range of the program? Too much
    # spilling? Actually Prof. Franz seeded this idea about which I had
    # not thought about till now. "Compile each function independently."

    # A temporary dictionary containing the nodes visited as the keys and
    # dummy True as the value. This dictionary must be reset for every
    # traversal.
    self.visited = {}
    self.analyze_basic_block_liveness(start, start)

  def populate_collisions(self, index):
    """Backtracking algorithm to find the register collision.

    Args:
      index: The index in the list of registers sorted by start first registers
          but in reverse order. The register that starts last is first in the
          self.sort_by_start list
    """
    # Base case for the backtracking algorithm
    if index < 0:
      return []

    self.populate_collisions(index - 1)

    register = self.sort_by_start[index]
    instructions = self.current_live_intervals[register]
    current_node = InterferenceNode(self.sort_by_start[index], instructions)

    previous_node = self.register_nodes.get(self.sort_by_start[index - 1],
                                            None)
    if previous_node and previous_node.instructions[1] > instructions[0]:
      current_node.append_edges(previous_node)
      for previous_collision in previous_node.edges:
        # Note we are deliberately leaving out the
        # previous_end == current_start case because in such cases the
        # previous register can be reused for the current register's
        # definition.
        if previous_collision.instructions[1] > instructions[0]:
          current_node.append_edges(previous_collision)

    self.register_nodes[register] = current_node

  def build_interference_graph(self, start):
    """Builds the interference graph for the given control flow subgraph.

    Args:
      node: The starting node of the subgraph that holds the liveness
          information for all the registers in the entire subgraph.
    """
    # Clear the register_nodes dictionary for a new start
    self.register_nodes = {}

    self.sort_by_start = sorted(
        start.live_intervals, key=lambda k: start.live_intervals[k][0],
        reverse=True)

    self.current_live_intervals = start.live_intervals
    self.populate_collisions(len(self.sort_by_start) - 1)

    nodes = self.register_nodes.values()
    interference_graph = InterferenceGraph(nodes)
    self.interference_graphs.append(interference_graph)

  def allocate(self):
    """Allocate the registers to the program.
    """
    self.allocate_virtual_registers()
    for dom_tree in self.ssa.cfg.dom_trees:
      self.liveness(dom_tree.other_universe_node)
      self.build_interference_graph(dom_tree.other_universe_node)

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
       virtual_alloc_str += '%10s  <- %s\n' % (
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
                           'of the interference graph.')
  parser.add_argument('--virtualreggraph', metavar="Virtual Registers Graph",
                      type=str, nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph '
                           'for the virtual registers allocated and liveness '
                           'computed for the subgraphs.')
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

    optimize = Optimize(ssa)
    optimize.optimize()

    regalloc = RegisterAllocator(ssa)
    regalloc.allocate()

    if args.vcg:
      external_file = isinstance(args.vcg, str)
      vcg_file = open(args.vcg, 'w') if external_file else sys.stdout
      for graph in regalloc.interference_graphs:
        vcg_file.write('%s\n' % graph.generate_vcg())

      if external_file:
        vcg_file.close()

    if args.virtualreggraph:
      external_file = isinstance(args.virtualreggraph, str)
      virtualreggraph_file = open(args.virtualreggraph, 'w') if \
          external_file else sys.stdout
      virtualreggraph_file.write(ssa.ssa_cfg.generate_virtual_reg_vcg(ssa=ssa))
      if external_file:
        virtualreggraph_file.close()

    if args.virtual:
      external_file = isinstance(args.virtual, str)
      virtual_file = open(args.virtual, 'w') if external_file \
          else sys.stdout
      virtual_file.write(regalloc.str_virtual_register_allocation())
      if external_file:
        virtual_file.close()

    return regalloc

  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  regalloc = bootstrap()
