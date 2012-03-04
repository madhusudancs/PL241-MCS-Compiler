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
import subprocess
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


class SSABrokenException(Exception):
  """Represents the exception when SSA structure is broken.
  """
  def __init__(self, instruction, *args, **kwargs):
    """Constructs the exception with the message.

    Args:
      instruction: The Instruction object at which the SSA structure is broken.
    """
    super(SSABrokenException, self).__init__(*args, **kwargs)
    self.msg = 'SSA structure broken variable redefined at instruction\n%s.' %(
        instruction)

  def __str__(self):
    return '%s: %s' % (self.__class__.__name__, self._msg)


class Register(object):
  """Represents a register.
  """

  name_counter = 0

  @classmethod
  def reset_name_counter(cls):
    """Resets the name counter for the new register allocation.
    """
    cls.name_counter = 0

  def __init__(self, name=None, new_allocation=False):
    """Constructs a register required for the program.

    Args:
      name: The name of the register this represents. This is an integer
          although it is called name for easier counting.
      new_allocation: If the current name run counter should be reset.
    """
    if new_allocation:
      self.__class__.reset_name_counter()
    if name:
      self.name = name
    else:
      self.name = self.__class__.name_counter
      self.__class__.name_counter += 1

    # The instruction object where the register is defined, part of the
    # def-use chain for the register.
    self.def_instruction = None

    # The list of instructions where the register is defined, part of the
    # def-use chain for the register.
    self.use_instructions = []

    # If the register is spilled it contains the two-tuple containing the
    # pointer to the instruction where it was spilled, pointer to the
    # instruction where this register needs to reloaded or None if not
    # spilled.
    self.spill = None

  def set_def(self, instruction):
    """Sets the instruction where this register is defined.

    Essentially sets up a def-use chain for the register.

    IMPORTANT: Since the program is in SSA form, if we already have a def for
               this register we should raise an exception.

    Args:
      instruction: The Instruction object where the variable is defined.
    """
    if self.def_instruction:
      raise SSABrokenException(instruction)
    self.def_instruction = instruction

  def defs(self):
    """Returns the instruction object where this register was defined.

    Returns None if the register is not defined yet.
    """
    return self.def_instruction

  def set_use(self, instruction):
    """Sets the instruction where this register is used.

    Essentially sets up a def-use chain for the register. A register can be
    used multiple times, so it is a set object.

    Args:
      instruction: The Instruction object where the variable is defined.
    """
    self.use_instructions.append(instruction)

  def uses(self):
    """Returns the set of instruction objects where this register is used.

    Returns an empty set if it is not used anywhere.
    """
    return self.use_instructions

  def __str__(self):
    """Returns the string representation for this register (preceded by r).
    """
    return 'r%d' % self.name

  def __repr__(self):
    """Returns the object representation string for this register.
    """
    return self.__str__()


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
      register = Register()
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
            new_register.set_use(instruction)

            instruction.operand2 = new_register

            phi_function['RHS'][0] = new_register

          new_operands = []
          for i, operand in enumerate(instruction.operands):
            if instruction.is_variable_or_label(operand):
              new_register = self.register_for_operand(operand)
              new_register.set_use(instruction)
              new_operands.append(new_register)

              phi_function['RHS'][i + 1] = new_register

          instruction.operands = new_operands

        # Reset all the function level datastructures
        Register.reset_name_counter()
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
          new_register.set_def(instruction)

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
          instruction.operand1.set_use(instruction)

        # The second operand of the branch instruction should still be a label.
        if instruction.instruction in ['beq', 'bne', 'blt',
                                       'ble', 'bgt', 'bge']:
          continue

        if instruction.is_variable_or_label(instruction.operand2):
          instruction.operand2 = self.register_for_operand(
              instruction.operand2)
          instruction.operand2.set_use(instruction)

        new_operands = []
        for operand in instruction.operands:
          if instruction.is_variable_or_label(operand):
            new_register = self.register_for_operand(operand)
            new_register.set_use(instruction)
            new_operands.append(new_register)

        instruction.operands = new_operands

        # After Copy propagation the only move instructions that remain
        # are for the function eplilogue, prologue and at the callee site
        # we need not allocate a register for results of these instructions.
        if instruction.instruction in ['move', 'store']:
          continue

        # Assign a register for the result of the instruction
        register = Register()
        register.set_def(instruction)
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

    Next-Farthest-Use for spilling uses the traces of the techniques of
    Belady's MIN Algorithm which is mentioned in the paper "Register
    allocation for programs in SSA-form" by Sebastian Hack, Daniel Grund,
    and Gerhard Goos available at:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.1578

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

    # We cannot do this processing until we find all the interferences at
    # the definition point of the current register.
    # Note we are using >= because when considering the numbers we need
    # to consider the current node also in addition to all the other
    # registers. For example if we have 8 registers and the current node
    # has 8 edges for the current node, we will have to spill because
    # we need one register for storing the result of the current instruction,
    # i.e. the current virtual registers interferes.
    if len(current_node.edges) >= self.num_registers:
      for edge in current_node.edges:
        # current_node.instructions[0] holds the current instruction we are
        # looking at and its edges give all the registers live at this point.
        # Obviously the current instruction is definition point for
        # the instruction.
        current_instruction = current_node.register.definition()
        if current_instruction.label != instructions[0]
          LOGGER.debug(
              'Something has horribly gone wrong: defintion point %d and '
              'the current start of live intervals %d not matching for the '
              ' current register %s. First fix the bugs you idiot!' % (
                  current_instruction.label, instructions[0],
                  current_node.register))

        # FIXME: Extremely inefficient (upto end of finding the next farthest),
        # use i.e. the max of the nearest uses for the current instruction.
        # Need a better datastructure, but my brain is exploding with so
        # many datastructures around. And Donald Knuth comes to my mind all
        # of a sudden: "Premature optimization is the root of ALL evil" :-P
        # Let us fix this if it is a bottleneck after profiling.
        reversed_uses = sorted(edge.register.uses(), key=lambda i: i.label,
                               reverse=True)

        # next_use because we are traversing the list reverse. Also we don't
        # test if this use is before the current instruction because, if that
        # is the case the register is already dead and not even interfering.
        next_use = reversed_uses[0]

        if next_use.label < current_instruction.label:
          LOGGER.debug(
              'Something has horribly gone wrong: Dead register %s is '
              'being considered for spilling at instruction %s where the '
              'new register %s is being defined.' % (
                  edge.register, current_instruction.label,
                  current_node.register))

        # A binary search may be better in this case than linear.
        for use in reversed_uses[1:]:
          if use.label < current_instruction.label:
            break
          next_use = use

        # Can't spill because it is required by current instruction.
        if next_use.label == current_instruction.label:
          continue

        # next_farthest_use is a two-tuple containing the instruction and
        # the register for the next farthest use.
        next_farthest_use = next_farthest_use if \
            (next_farthest_use[0].label > next_use.label) else \
                (next_use, edge.register)

    # Spill the register with the next farthest use.
    spill_register = next_farthest_use[1]
    spill_register.spill = (current_instruction, next_farthest_use[0])

  def build_interference_graph(self, live_intervals):
    """Builds the interference graph for the given control flow subgraph.

    Args:
      live_intervals: Dictionary containing the live intervals for the
          entire function.
    """
    # Clear the register_nodes dictionary for a new start
    self.register_nodes = {}

    self.sort_by_start = sorted(
        live_intervals, key=lambda k: live_intervals[k][0], reverse=True)

    self.current_live_intervals = live_intervals
    self.populate_collisions(len(self.sort_by_start) - 1)

    nodes = self.register_nodes.values()
    interference_graph = InterferenceGraph(nodes)
    self.interference_graphs.append(interference_graph)

    return interference_graph

  def spill(self, interference_graph):
    """Spill the registers in the interference graph.

    Args:
      interference_graph: The interference graph for which the SAT reduction
          should be performed.
    """
    pass

  def generate_node_bit_template(self):
    """Generates a template containing patterns for each node's clauses.

    This generates a template of position numbers whose literals should be
    negated. For example if we have 11 as the number of registers then
    the max bit value is 10 which in binary is 1010. We generate the template
    of the following format:
        [[0, 1],
         [0, 2, 3]
        ]
    """
    self.node_bit_template = []
    max_reg_binary = bin(self.num_registers - 1)[2:]

    # IMPORTANT: The bit_position is 0 for the Most Significant Bit (MSB)
    # to k - 1 for the Least Significant Bit (LSB) where k is the number
    # number of bits in the largest register value.
    ones_positions = []

    for bit_position, bit in enumerate(max_reg_binary):
      if bit == '1':
        ones_positions.append(bit_position)
      elif bit == '0':
        self.node_bit_template.append(ones_positions + [bit_position])

  def generate_edge_bit_template(self):
    """Generates a template containing patterns fo each edge's clauses.

    This template represents a variable or the negation of a variable. However
    because of the graph coloring problem, each of this variable is a
    combination of the corresponding literals in the binary representation
    of the color for two nodes joined by an edge. Example: if two nodes are
    r1 and r2 and there are 8 colors this can be represented as r10, r11, r12
    and r20, r21 and r22. Now if the bit template has has an entry "101",
    this can be encoded as (r12 | r22 | ~r11 | ~r21 | r10 | r20). This method
    only return the bit patterns, not the actual literals.
    """
    last_pattern = bin(self.num_registers - 1)[2:]

    # Number of literals per register.
    num_literals_per_reg = len(last_pattern)

    self.edge_bit_template = [last_pattern]
    for i in range(self.num_registers - 2, -1, -1):
      pattern = ('%s' % bin(i)[2:]).zfill(num_literals_per_reg)
      self.edge_bit_template.append(pattern)

    return num_literals_per_reg

  def generate_node_clauses(self, node):
    """Generates the clauses for the given node.

    Generate the clauses to exclude the numbers higher than the maximum
    number of registers - 1. For example, if we have 10 registers, then
    the maximum register number is 9 (0 to 9 registers), but since we
    use the 4 bit representation for this case, we need to ensure that
    the registers cannot take values from 10 to 15. So exlusion clauses
    are added to exclude these numbers for each register/node.

    Args:
      node: The node for which the exclusion clauses should be added.
    """
    clauses = []
    for template in self.node_bit_template:
      clause = ''
      for position in template:
        clause += '-%s%d' % (node, position)
      clauses.append(clause)

    return clauses

  def generate_edge_clauses(self, node1, node2):
    """Generates all the clauses for an edge.

    Args:
      node1, node2: The two ends of the edge for which the clauses should be
          generated.
    """
    clauses = []
    for template in self.edge_bit_template:
      clause = ''
      # IMPORTANT: The bit_position is 0 for the Most Significant Bit (MSB)
      # to k - 1 for the Least Significant Bit (LSB) where k is the number
      # number of bits in the largest register value.
      for bit_position, bit in enumerate(template):
        if bit == '0':
          clause += ('-%(node1)s%(bit_position)d '
              '-%(node2)s%(bit_position)d ') % {
                  'node1': node1,
                  'node2': node2,
                  'bit_position': bit_position,
              }
        else:
          clause += ('%(node1)s%(bit_position)d '
              '%(node2)s%(bit_position)d ') % {
                  'node1': node1,
                  'node2': node2,
                  'bit_position': bit_position,
                  }
      clauses.append(clause)

    return clauses

  def reduce_to_sat(self, interference_graph):
    """Reduces the graph coloring problem to the boolean satisfiability.

    * We use circuit encoding technique for reducing graph coloring to SAT.

    * The SAT represented is in Conjunctive Normal Form (CNF).

    * The techniques include the traces from the paper by Allen Van Gelder,
      titled "Another Look at Graph Coloring via Propositional Satisfiability"
      available at: www.soe.ucsc.edu/~avg/Papers/colorsat07.pdf

    Args:
      interference_graph: The interference graph for which the SAT reduction
          should be performed.
    """
    # Dictionary holding the
    processed = {}

    conflicting_registers = {}

    clauses = []

    self.generate_node_bit_template()

    num_literals_per_reg = self.generate_edge_bit_template()

    for node in interference_graph:
      node_reg = node.register

      # In the graph coloring problem
      if node.edges:
        conflicting_registers[node_reg] = True

      # Generate node specific clauses.
      clauses.extend(self.generate_node_clauses(node_reg))

      for edge in node.edges:
        edge_reg = edge.register
        if ((edge_reg, node_reg) in processed) or (
            (node_reg, edge_reg) in processed):
          continue

        processed[(node_reg, edge_reg)] = True

        conflicting_registers[edge_reg] = True

        # Generate edge specific clauses
        clauses.extend(self.generate_edge_clauses(node_reg, edge_reg))

    num_literals = len(conflicting_registers) * num_literals_per_reg

    return num_literals, clauses

  def sat_solve(self, interference_graph):
    """Converts the clauses to DIMACS format and feeds it to the SAT solver.

    This method also processes the ouput of the SAT solver.

    Args:
      interference_graph: the interference graph for which the SAT should be
          obtained and solved.
    """
    num_literals, clauses = self.reduce_to_sat(interference_graph)

    cnf = 'p cnf %d %d\n%s0' % (
         num_literals, len(clauses), '0\n'.join(clauses))

    process = subprocess.Popen('glucose_static', stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output = process.communicate(cnf)

    # Read last 2 lines of the STDOUT since if the SAT is SATISFIABLE the
    # pen-ultimate line will contain the exact string "s SATISFIABLE" and
    # last line contains the assignment else the last line contains the
    # exact string "s UNSATISFIABLE"
    lines = output[0].rsplit('\n', 3)

    if lines[-2] == 's UNSATISFIABLE':
      new_interference_graph = self.spill(interference_graph)
      self.sat_solve(new_interference_graph)
    elif lines[-2] == 's SATISFIABLE':
      print "Allocation: %s"

  def allocate(self):
    """Allocate the registers to the program.
    """
    self.allocate_virtual_registers()
    for dom_tree in self.ssa.cfg.dom_trees:
      self.liveness(dom_tree.other_universe_node)
      ifg = self.build_interference_graph(
          dom_tree.other_universe_node.live_intervals)
      #self.sat_solve(ifg)

  def is_register(self, operand):
    """Checks if the given operand is actually a register.
    """
    return True if (operand and isinstance(operand, Register)) else False

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
