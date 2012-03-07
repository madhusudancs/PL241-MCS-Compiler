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
from datastructures import LiveIntervalsHeap
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

    # If the register is spilled it contains a dictionary with three keys,
    #   "spilled_at": pointer to the instruction object where it was spilled
    #   "spilled_to": pointer to the instruction where this register needs
    #        to reloaded
    #   "register": register object where this register will be reloaded.
    # This attribute is None if the register is not spilled.
    self.spill = None

    # Color assignment after solving the K-coloring problem using SAT solver.
    # Note even though they are called colors they are just integral values.
    self.color = None

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

  def definition(self):
    """Returns the instruction object where this register was defined.

    Returns None if the register is not defined yet.
    """
    return self.def_instruction

  def set_use(self, *instructions):
    """Sets the instruction where this register is used.

    Essentially sets up a def-use chain for the register. A register can be
    used multiple times, so it is a set object.

    Args:
      instruction: The Instruction object where the variable is defined.
    """
    self.use_instructions.extend(instructions)

  def uses(self):
    """Returns the set of instruction objects where this register is used.

    Returns an empty set if it is not used anywhere.
    """
    return self.use_instructions

  def assignment(self, instruction):
    """Returns the color assignment for the register at the given instruction.

    This method is extremely intelligent :-P If the node is spilled at the
    given instruction it returns the color assigned to the new register. And
    this chains as long as the spill cycle exists :-)

    Args:
      instruction: The instruction object at which the register assignment
          should be obtained.
    """
    if not self.spill:
      return self.color

    # Either when the register is not spilled at all as above or spilled but
    # at any instruction before spill the same register is used.
    if instruction.label < self.spill['spilled_at'].label:
      return self.color

    # If the register is spilled and the current instruction is later than
    # or at the instruction where this register needs to be reloaded we
    # delegate to the reloaded register to do return its assignment.
    # NOTE: This gets recursive, if the reloaded register is spilled again.
    # This is very nice because we need not do this in a loop individually
    # for all the chained spills.
    if instruction.label >= self.spill['spilled_to'].label:
      print "--Spilled at: %d" % (self.spill['spilled_at'].label),
      print '(%s)' % self,
      print "--",
      return self.spill['register'].assignment(instruction)

    if (self.spill['spilled_at'].label <=
        instruction.label < self.spill['spilled_to'].label):
      LOGGER.debug('Things have gone terribly wrong since we have an '
          'instruction %s where the definition/use of register %s occurs '
          'when it was spilled before this instruction i.e. at the '
          'instruction %s and reloaded after this instruction i.e. at '
          'instruction %s' % (instruction, self, self.spill['spilled_at'],
                              self.spill['spilled_to']))

  def def_use_chains(self):
    """Returns the def-use chain pair for the register.
    """
    for use_instruction in self.use_instructions:
      yield self.def_instruction, use_instruction

  def __eq__(self, register):
    """Checks if the two registers are same.
    """
    # A bit too pessimistic at times, just making sure to check if the
    # def instructions match. It should not be needed most of the times.
    # But once we get everything up and running we can get rid of that
    # additional check of def_instruction.
    # IMPORTANT: Don't check if they are the same objects using "is", because
    # later some one may decide to simulate the register but with a new object
    # is created. The given two checks should be sufficient.
    return True if (self.name == register.name and
        self.def_instruction == register.def_instruction) else False

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
        for phi_instruction, phi_function in phi_instructions:
          operand2 = phi_instruction.operand2
          if phi_instruction.is_variable_or_label(operand2):
            new_register = self.register_for_operand(operand2)
            new_register.set_use(phi_instruction)

            phi_instruction.operand2 = new_register

            phi_function['RHS'][0] = new_register

          new_operands = []
          for i, operand in enumerate(phi_instruction.operands):
            if phi_instruction.is_variable_or_label(operand):
              new_register = self.register_for_operand(operand)
              new_register.set_use(phi_instruction)
              new_operands.append(new_register)

              phi_function['RHS'][i + 1] = new_register

          phi_instruction.operands = new_operands

        # We need to keep track of this for spilling registers which spawns
        # off new registers to give the new registers the names.
        # FIXME: should be eliminated once each function can be compiled
        # independently
        instruction.last_register_count = Register.name_counter

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
        if self.is_register(variable):
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

    # Walk the instructions in the reverse order, note -1 for stepping.
    for instruction in self.ssa.optimized(end, start - 1, reversed=True):
      # Keep track of the end of the function and append the last register
      # name value to the start node
      # FIXME: Get rid of this when functions can be compiled independently.
      if instruction.instruction.startswith('.end_'):
        start_node.last_register_count = instruction.last_register_count
        continue

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

    if node.phi_functions:
      start_node.phi_nodes.append(node)

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

  def spill(self, current_node, collisions):
    """Checks if the spilling is necessary, if so spills the register.

    Makes a spill decision and if required spills the register and makes
    all the manipulations required to the spilled register and creates a
    new register to keep the SSA properties intact.

    This runs a spill decision based on Next-Farthest-Use technique which
    contains the traces of the techniques of Belady's MIN Algorithm which
    is mentioned in the paper "Register allocation for programs in SSA-form"
    by Sebastian Hack, Daniel Grund, and Gerhard Goos available at:
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.86.1578

    Args:
      current_node: The current node whose register is starting at this
          instruction point and may need to replace another register.
      collisions: List of collisions with the current register node.
    """
    # Note we are using < because when considering the numbers we need
    # to consider the current node also in addition to all the other
    # registers. For example if we have 8 registers and the current node
    # has 8 edges for the current node, we will have to spill because
    # we need one register for storing the result of the current instruction,
    # i.e. the current virtual registers interferes.
    if len(collisions) < self.num_registers:
      # No need to spill anything, we are good still :-)
      return collisions

    register = current_node.register

    next_farthest_use = None

    for collision in collisions:
      # current_node.instructions[0] holds the current instruction we are
      # looking at and its edges give all the registers live at this point.
      # Obviously the current instruction is definition point for
      # the instruction.
      current_instruction = register.definition()
      if (current_instruction.label != current_node.instructions[0] and
          current_instruction.instruction != 'phi'):
        # The latter case is allowed because this is possible for phi
        # instructions because there is no ordering for phi instructions
        # and all the phi instruction results are assumed to start the
        # basic block although they have separate instructions.
        LOGGER.debug(
            'Something has horribly gone wrong: defintion point %d and '
            'the current start of live intervals %d not matching for the '
            ' current register %s. First fix the bugs you idiot!' % (
                current_instruction.label, current_node.instructions[0],
                register))

      collision_register_uses = collision.register.uses()
      if (collision_register_uses[-1].instruction == 'phi' and
          not collision_register_uses[-1].label >= current_instruction.label
          and current_instruction.label <
          collision.register.definition().label):
        next_farthest_use = None
        break
      else:
        # next_use because we are traversing the list reverse. Also we don't
        # test if this use is before the current instruction because, if that
        # is the case the register is already dead and not even interfering.
        next_use_index, next_use = -1, collision_register_uses[-1]

        if next_use.label < current_instruction.label and (
            next_use.instruction != 'phi'):
          # The latter case is possible for phi instructions because there is
          # in case of loops the phi operand from the loop footer is used
          # earlier by the instructions number than it is defined.
          LOGGER.debug(
              'Something has horribly gone wrong: Dead register %s is '
              'being considered for spilling at instruction %s where the '
              'new register %s is being defined.' % (
                  collision.register, current_instruction.label,
                  current_node.register))

        # A binary search may be better in this case than linear.
        for use_index, use in enumerate(collision_register_uses[-2::-1]):
          if use.label < current_instruction.label:
            break
          next_use_index, next_use = use_index, use

        # Can't spill because it is required by current instruction.
        if next_use.label == current_instruction.label:
          continue

        # next_farthest_use is a two-tuple containing the instruction and
        # the register interference node for the next farthest use.
        if not (next_farthest_use and
            (next_farthest_use['next_use'].label > next_use.label)):
          next_farthest_use = {
              'next_use_index': next_use_index,
              'next_use': next_use,
              'collision': collision
              }

    if next_farthest_use:
      # Spill the register with the next farthest use.
      spilling_node = next_farthest_use['collision']
      spill_register = spilling_node.register

      # Cut short the instruction range for the spilled register node.
      spilling_node.instructions[1] = current_instruction.label

      # Create a new register. The last register count is the count of the
      # number of registers assigned, the actual name of the last register
      # will be -1 of this value. So we directly assign this value to the new
      # register.
      # FIXME: We do not have to do this name assignments if we compile each
      # function independently
      new_register = Register(name=self.last_register_count)
      self.last_register_count += 1

      new_register.set_def(next_farthest_use['next_use'])
      new_register.set_use(
          spill_register.uses()[next_farthest_use['next_use_index']:])

      # Push the new register down the heap.
      self.live_intervals_heap.push(new_register,
          (next_farthest_use['next_use'].label, spilling_node.instructions[1]))

      # Update the spill information for the spilled register.
      spill_register.spill = {
          'spilled_at': current_instruction,
          'spilled_to': next_farthest_use['next_use'],
          'register': new_register
          }
    else:
      spilling_node = collision
      spill_register = spilling_node.register
      new_register = Register(name=self.last_register_count)
      new_register.set_def(spill_register.definition())
      new_register.set_use(spill_register.uses()[:-1])

      self.last_register_count += 1
      self.live_intervals_heap.push(new_register,
          (new_register.definition().label, spilling_node.instructions[1]))

    # Remove this from the collisions list of the currently
    # processing register.
    collisions.remove(spilling_node)

    return collisions

  def populate_collisions(self):
    """?Greedy?/?Dynamic Programming? algorithm to find the register collision.

    Args:
      index: The index in the list of registers sorted by start first registers
          but in reverse order. The register that starts last is first in the
          self.sort_by_start list
    """
    for register in self.live_intervals_heap:
      # The register object sorted by start of the liveness interval.

      # instructions is two-tuple containing the start and end of the
      # liveness range for register obtained in the previous statement.
      instructions = self.live_intervals_heap[register]

      # Create a new interference node for the current register.
      current_node = InterferenceNode(register, instructions)

      previous_node = self.register_nodes.get(
          self.live_intervals_heap.previous(), None)

      # Holds the list of colliding registers with this node.
      collisions = []

      # FIXME: May lead to bugs? Do we really have to check if the previous
      # node or any of its previous node is actually spilled? If it is
      # spilled it doesn't collide with the previous node right?
      if previous_node:
        if previous_node.instructions[1] > instructions[0]:
          collisions.append(previous_node)

        for previous_collision in previous_node.edges:
          # Note we are deliberately leaving out the
          # previous_end == current_start case because in such cases the
          # previous register can be reused for the current register's
          # definition.
          if previous_collision.instructions[1] > instructions[0]:
            collisions.append(previous_collision)

      # We cannot do this processing until we find all the interferences at
      # the definition point of the current register.
      collisions = self.spill(current_node, collisions)

      current_node.append_edges(*collisions)
      self.register_nodes[register] = current_node

  def build_interference_graph(self, live_intervals, phi_functions,
                               last_register_count):
    """Builds the interference graph for the given control flow subgraph.

    Args:
      live_intervals: Dictionary containing the live intervals for the
          entire function.
      phi_functions: Contains all the phi functions in the given program
          function.
      last_register_count: The count of the last register in this function.
          The actual name of the last register is -1 of this value.
    """
    # Clear the register_nodes dictionary for a new start
    self.register_nodes = {}

    self.live_intervals_heap = LiveIntervalsHeap(live_intervals)

    self.phi_functions = phi_functions

    # FIXME: We do not have to do this if we compile each function
    # independently.
    self.last_register_count = last_register_count

    self.current_live_intervals = live_intervals
    self.populate_collisions()

    nodes = self.register_nodes.values()

    interference_graph = InterferenceGraph(nodes)
    self.interference_graphs.append(interference_graph)

    return interference_graph

  def get_cnf_var(self, register, bit_position):
    """Returns the CNF variable for a register if it exists or creates one.

    Args:
      register: The register for which the CNF variable should be obtained.
      bit_position: The bit position for which the CNF variable should be
          obtained.
    """
    register_var = (register, bit_position)
    if register_var in self.register_cnf_map:
      cnf_var = self.register_cnf_map.get(register_var)
    else:
      self.cnf_var_count += 1
      self.register_cnf_map[register_var] = self.cnf_var_count
      self.cnf_register_map[self.cnf_var_count] = register_var
      cnf_var = self.register_cnf_map[register_var]

    return cnf_var

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

  def generate_node_clauses(self, register):
    """Generates the clauses for the given node.

    Generate the clauses to exclude the numbers higher than the maximum
    number of registers - 1. For example, if we have 10 registers, then
    the maximum register number is 9 (0 to 9 registers), but since we
    use the 4 bit representation for this case, we need to ensure that
    the registers cannot take values from 10 to 15. So exlusion clauses
    are added to exclude these numbers for each register/node.

    Args:
      register: The node for which the exclusion clauses should be added.
    """
    clauses = []
    for template in self.node_bit_template:
      clause = ''
      for bit_position in template:
        cnf_var = self.get_cnf_var(register, bit_position)

        clause += '-%s' % (cnf_var)

      clauses.append(clause)

    return clauses

  def generate_edge_clauses(self, register1, register2):
    """Generates all the clauses for an edge.

    Args:
      register1: One of the two ends of the edge for which the clauses
          should be generated.
      register2: One of the two ends of the edge for which the clauses
          should be generated. This can be None if a register has absolutely
          no collisions.
    """
    clauses = []

    # Handle the case when the register has absolutely no conflicts.
    # We have to just generate clauses containing one CNF variable for each
    # bit position.
    if not register2:
      # Since this template is expected to have the same bit-width for all
      # the patterns just pick the first one for the number of bits count.
      num_bits = len(self.edge_bit_template[0])
      for bit_position in range(num_bits):
        cnf1_var = self.get_cnf_var(register1, bit_position)
        clauses.append('%s ' % (cnf1_var))

      return clauses

    for template in self.edge_bit_template:
      clause = ''
      # IMPORTANT: The bit_position is 0 for the Most Significant Bit (MSB)
      # to k - 1 for the Least Significant Bit (LSB) where k is the number
      # number of bits in the largest register value.
      for bit_position, bit in enumerate(template):
        cnf1_var = self.get_cnf_var(register1, bit_position)
        cnf2_var = self.get_cnf_var(register2, bit_position)

        if bit == '0':
          clause += '-%s -%s ' % (cnf1_var, cnf2_var)
        else:
          clause += '%s %s ' % (cnf1_var, cnf2_var)

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

    # The maps storing the registers to CNF variable mappings. One small
    # difference between the two maps is that the first map stores the
    # mapping from the register bit variable to CNF variable, however the
    # latter stores the mapping from CNF variable to two tuple, containing
    # the register bit and the register object itself. If we don't store the
    # reference to the register object, there is no way for us to retrieve
    # it back later.
    self.register_cnf_map = {}
    self.cnf_register_map = {}

    self.cnf_var_count = 0

    clauses = []

    self.generate_node_bit_template()

    num_literals_per_reg = self.generate_edge_bit_template()

    for node in interference_graph:
      node_reg = node.register

      if node.edges:
        conflicting_registers[node_reg] = True

      # Generate node specific clauses.
      clauses.extend(self.generate_node_clauses(node_reg))

      if node.edges:
        for edge in node.edges:
          edge_reg = edge.register
          if ((edge_reg, node_reg) in processed) or (
              (node_reg, edge_reg) in processed):
            continue
          processed[(node_reg, edge_reg)] = True

          # Generate edge specific clauses
          clauses.extend(self.generate_edge_clauses(node_reg, edge_reg))
      elif node.register.uses():
        node_clauses = self.generate_edge_clauses(node_reg, None)
        clauses.extend(node_clauses)

    return self.cnf_var_count, clauses

  def generate_assignments(self, cnf_assignment):
    """Generates the assignments for the registers from the CNF form.

    Args:
      cnf_assignment: A string containing the CNF assignments returned from
          the SAT solver.
    """
    # The cnf_assignment string consists of a single line of starting with
    # the letter v followed by a space assignments which are space separated
    # and then ending with 0
    # Example: v -1 -2 -3 4 5 -6 -7 -8 -9 -10 11 -12 -13 -14 15 -16 -17 -18 0
    # So exclude the first and the last entry.
    assignments_str = cnf_assignment.split()[1:-1]

    # Dictionary containing the register objects as keys and the values are
    # another dictionary where the keys are bit positions and the values
    # are assignments.
    registers = collections.defaultdict(dict)

    for assignment_str in assignments_str:
      assignment = assignment_str.strip()
      if assignment[0] == '-':
        register, bit_position = self.cnf_register_map.get(int(assignment[1:]))
        # We want to have strings for binary to integer conversion, however
        # the bit positions must be integers
        registers[register][bit_position] = '0'
      else:
        register, bit_position = self.cnf_register_map.get(int(assignment))
        # We want to have strings for binary to integer conversion however
        # bit position must be integers
        registers[register][bit_position] = '1'

    LOGGER.debug(registers)

    # Since all the registers are supposed to have same number of bits, since
    # it is resolved like that during the conversion to CNF, we get the number
    # of bits from the first value in the dictionary
    num_bits = len(registers.values()[0])
    for register in registers:
      reg_binary = ''
      for bit_position in range(num_bits):
        reg_binary += registers[register][bit_position]

      # Do a conversion from binary string to integer using the built-in int
      # type constructor with base as the argument.
      register.color = int(reg_binary, 2)

      LOGGER.debug('Register: %s Color: %d' % (register, register.color))

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

    LOGGER.debug(cnf)

    process = subprocess.Popen('glucose_static', stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    output = process.communicate(cnf)

    LOGGER.debug(output[0])

    # Read last 2 lines of the STDOUT since if the SAT is SATISFIABLE the
    # pen-ultimate line will contain the exact string "s SATISFIABLE" and
    # last line contains the assignment else the last line contains the
    # exact string "s UNSATISFIABLE"
    lines = output[0].rsplit('\n', 3)

    if lines[-2] == 's UNSATISFIABLE':
      LOGGER.debug('SAT Unsatisfiable')
      return False, ''
    elif lines[-3] == 's SATISFIABLE':
      assignment = lines[-2]
      self.generate_assignments(assignment)
      LOGGER.debug('SAT Satisfiable! Allocation: %s' % (lines[-1]))
      return True, assignment

  def allocate(self):
    """Allocate the registers to the program.
    """
    self.allocate_virtual_registers()
    for dom_tree in self.ssa.cfg.dom_trees:
      self.liveness(dom_tree.other_universe_node)
      # FIXME: We do not have to pass the second argument if we compile
      # each function independently.
      ifg = self.build_interference_graph(
          dom_tree.other_universe_node.live_intervals,
          dom_tree.other_universe_node.phi_nodes,
          dom_tree.other_universe_node.last_register_count)
      is_allocated, allocation = self.sat_solve(ifg)
      if is_allocated:
        LOGGER.debug('Allocated for subgraph %s!' % (
            dom_tree.other_universe_node))
      else:
        LOGGER.debug('Allocation Failed for subgraph %s :-(' % (
            dom_tree.other_universe_node))
        # No point in proceeding if register allocation fails. Some major
        # bug in the code. So bail out.
        return False, dom_tree.other_universe_node

    return True, None

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

  def deconstruct_basic_block(self, node):
    """Processes basic blocks by performing a pre-order traversal.

    Args:
      node: The basic block node to process.
    """
    block_deconstruction_map = {}
    for instruction in self.ssa.optimized(node.value[0], node.value[1] + 1):
      if instruction.instruction == 'phi':
        #do something for phi
        #continue
        pass

      if self.is_register(instruction.result):
        assignment = instruction.result.assignment(instruction)
        assignment = '%10d' % assignment if assignment != None else '      None'
        print assignment,
        print '%10s' % ('(%s)' % instruction.result,),
      else:
        print '                     ',

      print '%10s' % instruction.label,
      print '    ',
      print '%10s' % instruction.instruction,

      if self.is_register(instruction.operand1):
        assignment = instruction.operand1.assignment(instruction)
        assignment = '%10d' % assignment if assignment != None else '      None'
        print assignment,
        print '%10s' % ('(%s)' % instruction.operand1,),
      else:
        print '%20s' % instruction.operand1,

      if self.is_register(instruction.operand2):
        assignment = instruction.operand2.assignment(instruction)
        assignment = '%10d' % assignment if assignment != None else '      None'
        print assignment,
        print '%10s' % ('(%s)' % instruction.operand2,),
      else:
        print '%20s' % instruction.operand2,

      for op in instruction.operands:
        if self.is_register(op):
          assignment = op.assignment(instruction)
          assignment = '%10d' % assignment if assignment != None else '      None'
          print assignment,
          print '%10s' % ('(%s)' % op,),
        else:
          print '%20s' % op,

      print

    for child in node.out_edges:
      if self.visited.get(child, False):
        self.loop_pair[child] = node
        continue

      self.visited[child] = True

      self.deconstruct_basic_block(child)

  def deconstruct_ssa(self):
    """Deconstruct SSA form along with inserting instructions for spills.
    """
    self.ssa_deconstructed_instructions = []

    # Reset visited dictionary for another traversal.
    self.visited = {}
    for dom_tree in self.ssa.cfg.dom_trees:
      print
      print
      print
      self.deconstruct_basic_block(dom_tree.other_universe_node)


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
    is_allocated, failed_subgraph = regalloc.allocate()
    regalloc.deconstruct_ssa()

    # If an allocation fails there is no point continuing, bail out.
    if not is_allocated:
      exit(1)

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
