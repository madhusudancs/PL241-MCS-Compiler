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

This code is architecture independent. The register assignments are mapped to
real machine registers just before binaries generation. So this file can still
be used for any architecture. The number of registers that must be used for
coloring is a constructor parameter to the RegisterAllocator class.
"""


import collections
import logging
import subprocess
import sys

from argparse import ArgumentParser

from datastructures import InterferenceGraph
from datastructures import InterferenceNode
from datastructures import LiveIntervalsHeap
from ir import is_variable
from ir import is_variable_or_label
from ir import Immediate
from ir import Instruction
from ir import IntermediateRepresentation
from ir import Memory
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


class RegisterAllocationFailedException(Exception):
  """Represents the exception when register allocation cannot complete.
  """

  def __init__(self, msg, *args, **kwargs):
    """Constructs the exception with the message.

    Args:
      msg: The exception message (this is optional).
    """
    super(RegisterAllocationFailedException, self).__init__(*args, **kwargs)
    self.msg = msg if msg else ''

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

    # The memory object this register points to.
    self.memory = None

    # The instruction object where the register is defined, part of the
    # def-use chain for the register.
    self.def_instruction = None

    # The list of instructions where the register is defined, part of the
    # def-use chain for the register.
    self.use_instructions = []

    # Flag to indicate if use_instructions was sorted.
    self.sorted = False

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
    self.sorted = False
    self.use_instructions.extend(instructions)

  def uses(self):
    """Returns the set of instruction objects where this register is used.

    Returns an empty set if it is not used anywhere.
    """
    if not self.sorted:
      self.use_instructions.sort(key=lambda i: i.label)
    return self.use_instructions

  def assignment(self, instruction):
    """Returns the color assignment for the register at the given instruction.

    Actually returns the register with the correct color assigned and
    returns the spill dictionary if spill is requried.

    This method is extremely intelligent :-P If the node is spilled at the
    given instruction it returns the color assigned to the new register. And
    this chains as long as the spill cycle exists :-)

    Args:
      instruction: The instruction object at which the register assignment
          should be obtained.
    """
    if not self.spill:
      return self, None

    # Either when the register is not spilled at all as above or spilled but
    # at any instruction before spill the same register is used.
    if instruction.label < self.spill['spilled_at'].label:
      return self, None

    # If the register is spilled and the current instruction is later than
    # or at the instruction where this register needs to be reloaded we
    # delegate to the reloaded register to return its assignment.
    # NOTE: This gets recursive, if the reloaded register is spilled again.
    # This is very nice because we need not do this in a loop individually
    # for all the chained spills.
    if instruction.label == self.spill['spilled_to'].label:
      new_register, new_spill = self.spill['register'].assignment(instruction)
      if new_spill:
        return new_register, new_spill
      else:
        return new_register, self.spill

    elif instruction.label >= self.spill['spilled_to'].label:
      new_register, new_spill = self.spill['register'].assignment(instruction)
      if new_spill:
        return new_register, new_spill
      else:
        return new_register, None

    # This occurs only in case of variables of the loop which doesn't
    # change during the entire loop. Loop invariants.
    elif not self.spill['spilled_to']:
      return self, spill

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

  def assignments_equal(self, register):
    """Checks if the two registers have the same color assignment.
    """
    return True if (isinstance(register, self.__class__) and
        self.color == register.color) else False

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
    return True if (isinstance(register, self.__class__) and
        self.name == register.name and
        self.def_instruction == register.def_instruction) else False

  def __str__(self):
    """Returns the string representation for this register (preceded by r).
    """
    return 'r%d' % self.name


class RegisterAllocator(object):
  """Allocates the registers for the given SSA form of the source code.
  """

  def __init__(self, ssa, num_registers=4):
    """Initializes the allocator with SSA.

    Args:
      ssa: the SSA object for the program IR in SSA form.
      num_registers: Number of Machine registers available.
    """
    self.ssa = ssa

    self.num_registers = num_registers

    # Reset the register counter for this register allocator.
    Register.reset_name_counter()

    # Dictionary whose keys are opernads (or variables) and the values are
    # the registers assigned in the virtual registers space.
    self.variable_register_map = {}

    # List of function parameters. This list is used for pre-coloring
    self.function_parameters = []

    # Dictionary containing the live ranges for each register
    self.live_ranges = {}

    # Interference graph for this allocator.
    self.interference_graph = None

    # Dictionary of color assigned SSA deconstructed instructions.
    # Key is the original label of the instruction and value is the list of
    # instructions including the spill/reload in the order.
    self.ssa_deconstructed_instructions = collections.defaultdict(list)


  def assign_memory(self, operand, register):
    """Assign the memory object to the register based on the symbol table.

    Args:
      operand: The operand that this register belongs to.
      register: The register which should have a corresponding memory location.
    """
    # We will not allocate memory for non-variable registers, i.e. the result
    # of the instructions right now. This is because most of them may not
    # need a memory. We will allocate memory to them as they are needed, in
    # case of spills.
    if is_variable(operand):
      variable, ssanumber = operand.rsplit('_', 1)
      symtab_entry = self.ssa.ir.local_symbol_table[variable]

      if 'memory' in symtab_entry:
        register.memory = symtab_entry['memory']
        return symtab_entry['memory']
      else:
        memory = Memory(name=variable, scope=self.ssa.ir.function_name)
        register.memory = memory
        symtab_entry['memory'] = memory
        return memory

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
      self.assign_memory(operand, register)
      return register

  def virtual_reg_basic_block(self, start_node, node):
    """Assign virtual registers to all the statements in a basic block.

    Args:
      start_node: The start node of the CFG
      node: The current node that is to be processed for assigning virtual
          registers.
    """
    if node.phi_functions:
      start_node.phi_nodes.append(node)

      for phi_function in node.phi_functions.values():
        # The LHS of the phi function is actually the result the function
        # RHS are the operands.
        operand = phi_function['LHS']
        new_register = self.register_for_operand(operand)
        new_register.set_def(self.ssa.ir.ir[node.value[0]])
        phi_function['LHS'] = new_register
        self.variable_register_map[operand] = new_register

    for instruction in self.ssa.optimized(node.value[0], node.value[1] + 1):
      if instruction.instruction == '.end_':
        continue
      elif instruction.instruction == '.begin_':
        new_operands = []
        for operand in instruction.operands:
          if instruction.is_variable_or_label(operand):
            new_register = self.register_for_operand(operand)
            new_register.set_def(instruction)
            new_operands.append(new_register)

        instruction.operands = new_operands
        self.function_parameters = instruction.operands
      else:
        # The first operand of the branch instruction should still be a label.
        # We directly assign that off as the assigned operand since no register
        # or memory allocation is required for it.
        if instruction.instruction == 'bra':
          instruction.assigned_operand1 = instruction.operand1
          continue

        # The second operand of the branch instruction should still be a label.
        # We directly assign that off as the assigned operand since no register
        # or memory allocation is required for it.
        if instruction.instruction in ['beq', 'bne', 'blt',
                                       'ble', 'bgt', 'bge']:
          if self.is_register(cmp_instruction.result):
            if instruction.is_variable_or_label(instruction.operand1):
              instruction.operand1 = self.register_for_operand(
                  instruction.operand1)
              instruction.operand1.set_use(instruction)
          instruction.assigned_operand2 = instruction.operand2
          continue

        if instruction.is_variable_or_label(instruction.operand1):
          instruction.operand1 = self.register_for_operand(
              instruction.operand1)
          instruction.operand1.set_use(instruction)


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
          elif self.is_memory(operand):
            new_operands.append(operand)

        instruction.operands = new_operands

        # After Copy propagation the only move instructions that remain
        # are for the function eplilogue, prologue and at the callee site
        # we need not allocate a register for results of these instructions.
        # We also need not allocate registers for the result of the return
        # instruction
        if instruction.instruction in ['move', 'store', 'ret']:
          continue

        # FIXME: This is architecture specific
        # No need to assign a register for the compare instruction for x86_64
        # since the architecture uses special flag register for this which
        # is not the General Purpose Register.
        # However if one of the operands is not a register, will want a
        # register to hold
        if instruction.instruction == 'cmp':
          cmp_instruction = instruction
          if not (
              self.is_memory(instruction.operand1) and
              self.is_memory(instruction.operand2)):
            continue

        # Assign a register for the result of the instruction
        register = Register()
        register.set_def(instruction)
        self.variable_register_map[instruction.label] = register
        instruction.result = register

    for child in node.out_edges:
      if self.visited.get(child, False):
        continue

      self.visited[child] = True

      self.virtual_reg_basic_block(start_node, child)

  def allocate_virtual_registers(self, start_node):
    """Allocate registers in virtual infinite space of registers.

    This is the most important phase because, we actually explicitly allocate
    some register for the result of the instructions in this phase.
    """
    self.visited = {}

    self.virtual_reg_basic_block(start_node, start_node)

    # Process all the phi-functions in the end of the function because
    # there are phi-functions especially in case of blocks whose operands
    # are defined after this block. And also it may so happen that, that
    # definition is removed because of copy propagation and only some
    # other instructions label remaining. This cannot be determined or
    # fixed before the result of the instruction whose result is the
    # operand for the phi function is computed
    for phi_node in start_node.phi_nodes:
      for phi_function in phi_node.phi_functions.values():
        for i, operand in enumerate(phi_function['RHS']):
          if is_variable_or_label(operand):
            new_register = self.register_for_operand(operand)
            use = self.ssa.ir.ir[phi_node.in_edges[i].value[1]]
            new_register.set_use(use)
            new_register.use_instructions.sort(
                key=lambda i: i.label)

            phi_function['RHS'][i] = new_register

  def analyze_basic_block_liveness(self, start_node, node):
    """Analyzes the liveness of the variables in the given basic block

    Performs a post-order traversal of the control flow graph for processing
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
        continue

      self.visited[child] = True

      self.analyze_basic_block_liveness(start_node, child)

    # The live variables set in the block where each key is the variable and
    # the value is a two elements first representing the start of the range
    # and second representing the end of the range.
    live = {}
    intervals = {}
    phi_operands = {}

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
        continue

      if instruction.instruction == '.begin_':
        for operand in instruction.operands:
          if self.is_register(operand):
            if operand in live:
              intervals[operand][0] = node.value[0]

            # Else, since we are traversing in the reverse order, begin
            # instruction of the function is the last function we encounter,
            # so, if this register is not live anywhere, it is Dead-on-Arrival
            # so don't bother about doing anything for it, just pass


        continue

      if self.is_register(instruction.result):
        if instruction.result not in live:
          # Dead-on-Arrival. I love Dead-on-Arrival stuff, more
          # optimizations! :-P
          # NOTE: pop is not added because it doesn't even exist
          # So don't bother about doing anything.
          if instruction.instruction != 'call':
            continue
          else:
            # For call instruction process its other operands.
            pass
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
      if phi_function['LHS'] not in intervals:
        intervals[phi_function['LHS']] = [None, None]
      else:
        intervals[phi_function['LHS']][0] = node.value[0]
        live.pop(phi_function['LHS'])

      phi_operands[phi_function['LHS']] = True

    for phi_function in node.phi_functions.values():
      for i, operand in enumerate(phi_function['RHS']):
        include[node.in_edges[i]].append(operand)
        phi_operands[operand] = True

        if operand in intervals:
          intervals[operand][1] = node.value[0]
        # Don't do else part, that over writes registers that get defined
        # inside the loop

    # Every operand that is live at the loop header and is not a phi-operand
    # or a phi-result should live from the beginning of the block to the end.
    # This is a guard to not spill these registers before other registers
    # whose next use is farther outside the loop.
    if node.loop_header:
      for operand in live:
        if operand not in phi_operands:
          # node's loop_header property points to loop footer.
          loop_footer = node.loop_header
          # Lives till the end of the loop footer block. Just make it to live
          # out of the block too by adding 1
          intervals[operand][1] = loop_footer.value[1]

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

    self.live_intervals = start.live_intervals

    for phi_node in start.phi_nodes:
      for phi_function in phi_node.phi_functions.values():
        for operand in phi_function['RHS']:
          if operand in self.live_intervals:
            if (operand.definition().label > operand.uses()[-1].label):
              self.live_intervals[operand][0] = operand.definition().label

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
        next_use_index, next_use = -use_index-2, use

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

    # Spill the register with the next farthest use.
    spilling_node = next_farthest_use['collision']
    spill_register = spilling_node.register

    # Create a new register. The last register count is the count of the
    # number of registers assigned, the actual name of the last register
    # will be -1 of this value. So we directly assign this value to the new
    # register.
    # FIXME: We do not have to do this name assignments if we compile each
    # function independently
    new_register = Register()
    new_register.memory = spill_register.memory

    new_register.set_def(next_farthest_use['next_use'])
    new_register.set_use(
        *spill_register.uses()[next_farthest_use['next_use_index']:])

    # Push the new register down the heap. If there is no next use, then
    # it should probably be part of the loop. So set things at the end of
    # loop.
    if next_farthest_use['next_use']:
      self.live_intervals_heap.push(new_register,
          [next_farthest_use['next_use'].label,
         spilling_node.instructions[1]])

    # Cut short the instruction range for the spilled register node.
    spilling_node.instructions[1] = current_instruction.label

    # Update the spill information for the spilled register.
    spill_register.spill = {
        'spilled_at': current_instruction,
        'spilled_to': next_farthest_use['next_use'],
        'register': new_register
        }

    # Remove this from the collisions list of the currently
    # processing register.
    collisions.remove(spilling_node)

    return collisions

  def populate_collisions(self, with_spill=False):
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

      if with_spill:
        collisions = self.spill(current_node, collisions)

      current_node.append_edges(*collisions)
      self.register_nodes[register] = current_node

  def build_interference_graph(self, with_spill=False):
    """Builds the interference graph for the given control flow subgraph.

    Args:
      live_intervals: Dictionary containing the live intervals for the
          entire function.
      phi_nodes: Contains all the phi nodes for the given function in the
          program.
    """
    # Clear the register_nodes dictionary for a new start
    self.register_nodes = {}

    self.live_intervals_heap = LiveIntervalsHeap(self.live_intervals)

    self.populate_collisions(with_spill)

    nodes = self.register_nodes.values()

    interference_graph = InterferenceGraph(nodes)
    self.interference_graph = interference_graph

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

  def precolor(self):
    """Generate clauses for precolored registers like function parameters.

    As per Linux AMD64 ABI function calling convention, function arguments
    must be passed in %rdi, %rsi, %rdx, %rcx, %r8 and %r9 in that order and
    the remaining arguments on the stack.So by forcing the function parameters
    to be in those registers by precoloring, we reduce the number of moves
    required.

    NOTE: This can still be architecture independent because to ensure that
    we use the architecture specific calling convention, we just need to
    name the registers in such a way that function argument registers have
    the same color as AMD64 function arguments registers.
    """
    # Imported here because of cyclic import issues.
    from x86_64 import FUNCTION_ARGUMENTS_COLORS

    max_reg_binary_len = len(bin(self.num_registers - 1)[2:])

    clauses = []

    for register, color in zip(
        self.function_parameters, FUNCTION_ARGUMENTS_COLORS):
      colorbits = bin(color)[2:]

      num_leading_zeros = 0
      while num_leading_zeros < (max_reg_binary_len - len(colorbits)):
        cnf_var = self.get_cnf_var(register, num_leading_zeros)
        clauses.append('-%s ' % (cnf_var))
        num_leading_zeros += 1

      for bit_position, bit in enumerate(colorbits):
        cnf_var = self.get_cnf_var(register, bit_position + num_leading_zeros)

        if bit == '0':
          clauses.append('-%s ' % (cnf_var))
        else:
          clauses.append('%s ' % (cnf_var))

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

    precolored_clauses = self.precolor()
    clauses.extend(precolored_clauses)

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
      # We cannot do this processing until we find all the interferences at
      # the definition point of the current register.
      # collisions = self.spill(current_node, collisions)
      LOGGER.debug('SAT Unsatisfiable')
      return False
    elif lines[-3] == 's SATISFIABLE':
      assignment = lines[-2]
      self.generate_assignments(assignment)
      LOGGER.debug('SAT Satisfiable! Allocation: %s' % (lines[-1]))
      return True

  def allocate(self):
    """Allocate the registers to the program.
    """
    start_node = self.ssa.cfg[0]
    self.allocate_virtual_registers(start_node)

    self.liveness(start_node)

    ifg = self.build_interference_graph()
    is_allocated = self.sat_solve(ifg)
    if is_allocated:
      LOGGER.debug('Allocated for subgraph %s!' % (
          start_node))
    else:
      ifg = self.build_interference_graph(with_spill=True)
      is_allocated = self.sat_solve(ifg)
      if not is_allocated:
        LOGGER.debug('Allocation Failed for subgraph %s :-(' % (
          start_node))
        # No point in proceeding if register allocation fails. Some major
        # bug in the code. So bail out.
        return False

    return True

  def is_register(self, operand):
    """Checks if the given operand is actually a register.
    """
    return True if (operand and isinstance(operand, Register)) else False

  def is_memory(self, operand):
    """Checks if the given operand is actually a memory object.
    """
    return True if (operand and isinstance(operand, Memory)) else False

  def is_immediate(self, operand):
    """Checks if the given operand is actually an immediate operand object.
    """
    return True if (operand and isinstance(operand, Immediate)) else False

  def str_virtual_register_allocation(self):
    """Gets the text representation of the program after virtual allocation.
    """
    bfs_queue = [self.ssa.cfg[0]]
    visited = set([])
    virtual_reg_blocks = []

    start_labels_to_blocks = {}

    while bfs_queue:
      virtual_reg = ''
      node = bfs_queue.pop(0)
      if node in visited:
        continue

      visited.add(node)
      bfs_queue.extend(node.out_edges[::-1])

      for phi_function in node.phi_functions.values():
        virtual_reg += '%10s  <- %4s: %5s' % (phi_function['LHS'], '', 'phi')

        for operand in phi_function['RHS']:
          virtual_reg += '%50s' % operand

        virtual_reg += '\n'

      start_labels_to_blocks[len(virtual_reg_blocks)] = node.value[0]
      for instruction in self.ssa.optimized(node.value[0], node.value[1] + 1):
        virtual_reg += '%10s  <- %s\n' % (
           instruction.result if instruction.result else '', instruction)

      virtual_reg_blocks.append(virtual_reg)

    # Sort the basic blocks according to their start instruction label
    sorted_blocks = sorted(
        enumerate(virtual_reg_blocks),
        key=lambda k: start_labels_to_blocks[k[0]])

    return '\n'.join([b[1] for b in sorted_blocks])

  def generate_assigned_instructions(self):
    """Generates instructions which have registers assigned and phi-resolved.
    """
    assigned_ssa = []

    for instruction in self.ssa.ir.ir:
      if instruction.label in self.ssa.ir.start_node_map:
        instructions = []

      if instruction in self.ssa_deconstructed_instructions:
        if self.ssa_deconstructed_instructions[instruction]:
          first, last = (self.ssa_deconstructed_instructions[instruction][0],
                         self.ssa_deconstructed_instructions[instruction][-1])
        assigned_ssa.extend(self.ssa_deconstructed_instructions[instruction])

        instructions.extend(self.ssa_deconstructed_instructions[instruction])

      if instruction.label in self.ssa.ir.end_node_map:
        node = self.ssa.ir.end_node_map[instruction.label]
        node.instructions = instructions

      if instruction in self.ssa_deconstructed_instructions:
        if self.ssa_deconstructed_instructions[instruction]:
          first.label, last.label = last.label, first.label

    self.ssa.ir.ir = assigned_ssa

  def insert_instruction(self, instruction, following_instruction):
    """Inserts the given instruction before the following instruction.

    Args:
      instruction: The instruction that must be inserted.
      following_instruction: The instruction before which this instruction
          must be inserted
    """
    self.ssa_deconstructed_instructions[following_instruction].insert(
        0, instruction)

  def insert_spill(self, register, spilled_at, memory):
    """Insert a spill instruction.

    Args:
      register: The original register that was spilled.
      spilled_at: The instruction before which the spill must be inserted.
      memory: The memory to spill to.
    """
    store_instruction = Instruction('store', register, memory)
    store_instruction.assigned_operand1 = register
    store_instruction.assigned_operand2 = memory
    self.insert_instruction(store_instruction, spilled_at)

  def insert_reload(self, memory, original_register, spilled_to, new_register):
    """Insert a reload instruction.

    Args:
      memory: The memory to reload from.
      original_register: The original register where this was.
      spilled_to: The instruction before which reload must be inserted.
      new_register: The new register to which this must be reloaded.
    """
    reload_instruction = Instruction('load', memory)
    reload_instruction.result = original_register
    reload_instruction.assigned_result = new_register
    reload_instruction.assigned_operand1 = memory
    self.insert_instruction(reload_instruction, spilled_to)

  def insert_spill_reload(self, register, spill):
    """Inserts the spill and reload instructions.

    Args:
      register: The new register that must be reloaded to.
      spill: Dictionary containing the spill information.
    """
    if not spill:
      return

    if register.memory:
      new_memory = register.memory
    else:
      new_memory = Memory(scope=self.ssa.ir.function_name)
      register.memory = new_memory

    self.insert_spill(register, spill['spilled_at'], new_memory)
    self.insert_reload(new_memory, register, spill['spilled_to'],
                       spill['register'])

  def resolve_noncontiguous_blocks(self, node, result, assignment, spilled):
    """Insert reload/move instructions for phi operands and other registers.

    Args:
      node: The node at the end of which these instructions must be inserted.
      result: The result instruction/destination of the movement.
      assignment: The source of the movement.
      spill: Dictionary containing spill information.
    """
    if spilled:
      if assignment.memory:
        memory = assignment.memory
      else:
        memory = Memory(scope=self.ssa.ir.function_name)
        assignment.memory = memory

      self.insert_spill(assignment, spilled['spilled_at'], memory)

      reload_instruction = Instruction('load', memory)
      reload_instruction.result = result
      reload_instruction.assigned_result = result
      reload_instruction.assigned_operand1 = memory
      self.phi_map[node].append(reload_instruction)
    # We don't do anything for the case where assignment and
    # phi_result are the same. They just remain the way they are :-)
    elif not (self.is_register(assignment) and
        assignment.assignments_equal(result)):
      move_instruction = Instruction('move', assignment, result)
      move_instruction.result = None
      move_instruction.assigned_result = None
      move_instruction.assigned_operand1 = assignment
      move_instruction.assigned_operand2 = result
      self.phi_map[node].append(move_instruction)

  def deconstruct_basic_block(self, node):
    """Processes basic blocks by performing a pre-order traversal.

    Args:
      node: The basic block node to process.
    """
    phi_operands = {}

    for phi_function in node.phi_functions.values():
      # The result of this phi instruction is never used, so no move
      # instructions.
      if (self.is_register(phi_function['LHS']) and
          phi_function['LHS'].color == None):
        continue

      # We should not add phi instruction to the resulting instructions.
      phi_operands[phi_function['LHS']] = True

      for i, predecessor in enumerate(node.in_edges):
        operand = phi_function['RHS'][i]
        if self.is_register(operand):
          assignment, spilled = operand.assignment(
              self.ssa.ir.ir[predecessor.value[1]])
        else:
          assignment, spilled = operand, None

        # Note we want original registers not the assigned registers. This
        # is because the live intervals store the original registers not
        # the assignments.
        phi_operands[operand] = True

        self.resolve_noncontiguous_blocks(predecessor, phi_function['LHS'],
                                          operand, spilled)

    for instruction in self.ssa.optimized(node.value[0], node.value[1] + 1):
      self.ssa_deconstructed_instructions[instruction].append(instruction)

      result = instruction.result
      if self.is_register(result):
        # Since we are using SSA there won't be any reloads for the result
        # So we don't call the corresponding insertion method.
        instruction.assigned_result, spill = result.assignment(instruction)

      operand1 = instruction.operand1
      if self.is_register(operand1):
        instruction.assigned_operand1, spill = operand1.assignment(
            instruction)
        self.insert_spill_reload(operand1, spill)
      elif self.is_memory(operand1) or self.is_immediate(operand1):
        instruction.assigned_operand1 = operand1

      operand2 = instruction.operand2
      if self.is_register(operand2):
        instruction.assigned_operand2, spill = operand2.assignment(
            instruction)
        self.insert_spill_reload(operand2, spill)
      elif self.is_memory(operand2) or self.is_immediate(operand2):
        instruction.assigned_operand2 = operand2

      operands_assigned = []
      for operand in instruction.operands:
        if self.is_register(operand):
          operand_assigned, spill = operand.assignment(instruction)
          self.insert_spill_reload(operand, spill)
          operands_assigned.append(operand_assigned)
        elif self.is_memory(operand) or self.is_immediate(operand):
          operands_assigned.append(operand)

      instruction.assigned_operands = operands_assigned

    for child in node.out_edges:
      if self.visited.get(child, False):
        continue

      self.visited[child] = True

      self.deconstruct_basic_block(child)

    if node.loop_header:
      loop_footer = node.loop_header
      loop_last_instruction = self.ssa.ir.ir[loop_footer.value[1]]
      while loop_last_instruction in self.ssa.optimized_removal:
        loop_last_instruction -= 1

      for operand in node.live_in:
        if operand not in phi_operands:
          # Even if it is spilled we don't insert a reload here, we just
          # make sure that the loop footer moves to the required assignment.
          assignment, spilled = operand.assignment(
              self.ssa.ir.ir[node.value[0]])
          new_assignment, new_spilled = operand.assignment(
              loop_last_instruction)
          # Lives till the end of the loop footer block.
          if new_spilled and new_spilled['spilled_to'] == None:
            new_spilled['spilled_to'] = loop_last_instruction

          if not spilled:
            self.resolve_noncontiguous_blocks(loop_footer,
                                              assignment, new_assignment,
                                              new_spilled)
          # In extremely rare cases when both are spilled a memory to memory
          # move is required which we just represent by a mmove instruction
          # for now.
          elif spilled and new_spilled:
            raise RegisterAllocationFailedException('A memory to memory move '
                'is required for Register Allocation to complete while '
                'resolving spills across loops and there is no memory to '
                'memory move instruction available. Try to recompile.')
            loop_enter_memory = \
                self.ssa_deconstructed_instructions[
                    spilled['spilled_to']][0].assigned_operand1
            if new_assignment.memory:
              new_memory = new_assignment.memory
            else:
              new_memory = Memory(scope=self.ssa.ir.function_name)
              new_assignment.memory = new_memory

            self.insert_spill(new_assignment, spilled['spilled_at'],
                              new_memory)
            mmove_instruction = Instruction('mmove', new_memory,
                                            loop_enter_memory)
            mmove_instruction.result = None
            mmove_instruction.assigned_result = None
            mmove_instruction.assigned_operand1 = new_memory
            mmove_instruction.assigned_operand2 = loop_enter_memory
            self.phi_map[node].append(mmove_instruction)

  def deconstruct_ssa(self):
    """Deconstruct SSA form along with inserting instructions for spills.
    """
    # Reset visited dictionary for another traversal.
    self.visited = {}

    self.phi_map = collections.defaultdict(list)
    self.deconstruct_basic_block(self.ssa.cfg[0])

    def key_func(instruction):
      """Returns the sort key for the phi's resolved instructions.
      """
      if instruction.instruction == 'move':
        return instruction.assigned_operand2, instruction.assigned_operand1
      elif instruction.instruction == 'load':
        return instruction.assigned_result, instruction.assigned_operand1

    def cmp_func(pair1, pair2):
      """Defines how the result, operand pair should be compared.
      """
      if (pair1[0].assignments_equal(pair2[1]) and
          pair2[0].assignments_equal(pair1[1])):
        # Since two move instructions are such that the result of one is the
        # operand for the other and hence they cannot be ordered, we insert
        # a XCHG instruction.
        # Architectures that don't support XCHG can implement this using
        # 3 XOR instructions.
        xchg_map[pair1] = pair2
        xchg_map[pair2] = pair1
        return 0
      elif pair1[0].assignments_equal(pair2[1]):
        return 1
      elif pair2[0].assignments_equal(pair1[1]):
        return -1
      else:
        return 0

    # Insert the respective instructions for phi-functions in the predecissor.
    for predecessor in self.phi_map:
      xchg_map = {}
      instructions = sorted(self.phi_map[predecessor],
                            cmp=cmp_func, key=key_func)
      if xchg_map:
        new_instructions = []
        xchg_processed = set([])
        for instruction in instructions:
          if instruction.instruction == 'move':
            pair = instruction.operand2, instruction.operand1
          elif instruction.instruction == 'load':
            pair = instruction.result, instruction.operand1
          if pair in xchg_map:
            if pair not in xchg_processed:
              xchg_processed.add(pair)
              xchg_processed.add(xchg_map[pair])
              xchg_instruction = Instruction('xchg', pair[0], pair[1])
              xchg_instruction.assigned_operand1 = pair[0]
              xchg_instruction.assigned_operand2 = pair[1]
              new_instructions.append(xchg_instruction)
          else:
            new_instructions.append(instruction)

        instructions = new_instructions

      if predecessor.value[1] in self.ssa.optimized_removal:
        last_existing_label = predecessor.value[1] - 1
        while last_existing_label in self.ssa.optimized_removal:
          last_existing_label -= 1

        instruction = self.ssa.ir.ir[last_existing_label]
        self.ssa_deconstructed_instructions[instruction].extend(instructions)
      elif self.ssa.ir.ir[predecessor.value[1]].instruction == 'bra':
        instruction = self.ssa.ir.ir[predecessor.value[1]]
        self.ssa_deconstructed_instructions[instruction] = instructions + \
            self.ssa_deconstructed_instructions[instruction]
      elif (self.ssa.ir.ir[predecessor.value[1] - 1].instruction == 'cmp' and
          self.ssa.ir.ir[predecessor.value[1]].instruction in [
          'beq', 'bne', 'blt', 'ble', 'bgt', 'bge']):
        instruction = self.ssa.ir.ir[predecessor.value[1] - 1]
        self.ssa_deconstructed_instructions[instruction] = instructions + \
            self.ssa_deconstructed_instructions[instruction]
      else:
        instruction = self.ssa.ir.ir[predecessor.value[1]]
        self.ssa_deconstructed_instructions[instruction].extend(instructions)

    self.generate_assigned_instructions()

  def registers_assigned_instructions(self):
    """Return the printable string for the registers assigned instructions
    """
    instructions = []
    for instruction in self.ssa.optimized():
      instruction_str = ''
      if self.is_register(instruction.result):
        assignment = instruction.result.assignment(instruction)
        instruction_str += '%10d ' % assignment if assignment != None else \
            '      None'
        instruction_str += ' %10s ' % ('(%s)' % instruction.result,)
      else:
        instruction_str += '                      '

      instruction_str += '%10s ' % instruction.label
      instruction_str += '     '
      instruction_str += '%10s ' % instruction.instruction

      if self.is_register(instruction.operand1):
        assignment = instruction.operand1.assignment(instruction)
        instruction_str += '%10d ' % assignment if assignment != None else \
            '      None '
        instruction_str += assignment
        instruction_str += ' %10s ' % ('(%s)' % instruction.operand1,)
      else:
        instruction_str += '%20s ' % instruction.operand1

      if self.is_register(instruction.operand2):
        assignment = instruction.operand2.assignment(instruction)
        instruction_str += '%10d ' % assignment if assignment != None else \
            '      None '
        instruction_str += assignment
        instruction_str += ' %10s ' % ('(%s)' % instruction.operand2,)
      else:
        instruction_str += '%20s ' % instruction.operand2

      for op in instruction.operands:
        if self.is_register(op):
          assignment = op.assignment(instruction)
          instruction_str += '%10d ' % assignment if assignment != None else \
              '      None '
          instruction_str += assignment
          instruction_str += ' %10s ' % ('(%s)' % op,)
        else:
          instruction_str += '%20s ' % op

      instructions.append(instruction_str)

    return '\n'.join(instructions)

  def almost_machine_instructions(self):
    """Return the printable string for the SSA deconstructed instructions.
    """
    instructions = []
    for ssa_instruction in self.ssa.ir.ir:
      if ssa_instruction.instruction == 'phi':
        continue

      for instruction in self.ssa_deconstructed_instructions[ssa_instruction]:
        instruction_str = ' ' * 5
        if self.is_register(instruction.result):
          assignment = instruction.assigned_result
          instruction_str += '%-4d' % assignment.color if \
              assignment.color != None else '      None'
          instruction_str += '%-10s' % ('(%s)' % instruction.result,)
        else:
          instruction_str += ' ' * 14

        instruction_str += '%-10s' % ('%s:' % (instruction.label,))
        instruction_str += '%-30s' % instruction.instruction

        if self.is_register(instruction.operand1):
          assignment = instruction.assigned_operand1
          instruction_str += '%-10d' % assignment.color if \
              assignment.color != None else ('' * 6 + 'None')
          instruction_str += '%-10s' % (assignment,)
          instruction_str += '%-30s' % ('(%s)' % instruction.operand1,)
        else:
          instruction_str += '%-50s' % instruction.operand1

        if self.is_register(instruction.operand2):
          assignment = instruction.assigned_operand2
          instruction_str += '%-10d' % assignment.color if \
              assignment.color != None else '      None '
          instruction_str += '%s   ' % (assignment,)
          instruction_str += '%-10s' % ('(%s)' % instruction.operand2,)
        else:
          instruction_str += '%-20s' % instruction.operand2

        for op in getattr(instruction, 'assigned_operands', []):
          if self.is_register(op):
            assignment = op
            instruction_str += '%-10d' % assignment.color if \
                assignment.color != None else '      None '
            instruction_str += '%s   ' % (assignment,)
            instruction_str += '%-10s' % ('(%s)' % op,)
          else:
            instruction_str += '%-20s' % op

        instructions.append(instruction_str)

    return '\n'.join(instructions)



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
  parser.add_argument('--assigned', metavar="Assigned", type=str,
                      nargs='?', const=True,
                      help='Generate the instructions with registers '
                      'and phi functions resolved.')
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
    is_allocated = regalloc.allocate()
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

    if args.assigned:
      external_file = isinstance(args.assigned, str)
      assigned_file = open(args.assigned, 'w') if external_file else sys.stdout
      assigned_file.write('%s\n' % regalloc.almost_machine_instructions())

      if external_file:
        assigned_file.close()

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
