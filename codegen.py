# Copyright 2012 Madhusudan C.S.
#
# This file codegen.py is part of PL241-MCS compiler.
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
#

"""This module generates the code for our intermediate representation.

The intermediate form is assumed to have registers allocated. As much as an
effort is made to keep this machine independent, it is impossible since we
will have to make use of architecture specific details from this phase.

The instructions that will be processed are:
    add x y               addition
    sub x y               subtraction
    mul x y               multiplication
    div x y               division
    cmp x y               comparison
    adda x y              add two addresses x und y (used only with arrays)
    load y                load from memory address y
    store y x             store y to memory address x
    move y x              assign x := y
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
    call x1, x2, ...      Call instruction

Additionally we also handle prologues and epilogues of all the function
defintions.
"""


import collections
import logging

from argparse import ArgumentParser

from ir import Immediate
from ir import IntermediateRepresentation
from ir import Memory
from optimizations import Optimize
from parser import Parser
from regalloc import Register
from regalloc import RegisterAllocator

# Architecture specific imports
from x86_64 import ADD
from x86_64 import CALL
from x86_64 import CMP
from x86_64 import FUNCTION_ARGUMENTS_COLORS
from x86_64 import IDIV
from x86_64 import IMUL
from x86_64 import JE
from x86_64 import JG
from x86_64 import JGE
from x86_64 import JL
from x86_64 import JLE
from x86_64 import JMP
from x86_64 import JNE
from x86_64 import LEA
from x86_64 import MEMORY_WIDTH
from x86_64 import MOV
from x86_64 import POP
from x86_64 import PUSH
from x86_64 import REGISTERS_COLOR_SET
from x86_64 import RET
from x86_64 import SUB
from x86_64 import XCHG
from x86_64 import XOR


# Module level logger object
LOGGER = logging.getLogger(__name__)


def allocate_global_memory(global_symbol_table):
  """Allocate memory for global datastructures.

  Args:
    global_symbol_table: Dictionary containing the global symbol table whose
        keys are symbol names and values are symbol table data like memory
        object for this symbol and type.
  """
  memory_offset = 0
  for symbol, symtab_entry in global_symbol_table.iteritems():
    if symtab_entry.get('type') == 'function_name' or (
        symbol in ['InputNum', 'OutputNum', 'OutputNewLine']):
      # These symbols are function names, don't do anything for them here.
      continue
    elif symtab_entry.get('dimensions', None) != None:
      # This entry is an array
      if 'memory' not in symtab_entry:
        symtab_entry['memory'] = Memory()

      symtab_entry['memory'].offset = memory_offset
      total_size = 1
      for dimension in symtab_entry['dimensions']:
        total_size *= dimension

      symtab_entry['memory'].size = total_size * MEMORY_WIDTH
      symtab_entry['memory'].base = 'rip'

      memory_offset += total_size * MEMORY_WIDTH
    else:
      # This entry is a regular integer.
      if 'memory' not in symtab_entry:
        symtab_entry['memory'] = Memory()

      symtab_entry['memory'].offset = memory_offset
      symtab_entry['memory'].size = MEMORY_WIDTH
      symtab_entry['memory'].base = 'rip'

      memory_offset += MEMORY_WIDTH

  return memory_offset


class CodeGenerator(object):
  """Generates the code for the given SSA object.
  """

  def __init__(self, ir):
    """Constructs the datastructures required for code generation.

    Args:
      ir: The Intermediate Representation object which contains the registers
          allocated instructions.
    """
    self.ir = ir

    # Final binary generated for the function
    self.binary = None

    # Contains the set of instructions generated.
    self.instructions = []

    # Keeps track of the number of bytes of instructions generated to obtain
    # the offset for the next instruction.
    self.instruction_offset = 0

    # Dictionary containing the mapping from instructions to start and end
    # offsets for the instruction
    self.instruction_offsets_map = {}

    # Dictionary containing the live intervals of the registers allocated.
    # Key will be the color of the register and the value will be a two tuple
    # indicating the liveness range.
    self.live_intervals = {}

    # Dictionary containing a mapping from the instruction label to the
    # set of register colors live at that instruction.
    self.instruction_live_registers = collections.defaultdict(dict)

    # A dictionary containing a mapping from the IR instructions' labels
    # to another dictionary which contains the actual instruction and the
    # byte code offset where it begins in the binary.
    self.processed_ir_instructions = {}

    # A dictionary containing IR instruction labels as the key and the values
    # are the list of machine instructions corresponding to it.
    self.ir_machine_map = collections.defaultdict(list)

    # A two-tuple containing the instruction and its original target operand
    # for which the target code is yet to be processed.
    self.targets_to_process = []

    # A two-tuple containing the instruction and its memory object to process
    # to which this call should be linked
    self.globals_to_process = []

    # A two-tuple containing the instruction and its SIB base memory object to
    # process.
    self.global_sibs_to_process = []

    # A two-tuple containing the instruction and its function name
    # to which this call should be linked
    self.calls_to_link = []

    # List of instructions to process for function return targets
    self.returns_to_process = []

    # Memory offset of the current scope.
    self.memory_offset = None

  def analyze_basic_block_liveness(self, node):
    """Analyzes the liveness of the variables in the given basic block

    Performs a post-order traversal of the control flow graph for processing
    the liveness, so the traversal is on the dominator tree.

    The algorithm is similar to the algorithm used for register allocator, but
    with some differences for checking the liveness ranges of register colors:
    and doesn't include any phi instructions, because we have already
    deconstructed the SSA. So we cannot really re-use that code here. We
    instead write a new method for performing the liveness analysis on these
    colored registers to be explicit.

    Args:
      node: The node of the control flow subgraph on which post-order
          traversal should be performed.
    """
    for child in node.out_edges:
      if self.visited.get(child, False):
        continue

      child.live_in = {}
      child.live_intervals = {}
      child.live_include = {}

      self.visited[child] = True

      self.analyze_basic_block_liveness(child)


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

      successor_live_in = set(
          successor.live_in.keys()) - set(exclude)

      for register in successor_live_in:
        # None for first element of the list means it starts from the
        # beginning of the block and None for the second element means
        # it runs until the end of the block. See datastructures.CFGNode
        # __init__() method for matching how things work with live
        # dictionaries.
        if self.is_register(register):
          live[register] = True
          intervals[register] = list([node.instructions[0].label,
                                      node.instructions[-1].label])


    # Walk the instructions in the reverse order, note -1 for stepping.
    for instruction in node.instructions[-1::-1]:
      if instruction.instruction.startswith('.end_'):
        continue

      if instruction.instruction == '.begin_':
        for operand in instruction.assigned_operands:
          if self.is_register(operand):
            if operand not in live:
              intervals[operand] = [node.instructions[0].label,
                                    instruction.label]
              live[operand] = True
            else:
              intervals[operand][0] = node.instructions[0].label
      elif instruction.instruction == 'move':
        if self.is_register(instruction.assigned_operand2):
          if instruction.assigned_operand2 not in live:
            # Dead-on-Arrival. I love Dead-on-Arrival stuff, more
            # optimizations! :-P
            # After all the interval calculations if instructions range remains
            # the same, don't use that register at all. That instruction
            # itself need not be used for generating binaries.
            intervals[instruction.assigned_operand2] = [instruction.label,
                                                        instruction.label]
            # NOTE: pop is not added because it doesn't even exist
          else:
            intervals[instruction.assigned_operand2][0] = instruction.label
            live.pop(instruction.assigned_operand2)

        # We need to process the input operands if and only if they don't
        # really exist, if they already exist, it means that they are used
        # else where after this basic block and hence should stay alive.
        # Only the last use (or the first sighting since we are coming from
        # reverse now should get an lifetime end set.)
        operand1 = instruction.assigned_operand1
        if self.is_register(operand1) and operand1 not in live:
          live[operand1] = True
          intervals[operand1] = [node.instructions[0].label, instruction.label]
      else:
        if self.is_register(instruction.assigned_result):
          if instruction.assigned_result not in live:
            # Dead-on-Arrival. I love Dead-on-Arrival stuff, more
            # optimizations! :-P
            intervals[instruction.assigned_result] = [instruction.label,
                                                      instruction.label]
            # NOTE: pop is not added because it doesn't even exist
          else:
            intervals[instruction.assigned_result][0] = instruction.label
            live.pop(instruction.assigned_result)

        # We need to process the input operands if and only if they don't
        # really exist, if they already exist, it means that they are used
        # else where after this basic block and hence should stay alive.
        # Only the last use (or the first sighting since we are coming from
        # reverse now should get an lifetime end set.)
        operand1 = instruction.assigned_operand1
        if self.is_register(operand1) and operand1 not in live:
          live[operand1] = True
          intervals[operand1] = [node.instructions[0].label, instruction.label]

        operand2 = instruction.assigned_operand2
        if self.is_register(operand2) and operand2 not in live:
          live[operand2] = True
          intervals[operand2] = [node.instructions[0].label, instruction.label]

        for operand in instruction.assigned_operands:
          if self.is_register(operand) and operand not in live:
            intervals[operand] = [node.instructions[0].label, instruction.label]
            live[operand] = True


    # Every operand that is live at the loop header and is not a phi-operand
    # or a phi-result should live from the beginning of the block to the end.
    # This is a guard to not spill these registers before other registers
    # whose next use is farther outside the loop.
    if node.loop_header:
      for operand in live:
        # Loop header attribute points to the loop footer.
        loop_footer = node.loop_header
        # Lives till the end of the loop footer block.
        intervals[operand][1] = loop_footer.instructions[-1].label

    node.live_in = live
    node.live_include = include

    for operand in intervals:
      if operand in self.live_intervals:
        # Merge the intervals
        self.live_intervals[operand] = [
            min(intervals[operand][0], self.live_intervals[operand][0]),
            max(intervals[operand][1], self.live_intervals[operand][1]),
            ]
      else:
        # Add a new interval
        self.live_intervals[operand] = intervals[operand]

  def liveness(self):
    """Performs the live range analysis again on allocated registers.
    """
    old_to_new_labels = {}

    # Preprocessing step to assign new labels, also make the inverted map
    # from labels to instructions available for branch instructions.
    for i, instruction in enumerate(self.ir.ir):
      if instruction.label not in old_to_new_labels:
        old_to_new_labels[instruction.label] = i
        instruction.label = i
        self.processed_ir_instructions[i] = {
            'instruction': instruction,
            'start_byte_offset': 0
            }

    for instruction in self.ir.ir:
      if instruction.instruction == 'bra':
        target_node = instruction.assigned_operand1
        if target_node.instructions:
          instruction.assigned_operand1 = target_node.instructions[0].label
      if instruction.instruction in ['beq', 'bne', 'blt', 'ble', 'bgt', 'bge']:
        target_node = instruction.assigned_operand2
        if target_node.instructions:
          instruction.assigned_operand2 = \
              target_node.instructions[0].label

    # A temporary dictionary containing the nodes visited as the keys and
    # dummy True as the value. This dictionary must be reset for every
    # traversal.
    self.visited = {}
    self.live_intervals = {}
    self.analyze_basic_block_liveness(self.ir.cfg[0])

    for register, liveness in self.live_intervals.iteritems():
      # Note the register is live upto the instruction that is before its
      # last usage, since it will anyway be killed in this instruction, so
      # for the range, we don't add liveness[1] + 1
      for label in range(liveness[0], liveness[1]):
        if register.color in self.instruction_live_registers[label]:
#          raise RegisterAllocationFailedException(
#              'Register Allocation has failed, color %d assigned more than '
#              'once at the instruction %d point. The virtual registers '
#              'involved are %s and %s.' % (
#              register.color, label, register,
#              self.instruction_live_registers[label][register.color]
#              ))
          pass
        else:
          self.instruction_live_registers[label][register.color] = \
              register

  def is_register(self, register):
    """Checks if the register has been assigned a real machine register color.
    """
    return True if (register and isinstance(register, Register) and
        isinstance(register.color, int)) else False

  def add_instruction(self, label, instruction):
    """Adds the instruction to the instruction list and record the label.

    Args:
      label: The label of the original instruction in IR.
    """
    start_offset = self.instruction_offset
    self.instruction_offset += len(instruction.binary)
    self.instructions.append(instruction)
    self.ir_machine_map[label].append(instruction)
    self.instruction_offsets_map[instruction] = {
        'start_offset': start_offset,
        'end_offset': self.instruction_offset
        }

    destination = instruction.destination
    source = instruction.source

    if isinstance(destination, Memory) and destination.base == 'rip':
      self.globals_to_process.append((instruction, destination))
    elif (isinstance(destination, Memory) and isinstance(destination.base, Memory)
        and (destination.base.base == 'rip')):
      self.global_sibs_to_process.append((instruction, destination.base))
    elif (isinstance(destination, Register) and
        isinstance(destination.color, Memory) and
        destination.color.base == 'rip'):
      self.globals_to_process.append((instruction, destination.color))

    if isinstance(source, Memory) and source.base == 'rip':
      self.globals_to_process.append((instruction, source))
    elif (isinstance(source, Memory) and isinstance(source.base, Memory)
        and (source.base.base == 'rip')):
      self.global_sibs_to_process.append((instruction, source.base))
    elif (isinstance(source, Register) and
        isinstance(source.color, Memory) and
        source.color.base == 'rip'):
      self.globals_to_process.append((instruction, source.color))

  def handle_add(self, label, result, *operands):
    """Handles the add instruction of IR.
    """
    if self.is_register(operands[0]) and (result.color == operands[0].color):
      operand = operands[1]
    elif self.is_register(operands[1]) and (result.color == operands[1].color):
      operand = operands[0]
    else:
      mov = MOV(result, operands[0])
      self.add_instruction(label, mov)
      operand = operands[1]

    add = ADD(result, operand)
    self.add_instruction(label, add)

  def handle_adda(self, label, result, *operands):
    """Handles the adda instruction of IR.
    """
    # Loads the effective base address of the array to the resulting register.
    lea = LEA(result, operands[1])
    self.add_instruction(label, lea)

    # Add that base array address to the array element offset.
    add = ADD(result, operands[0])
    self.add_instruction(label, add)

  def handle_beq(self, label, result, *operands):
    """Handles the beq instruction of IR.
    """
    je = JE()
    self.add_instruction(label, je)
    self.targets_to_process.append((je, operands[1]))

  def handle_bge(self, label, result, *operands):
    """Handles the bge instrucion of IR.
    """
    jge = JGE()
    self.add_instruction(label, jge)
    self.targets_to_process.append((jge, operands[1]))

  def handle_bgt(self, label, result, *operands):
    """Handles the bgt instrucion of IR.
    """
    jg = JG()
    self.add_instruction(label, jg)
    self.targets_to_process.append((jg, operands[1]))

  def handle_ble(self, label, result, *operands):
    """Handles the ble instrucion of IR.
    """
    jle = JLE()
    self.add_instruction(label, jle)
    self.targets_to_process.append((jle, operands[1]))

  def handle_blt(self, label, result, *operands):
    """Handles the blt instrucion of IR.
    """
    jl = JL()
    self.add_instruction(label, jl)
    self.targets_to_process.append((jl, operands[1]))

  def handle_bne(self, label, result, *operands):
    """Handles the bne instruction of IR.
    """
    jne = JNE()
    self.add_instruction(label, jne)
    self.targets_to_process.append((jne, operands[1]))

  def handle_bra(self, label, result, *operands):
    """Handles the bra instrucion of IR.
    """
    jmp = JMP()
    self.add_instruction(label, jmp)
    self.targets_to_process.append((jmp, operands[0]))

  def handle_call(self, function_name, label, result, *operands):
    """Handles the call instruction of IR.
    """
    pushed_registers = []
    if 0 in self.instruction_live_registers[label]:
      self.instruction_live_registers[label].pop(0)
      if not self.is_register(result) or result.color != 0:
        # Create a dummy register
        register = Register()
        register.color = 0
        push = PUSH(register)
        self.add_instruction(label, push)
        pushed_registers.append(register)


    for register_color in self.instruction_live_registers[label]:
      # Create a dummy register
      register = Register()
      register.color = register_color
      push = PUSH(register)
      self.add_instruction(label, push)
      pushed_registers.append(register)

    # Arguments are passed through %rdi, %rsi, %rdx, %rcx, %r8, %r9
    for argument, register_color in zip(operands, FUNCTION_ARGUMENTS_COLORS):
      if not (isinstance(argument, Register) and
          argument.color == register_color):
        # Create a dummy register for %rax.
        new_register = Register()
        new_register.color = register_color
        mov = MOV(new_register, argument)
        self.add_instruction(label, mov)

    # If there are more than arguments to the function, push them on to the
    # stack in the right-to-left order
    if len(operands) > 6:
      # Note we run this upto 5 but not including 5 which is the 6th argument.
      for argument in operands[-1:5:-1]:
        push = PUSH(argument)
        self.add_instruction(label, push)

    call = CALL()
    self.add_instruction(label, call)
    self.calls_to_link.append((call, function_name))

    # Popping should be in the reverse order
    pop_rax_later = False
    for register in pushed_registers[-1::-1]:
      # Don't pop %rax yet, if result exists, we need to mov %rax to
      # actual result register and then pop
      if register.color == 0:
        pop_rax_later = True
        continue

      pop = POP(register)
      self.add_instruction(label, pop)

    # NOTE: The following move is done after popping all the values because
    # popping overwrites the values otherwise.
    # If the result of the call instruction is to be passed on to some other
    # register than %rax, we need to insert a move for it.
    if self.is_register(result) and result.color != 0:
      rax = Register()
      rax.color = 0

      mov = MOV(result, rax)
      self.add_instruction(label, mov)

      if pop_rax_later:
        register = Register()
        register.color = 0
        pop = POP(register)
        self.add_instruction(label, pop)

  def handle_cmp(self, label, result, *operands):
    """Handles the cmp instruction of IR.
    """
    if isinstance(operands[0], Memory) and isinstance(operands[1], Memory):
      mov = MOV(result, operands[0])
      self.add_instruction(label, mov)

      cmp_instruction = CMP(result, operands[1])
      self.add_instruction(label, cmp_instruction)
    else:
      cmp_instruction = CMP(operands[0], operands[1])
      self.add_instruction(label, cmp_instruction)

  def handle_div(self, label, result, *operands):
    """Handles the div instruction of IR.
    """
    restore_rax = True
    restore_rdx = True

    # This will be true only if the second operand is immediate.
    restore_r15 = False

    # Create dummy register objects for RAX and RDX and force color them to
    # ensure they take RAX and RDX values. So we can spill these two registers.
    # Also create memory object to where they get spilled.
    if self.is_register(result):
      if result.color == 0:
        restore_rax = False
      elif result.color == 3:
        restore_rdx = False

    # We may want %rax register to move the first dividend to %rax if it is
    # not already in %rax, so create a dummy %rax register in any case.
    rax = Register()
    rax.color = 0     # Color of RAX

    # We want to clear the contents of %rdx no matter what, so create a dummy
    # register for it.
    rdx = Register()
    rdx.color = 3     # Color of RDX

    # Store %rax and %rdx in memory if they have to be restored
    if restore_rax:
      rax.memory = Memory()

      rax_operands = (rax, rax.memory)
      self.handle_store(label, None, *rax_operands)

    if restore_rdx:
      rdx.memory = Memory()

      rdx_operands = (rdx, rdx.memory)
      self.handle_store(label, None, *rdx_operands)

    if not (self.is_register(operands[0]) and operands[0].color == 0):
      mov = MOV(rax, operands[0])
      self.add_instruction(label, mov)

    # Clear %rdx, because x86_64 DIV instruction uses both %rax and %rdx as
    # the divisor
    xor = XOR(rdx, rdx)
    self.add_instruction(label, xor)

    if isinstance(operands[1], Immediate):
      # Create a dummy register object for %r15 for moving the operand to it.
      r15 = Register()
      r15.color = 13

      if not(self.is_register(result) and result.color == 13):
        restore_r15 = True
        r15.memory = Memory()
        r15_operands = (r15, r15.memory)
        self.handle_store(label, None, *r15_operands)

      mov = MOV(r15, operands[1])
      self.add_instruction(label, mov)

      idiv = IDIV(r15)
    else:
      idiv = IDIV(operands[1])

    self.add_instruction(label, idiv)

    mov = MOV(result, rax)
    self.add_instruction(label, mov)

    if restore_rax:
      self.handle_load(label, rax, rax.memory)
    if restore_rdx:
      self.handle_load(label, rdx, rdx.memory)
    if restore_r15:
      self.handle_load(label, r15, r15.memory)

  def handle_move(self, label, result, *operands):
    """Handles the move instruction of IR.
    """
    mov = MOV(operands[1], operands[0])
    self.add_instruction(label, mov)

  def handle_mul(self, label, result, *operands):
    """Handles the mul instruction of IR.
    """
    if not isinstance(result, Register):
      raise NotImplementedError(
          'Only two registers or two immediates or one register and another '
          'immediate multiplications are supported at the moment.')

    if (isinstance(operands[0], Immediate) and
        isinstance(operands[1], Immediate)):
      mov = MOV(result, operands[0])
      self.add_instruction(label, mov)

      imul = IMUL(result, result, operands[1])
    elif isinstance(operands[1], Immediate):
      imul = IMUL(result, operands[0], operands[1])
      self.add_instruction(label, imul)
    elif isinstance(operands[0], Immediate):
      imul = IMUL(result, operands[1], operands[0])
      self.add_instruction(label, imul)
    else:
      if self.is_register(operands[0]) and (result.color == operands[0].color):
        operand = operands[1]
      elif self.is_register(operands[1]) and (result.color == operands[1].color):
        operand = operands[0]
      else:
        mov = MOV(result, operands[0])
        self.add_instruction(label, mov)
        operand = operands[1]

      imul = IMUL(result, operand)
      self.add_instruction(label, imul)

  def handle_ret(self, label, result, *operands):
    """Handles the read instruction of IR.
    """
    # Return is nothing but the jump to the end of the function.
    # If it has an operand it should be returned in %rax, according to Linux
    # Application Binary Interface (ABI).
    if operands:
      # 0 is the Register color for %rax.
      if operands[0]:
        # Create a dummy register for %rax.
        rax = Register()
        rax.color = 0
        mov = MOV(rax, operands[0])
        self.add_instruction(label, mov)

    jmp = JMP()
    self.add_instruction(label, jmp)
    self.returns_to_process.append(jmp)

  def handle_load(self, label, result, *operands):
    """Handles the load instruction of IR.
    """
    register = result

    if isinstance(operands[0], Immediate) and isinstance(operands[1], Memory):
      memory = Memory(
          base=operands[1].base,
          offset=operands[1].offset + (operands[0].value * MEMORY_WIDTH))
    elif isinstance(operands[0], Register):
      # This happens only in case of arrays
      memory = Memory(base=operands[1], offset=operands[0])
    elif isinstance(operands[0], Memory):
      memory = operands[0]
      if memory.offset == None:
        memory.offset = self.memory_offset
        self.memory_offset += MEMORY_WIDTH

    mov = MOV(register, memory)
    self.add_instruction(label, mov)

  def handle_store(self, label, result, *operands):
    """Handles the store instruction of IR.
    """
    if isinstance(operands[1], Immediate) and isinstance(operands[2], Memory):
      memory = Memory(
          base=operands[2].base,
          offset=operands[2].offset + (operands[1].value * MEMORY_WIDTH))
    elif isinstance(operands[1], Register):
      memory = Memory()
      memory.base = operands[2]
      memory.offset = operands[1]
    elif isinstance(operands[1], Memory):
      memory = operands[1]
      memory.offset = self.memory_offset

      self.memory_offset += MEMORY_WIDTH

    if isinstance(operands[0], Memory) or (isinstance(operands[0], Register)
        and isinstance(operands[0].color, Memory)):

      # Take the set difference
      free_registers = (REGISTERS_COLOR_SET -
          set(self.instruction_live_registers[label].keys()))
      if free_registers:
        # Create a dummy register for this color
        register = Register()
        # pop and add back since sets don't support indexing
        register.color = free_registers.pop()
        free_registers.add(register.color)

        mov1 = MOV(register, operands[0])
        self.add_instruction(label, mov1)

        mov2 = MOV(memory, register)
        self.add_instruction(label, mov2)
      else:
        # We don't have any free registers :(
        # We can't do much, randomly pick a register, push it to the stack,
        # do the memory move from the source to this register and then use
        # this register to move the source to the destination and then
        # pop back our register.
        # NOTE: We pick %rax for this temporary operation
        # Create a dummy register for %rax
        register = Register()
        register.color = 0
        push = PUSH(register)
        self.add_instruction(label, push)

        mov1 = MOV(register, operands[0])
        self.add_instruction(label, mov1)

        mov2 = MOV(memory, register)
        self.add_instruction(label, mov2)

        pop = POP(register)
        self.add_instruction(label, pop)
    else:
      register = operands[0]

      mov = MOV(memory, register)
      self.add_instruction(label, mov)

  def handle_write(self, label, result, *operands):
    """Handles the write instruction of IR.
    """
    pass

  def handle_wln(self, label, result, *operands):
    """Handles the wln instruction of IR.
    """
    pass

  def handle_sub(self, label, result, *operands):
    """Handles the sub instruction of IR.
    """
    if self.is_register(operands[0]) and (result.color == operands[0].color):
      operand = operands[1]
    elif self.is_register(operands[1]) and (result.color == operands[1].color):
      xchg = XCHG(operands[0], operands[1])
      self.add_instruction(label, xchg)

      operand = operands[0]
    else:
      mov = MOV(result, operands[0])
      self.add_instruction(label, mov)
      operand = operands[1]

    sub = SUB(result, operand)
    self.add_instruction(label, sub)

  def handle_xchg(self, label, result, *operands):
    """Handles the xchg instruction of IR.
    """
    xchg = XCHG(operands[0], operands[1])
    self.add_instruction(label, xchg)

  def create_stack(self):
    """Creates the stack with the current memory offset value.
    """
    # Create a dummy register and give it a color to keep the API consistent.
    rbp = Register()
    rbp.color = 'rbp'        # The color of %rbp
    push = PUSH(rbp)
    self.add_instruction(0, push)

    # Another dummy register for %rsp
    rsp = Register()
    rsp.color = 'rsp'
    mov = MOV(rbp, rsp)
    self.add_instruction(0, mov)

  def handle_prologue(self, func_name, *operands):
    """Handles the prologue of the function definition in IR.
    """
    self.memory_offset = 0

    # Allocate memory for both the defined variables and the formal parameters.
    # Symbol table will have entry for all of them.
    for symtab_entry in self.ir.local_symbol_table.values():
      if 'memory' in symtab_entry:
        symtab_entry['memory'].offset = self.memory_offset
        symtab_entry['memory'].size = MEMORY_WIDTH
        symtab_entry['memory'].base = 0

        self.memory_offset += MEMORY_WIDTH

      elif symtab_entry.get('dimensions'):
        if 'memory' not in symtab_entry:
          symtab_entry['memory'] = Memory()

        symtab_entry['memory'].offset = self.memory_offset
        total_size = 1
        for dimension in symtab_entry['dimensions']:
          total_size *= dimension

        symtab_entry['memory'].size = total_size * MEMORY_WIDTH
        symtab_entry['memory'].base = 0

        self.memory_offset += total_size * MEMORY_WIDTH

    self.create_stack()

  def handle_epilogue(self, func_name, *operands):
    """Handles the epilogue of the function definition in IR.
    """
    # Create a dummy register and give it a color to keep the API consistent.
    rbp = Register()
    rbp.color = 'rbp'        # The color of %rbp
    pop = POP(rbp)
    self.add_instruction(len(self.ir.ir) - 1, pop)

    ret = RET()
    self.add_instruction(len(self.ir.ir) - 1, ret)

  def link_jump_targets(self):
    """Link together all the jump targets
    """
    for instruction, target in self.targets_to_process:
       target_instruction = self.ir_machine_map[target][0]
       target_start_offset = self.instruction_offsets_map[
           target_instruction]['start_offset']
       next_offset = self.instruction_offsets_map[instruction]['end_offset']
       jump_offset = target_start_offset - next_offset
       instruction.set_target(jump_offset)

  def build(self):
    """Builds the binary for the function.
    """
    instruction_binaries = [instruction.binary
        for instruction in self.instructions]
    self.binary = ''.join(instruction_binaries)

    return self.binary

  def generate(self):
    """Bootstraps the code generation for the SSA object.
    """
    self.liveness()

    for instruction in self.ir.ir:
      mnemonic = instruction.instruction
      if mnemonic.startswith('.begin_'):
        self.handle_prologue(instruction.function_name, *instruction.operands)
      elif mnemonic.startswith('.end_'):
        self.handle_epilogue(instruction.function_name, *instruction.operands)
        break
      elif mnemonic == 'call':
        self.handle_call(instruction.function_name, instruction.label,
                         *instruction.operands)
      else:
        func = getattr(self, 'handle_%s' % mnemonic)
        func(instruction.label, instruction.assigned_result,
             instruction.assigned_operand1, instruction.assigned_operand2,
             *instruction.assigned_operands)

    self.link_jump_targets()
    self.build()



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

  cg = CodeGenerator(ssa)
  cg.generate()

  return cg


if __name__ == '__main__':
  cg = bootstrap()
