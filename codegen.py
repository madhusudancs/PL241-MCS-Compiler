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
import subprocess
import sys

from argparse import ArgumentParser

from ir import Immediate
from ir import IntermediateRepresentation
from ir import Memory
from optimizations import Optimize
from parser import LanguageSyntaxError
from parser import Parser
from regalloc import Register
from regalloc import RegisterAllocator
from ssa import SSA

# Architecture specific imports
from x86_64 import ADD
from x86_64 import CMP
from x86_64 import IDIV
from x86_64 import IMUL
from x86_64 import JE
from x86_64 import JG
from x86_64 import JGE
from x86_64 import JL
from x86_64 import JLE
from x86_64 import JMP
from x86_64 import JNE
from x86_64 import MEMORY_WIDTH
from x86_64 import MOV
from x86_64 import SUB
from x86_64 import Instruction


# Module level logger object
LOGGER = logging.getLogger(__name__)



class CodeGenerator(object):
  """Generates the code for the given SSA object.
  """

  def __init__(self, ssa):
    """Constructs the datastructures required for code generation.

    Args:
      ssa: The SSA object which contains the registers allocated instructions.
    """
    self.ssa = ssa

    # Contains the set of instructions generated.
    self.instructions = []

    # A dictionary containing a mapping from the IR instructions' labels
    # the actual instruction if they are processed.
    self.processed_ir_instructions = {}

    # A two-tuple containing the instruction and its original target operand
    # for which the target code is yet to be processed.
    self.targets_to_process = []

    # Memory offset of the current scope.
    self.memory_offset = None

  def handle_add(self, result, *operands):
    """Handles the add instruction of IR.
    """
    mov = MOV(result, operands[0])
    self.instructions.append(mov)

    add = ADD(result, operands[1])
    self.instructions.append(add)

  def handle_adda(self, result, *operands):
    """Handles the adda instruction of IR.
    """
    pass

  def handle_beq(self, result, *operands):
    """Handles the beq instruction of IR.
    """
    je = JE(result, operands[1])
    self.instructions.append(je)
    self.targets_to_process.append((je, operands[1]))

  def handle_bge(self, result, *operands):
    """Handles the bge instrucion of IR.
    """
    jge = JGE()
    self.instructions.append(jge)
    self.targets_to_process.append((jge, operands[1]))

  def handle_bgt(self, result, *operands):
    """Handles the bgt instrucion of IR.
    """
    jg = JG()
    self.instructions.append(jg)
    self.targets_to_process.append((jg, operands[1]))

  def handle_ble(self, result, *operands):
    """Handles the ble instrucion of IR.
    """
    jle = JLE()
    self.instructions.append(jle)
    self.targets_to_process.append((jle, operands[1]))

  def handle_blt(self, result, *operands):
    """Handles the blt instrucion of IR.
    """
    jl = JL()
    self.instructions.append(jl)
    self.targets_to_process.append((jl, operands[1]))

  def handle_bne(self, result, *operands):
    """Handles the bne instruction of IR.
    """
    jne = JNE()
    self.instructions.append(jne)
    self.targets_to_process.append((jne, operands[1]))

  def handle_bra(self, result, *operands):
    """Handles the bra instrucion of IR.
    """
    jmp = JMP()
    self.instructions.append(jmp)
    self.targets_to_process.append((jmp, operands[0]))

  def handle_call(self, *operands):
    """Handles the call instruction of IR.
    """
    pass

  def handle_cmp(self, result, *operands):
    """Handles the cmp instruction of IR.
    """
    cmp_instruction = CMP(operands[0], operands[1])
    self.instructions.append(cmp_instruction)

  def handle_div(self, result, *operands):
    """Handles the div instruction of IR.
    """
    # Create dummy register objects for RAX and RDX and force color them to
    # ensure they take RAX and RDX values. So we can spill these two registers.
    rax = Register()
    rax.color = 0     # Color of RAX
    rdx = Register()
    rdx.color = 2     # Color of RDX

    # Also create memory object to where they get spilled.
    memory_rax = Memory()
    memory_rbx = Memory()

    operands = (rax, memory_rax)
    store_rax = self.store(result=None, *operands)
    self.instructions.append(store_rax)
    operands = (rax, memory_rdx)
    store_rdx = self.store(result=None, *operands)
    self.instructions.append(store_rdx)

    mov = MOV(rax, operands[0])
    self.instructions.append(mov)

    idiv = IDIV(operands[1])
    self.instructions.append(idiv)

    mov_result = MOV(result, rax)
    self.instructions.append(mov_result)

    load_rax = self.load(rax, memory_rax)
    self.instructions.append(load_rax)
    load_rdx = self.load(rdx, memory_rdx)
    self.instructions.append(load_rdx)

  def handle_load(self, result, *operands):
    """Handles the load instruction of IR.
    """
    register = result

    if isinstance(operands[0], Register):
      memory = Memory()
      memory.offset = operands[0]
    elif isinstance(operands[0], Memory):
      memory = operands[0]
      self.memory_offset -= memory.offset

    mov = MOV(register, memory)
    self.instructions.append(mov)

  def handle_move(self, result, *operands):
    """Handles the move instruction of IR.
    """
    mov = MOV(operands[1], operands[0])
    self.instructions.append(mov)

  def handle_mul(self, result, *operands):
    """Handles the mul instruction of IR.
    """
    if ((isinstance(operands[0], Register) or
        isinstance(operands[0], Immediate)) and
        isinstance(operands[1], Register)):
      mov = MOV(result, operands[0])
      self.instructions.append(mov)

      imul = IMUL(result, operands[1])
      self.instructions.append(imul)
    elif (isinstance(operands[0], Register) and
        isinstance(operands[1], Immediate)):
      mov = MOV(result, operands[1])
      self.instructions.append(mov)

      imul = IMUL(result, operands[0])
      self.instructions.append(imul)
    elif (isinstance(operands[0], Immediate) and
        isinstance(operands[1], Immediate)):
      mov = MOV(result, operands[0])
      self.instructions.append(mov)

      imul = IMUL(result, result, operands[0])
      self.instructions.append(imul)
    else:
      raise NotImplementedError(
          'Only two registers or two immediates or one register and another '
          'immediate multiplications are supported at the moment.')

  def handle_read(self, result, *operands):
    """Handles the read instruction of IR.
    """
    pass

  def handle_store(self, result, *operands):
    """Handles the store instruction of IR.
    """
    register = operands[0]
    if isinstance(operands[1], Register):
      memory = Memory()
      memory.offset = operands[1]
    elif isinstance(operands[1], Memory):
      memory = operands[1]
      memory.offset = self.memory_offset

      self.memory_offset += MEMORY_WIDTH

    mov = MOV(memory, register)
    self.instructions.append(mov)

  def handle_write(self, result, *operands):
    """Handles the write instruction of IR.
    """
    pass

  def handle_wln(self, result, *operands):
    """Handles the wln instruction of IR.
    """
    pass

  def handle_sub(self, result, *operands):
    """Handles the sub instruction of IR.
    """
    mov = MOV(result, operands[0])
    self.instructions.append(mov)

    sub = SUB(result, operands[1])
    self.instructions.append(sub)

  def handle_prologue(self, func_name, *operands):
    """Handles the prologue of the function definition in IR.
    """
    self.memory_offset = 0

  def handle_epilogue(self, func_name, *operands):
    """Handles the epilogue of the function definition in IR.
    """
    pass

  def generate(self):
    """Bootstraps the code generation for the SSA object.
    """
    for instruction in self.ssa.ssa:
      mnemonic = instruction.instruction
      if mnemonic.startswith('.begin_'):
        self.handle_prologue(mnemonic[7:], instruction.operands)
      elif mnemonic.startswith('.end_'):
        self.handle_epilogue(mnemonic[5:], instruction.operands)
      elif mnemonic == 'call':
        self.handle_call(instruction.operands)
      else:
        func = getattr(self, 'handle_%s' % mnemonic)
        func(instruction.assigned_result, instruction.assigned_operand1,
             instruction.assigned_operand2, *instruction.assigned_operands)
        self.processed_ir_instructions[instruction.label] = instruction


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
