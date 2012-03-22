# Copyright 2012 Madhusudan C.S.
#
# This file x86_64.py is part of PL241-MCS compiler.
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

"""This module contains the x86_64 specific classes and definitions.
"""


import logging
import struct

from argparse import ArgumentParser

from ir import Immediate
from ir import Memory
from regalloc import Register


# Module level logger object
LOGGER = logging.getLogger(__name__)


# The following two constants holds for everything within x86_64 architecture.
# This won't change within the architecture. And across architectures when
# one changes the other should also change. The first is the name of the
# byte ordering, second is the format required by Python's struct module to
# define byte ordering.
ENDIANNESS = 'little'
BYTE_ORDERING_FMT = '<'


# All the objects in memory are 8-bytes (64 bits) wide.
MEMORY_WIDTH = 0x8

# From the register allocator, we have the registers as the numbers from 0
# to N - 1 where N is the number of registers allowed for allocation. This
# map translates those colors into the x86_64 register codes. The names of
# the registers are given as comments next to each mapping.

# IMPORTANT: The REX value here is the binary bit that should be set or not.
# Since the position in the REX byte changes depending on whether the register
# is source or destination, it will be encoded to right place in REX byte
# while building the instruction.
REGISTER_COLOR_TO_CODE_MAP = {
    0:      { 'REX': 0b0, 'REG': 0b000 },   # rax
    1:      { 'REX': 0b0, 'REG': 0b011 },   # rbx
    2:      { 'REX': 0b0, 'REG': 0b001 },   # rcx
    3:      { 'REX': 0b0, 'REG': 0b010 },   # rdx
    4:      { 'REX': 0b0, 'REG': 0b110 },   # rsi
    5:      { 'REX': 0b0, 'REG': 0b111 },   # rdi
    6:      { 'REX': 0b1, 'REG': 0b000 },   # r8
    7:      { 'REX': 0b1, 'REG': 0b001 },   # r9
    8:      { 'REX': 0b1, 'REG': 0b010 },   # r10
    9:      { 'REX': 0b1, 'REG': 0b011 },   # r11
    10:     { 'REX': 0b1, 'REG': 0b100 },   # r12
    11:     { 'REX': 0b1, 'REG': 0b101 },   # r13
    12:     { 'REX': 0b1, 'REG': 0b110 },   # r14
    13:     { 'REX': 0b1, 'REG': 0b111 },   # r15
    'rsp':  { 'REX': 0b0, 'REG': 0b100 },   # rsp
    'rbp':  { 'REX': 0b0, 'REG': 0b101 }    # rbp
    }


# Keep this pre-computed for faster set operations on register colors, like
# set difference operation etc. which will be used in code generator.
REGISTERS_COLOR_SET = set(range(0, 14))


class InvalidInstructionException(Exception):
  """Represents the exception when the instruction is invalid.
  """

  def __init__(self, msg, *args, **kwargs):
    """Constructs the exception with the message.

    Args:
      msg: The exception message (this is optional).
    """
    super(InvalidInstructionException, self).__init__(*args, **kwargs)
    self.msg = msg if msg else ''

  def __str__(self):
    return '%s: %s' % (self.__class__.__name__, self._msg)


class Instruction(object):
  """Abstracts the instructions in x86_64 architecture.
  """

  def __init__(self, destination=None, source=None, offset=None):
    """Constructs the instruction object for x86_64.

    Args:
      mnemonic: The mnemonic of this instruction.
      source: The source operand of this instruction.
      destination: The destination operand of this instruction.
      offset: Offset from the starting of the program for this instruction.
    """
    self.source = source
    self.destination = destination

    self.offset = offset

    self.rex = None
    self.opcode = None
    self.modregrm = None
    self.displacement = None
    self.immediate = None

    # The actual binary that is built for this instruction.
    self.binary = None

    self.build()

  def mod_reg_rm_byte(self, mod, reg, rm):
    """Returns the byte containing the mod, reg and rm values.

    Args:
      mod: The 2-bits mod value.
      reg: The 3-bits reg value.
      rm: The 3-bits rm value.
    """
    return ((mod << 6) | (reg << 3) | ( rm ))

  def rex_byte(self, base=0b0, R=0b0, X=0b0, B=0b0):
    """Returns the byte containing the mod, reg and rm values.

    Args:
      base: The base rex byte defined with the instruction opcode in the
          Intel spec sheet.
      R: The single bit R value.
      X: The single bit X value.
      B: The single bit B value.
    """
    return ((base) | (R << 2) | (X << 1) | (B))

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''
    if (isinstance(self.destination, Register) and
        isinstance(self.source, Register)):
      source_reg = REGISTER_COLOR_TO_CODE_MAP[self.source.color]
      dest_reg = REGISTER_COLOR_TO_CODE_MAP[self.destination.color]

      opcode_entry = self.OPCODE_TABLE[('reg64', 'rm64')]

      mod = 0b11
      reg = dest_reg['REG']
      rm = source_reg['REG']
      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=dest_reg['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)

    elif (isinstance(self.destination, Register) and
        isinstance(self.source, Memory)):
      source_mem = self.source
      dest_reg = REGISTER_COLOR_TO_CODE_MAP[self.destination.color]

      opcode_entry = self.OPCODE_TABLE[('reg64', 'rm64')]
      mod = 0b10
      reg = dest_reg['REG']
      if isinstance(source_mem.offset, Register):
        source_reg = REGISTER_COLOR_TO_CODE_MAP[source_mem.offset.color]
        rm = source_reg['REG']
        offset = 0
      else:
        source_reg = REGISTER_COLOR_TO_CODE_MAP['rbp']
        rm = source_reg['REG']
        offset = -source_mem.offset if source_mem.offset else \
            (source_mem.offset if source_mem.offset else 0) * MEMORY_WIDTH

      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=dest_reg['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)
      self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, offset)

    elif (isinstance(self.destination, Memory) and
        isinstance(self.source, Register)):
      source_reg = REGISTER_COLOR_TO_CODE_MAP[self.source.color]
      dest_mem = self.destination

      opcode_entry = self.OPCODE_TABLE[('rm64', 'reg64')]
      mod = 0b10
      reg = source_reg['REG']

      if isinstance(dest_mem.offset, Register):
        dest_reg = REGISTER_COLOR_TO_CODE_MAP[dest_mem.offset.color]
        rm = dest_reg['REG']
        offset = 0
      else:
        dest_reg = REGISTER_COLOR_TO_CODE_MAP['rbp']
        rm = dest_reg['REG']
        offset = -dest_mem.offset if dest_mem.offset != None else 0

      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=source_reg['REX'],
                          B=dest_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)
      self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, offset)


    # This case is very important since the destination register is encoded
    # into opcode. See the binary generation for OPCODE
    elif (isinstance(self.destination, Register) and
        isinstance(self.source, Immediate)):
      source_imm = self.source.value
      dest_reg = REGISTER_COLOR_TO_CODE_MAP[self.destination.color]

      opcode_entry = self.OPCODE_TABLE[('rm64', 'imm32')]

      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=dest_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack(
          '%sB' % BYTE_ORDERING_FMT,
          opcode_entry['OPCODE'] + dest_reg['REG'])

      self.binary += struct.pack('%sq' % BYTE_ORDERING_FMT, source_imm)

    elif (isinstance(self.destination, Memory) and
        isinstance(self.source, Immediate)):
      source_imm = self.source.value
      dest_mem = self.destination

      opcode_entry = self.OPCODE_TABLE[('rm64', 'imm32')]

      mod = 0b01

      if isinstance(dest_mem.offset, Register):
        dest_reg = REGISTER_COLOR_TO_CODE_MAP[dest_mem.offset.color]
        rm = dest_reg['REG']
        offset = 0
      else:
        dest_reg = REGISTER_COLOR_TO_CODE_MAP['rbp']
        rm = dest_reg['REG']
        offset = -dest_mem.offset if dest_mem.offset != None else 0

      modregrm = self.mod_reg_rm_byte(mod, 000, rm)

      rex = self.rex_byte(base=opcode_entry['REX'],
                          B=dest_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)


      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])

      self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, offset)

      self.binary += struct.pack('%sq' % BYTE_ORDERING_FMT, source_imm)
    else:
      raise NotImplementedError('The operands for the instruction could not '
          'be encoded. Destination: %s Source: %s' % (
              self.destination, self.source))

  def __len__(self):
    """Returns the length of the binary bytes for the instruction.
    """
    return len(self.binary)

  def __str__(self):
    """Returns the binary bytes for the instruction.
    """
    return self.binary


class JumpInstruction(Instruction):
  """Abstracts the jump instructions (this includes call also).
  """

  def __init__(self, target=None):
    """Constructs the JE instruction.
    """
    self.target = target
    super(JumpInstruction,self).__init__()

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if not self.target:
      self.target = 0x0

    if not isinstance(self.target, int):
      raise InvalidInstructionException(
          'Target of branch instruction is not integer. The value given is '
          '%s.' % self.target)

    self.binary += struct.pack('%sH' % BYTE_ORDERING_FMT,
                               self.OPCODE_TABLE['rel32']['OPCODE'])

    self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, self.target)


class ADD(Instruction):
  """Implements the ADD instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'): { 'REX': 0x48, 'OPCODE': 0x03 },
      ('rm64', 'reg64'): { 'REX': 0x48, 'OPCODE': 0x01 },
      ('rm64', 'imm32'): { 'REX': 0x48, 'OPCODE': 0x81 }
      }

  def __init__(self, destination, source):
    """Constructs the ADD instruction.
    """
    super(ADD, self).__init__(destination, source)


class CALL(JumpInstruction):
  """Implements the CALL instruction.
  """

  OPCODE_TABLE = {
      'rel32': { 'REX': 0x0, 'OPCODE': 0xE8 }
      }

  def __init__(self, target=None):
    """Constructs the JMP instruction.
    """
    super(JMP,self).__init__()

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if not self.target:
      self.target = 0x0

    if not isinstance(self.target, int):
      raise InvalidInstructionException(
          'Target of branch instruction is not integer. The value given is '
          '%s.' % self.target)

    self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                               self.OPCODE_TABLE['rel32']['OPCODE'])

    self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, self.target)


class CMP(Instruction):
  """Implements the CMP instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'): { 'REX': 0x48, 'OPCODE': 0x3B },
      ('rm64', 'reg64'): { 'REX': 0x48, 'OPCODE': 0x39 },
      ('rm64', 'imm32'): { 'REX': 0x48, 'OPCODE': 0x81 }
      }

  def __init__(self, destination, source):
    """Constructs the CMP instruction.
    """
    super(CMP, self).__init__(destination, source)


class IDIV(Instruction):
  """Implements the IDIV instruction.

  NOTES: RDX and RAX must be cleared for division. RAX holds the quotient
  and RDX will hold the Remainder. So we will have to check if the existing
  registers for the result are RAX and RDX if not we will have to spill and
  immediately reload after the DIV instruction.
  """

  OPCODE_TABLE = {
      ('rm64'): { 'REX': 0x48, 'OPCODE': 0xF7 },
      }

  def __init__(self, source):
    """Constructs the IDIV instruction.
    """
    super(IDIV,self).__init__(destination=None, source=source)

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if isinstance(self.source, Register):
      source_reg = REGISTER_COLOR_TO_CODE_MAP[self.source.color]

      opcode_entry = self.OPCODE_TABLE['rm64']

      # FIXME: May be buggy because reg may be 0xb111 instead of 0
      mod = 0b11
      reg = 0b0
      rm = src_reg['REG']
      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)
    elif isinstance(self.source, Memory):
      source_mem = self.source

      opcode_entry = self.OPCODE_TABLE['rm64']

      # FIXME: May be buggy because reg may be 0xb111 instead of 0
      mod = 0b10
      reg = 0b0
      if isinstance(source_mem.offset, Register):
        source_reg = REGISTER_COLOR_TO_CODE_MAP[source_mem.offset.color]
        rm = source_reg['REG']
        offset = 0
      else:
        source_reg = REGISTER_COLOR_TO_CODE_MAP['rbp']
        rm = source_reg['REG']
        offset = -source_mem.offset

      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)
      self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, offset)

    else:
      raise NotImplementedError('Signed integer division is not implemented '
                                'on non register source operands.')


class IMUL(Instruction):
  """Implements the IMUL instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'):          { 'REX': 0x48, 'OPCODE': 0x0FAF },
      ('reg64', 'rm64', 'imm32'): { 'REX': 0x48, 'OPCODE': 0x69 },
      }

  def __init__(self, destination, source, immediate=None):
    """Constructs the IMUL instruction.
    """
    self.immediate = immediate
    super(IMUL,self).__init__(destination, source)

  def build(self):
    """Builds the instruction bytes.

    IMPORTANT: The only difference between this method and the Instruction
    class's build method is that the instruction opcode is encoded into
    2 bytes not a single byte.
    """
    self.binary = ''
    if self.immediate != None:
      source_reg = REGISTER_COLOR_TO_CODE_MAP[self.source.color]
      dest_reg = REGISTER_COLOR_TO_CODE_MAP[self.destination.color]
      imm32 = self.immediate.value

      opcode_entry = self.OPCODE_TABLE[('reg64', 'rm64', 'imm32')]

      mod = 0b11
      reg = dest_reg['REG']
      rm = src_reg['REG']
      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=dest_reg['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                 modregrm)
      self.binary += struct.pack('%si' % BYTE_ORDERING_FMT,
                                 imm32)
    elif (isinstance(self.destination, Register) and
        isinstance(self.source, Register)):
      source_reg = REGISTER_COLOR_TO_CODE_MAP[self.source.color]
      dest_reg = REGISTER_COLOR_TO_CODE_MAP[self.destination.color]

      opcode_entry = self.OPCODE_TABLE[('reg64', 'rm64')]

      mod = 0b11
      reg = dest_reg['REG']
      rm = source_reg['REG']
      modregrm = self.mod_reg_rm_byte(mod, reg, rm)
      rex = self.rex_byte(base=opcode_entry['REX'],
                          R=dest_reg['REX'],
                          B=source_reg['REX'])

      if rex:
        self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, rex)

      self.binary += struct.pack('%sH' % BYTE_ORDERING_FMT,
                                opcode_entry['OPCODE'])
      self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                                modregrm)
    else:
      raise NotImplementedError('Only register to register multiplications '
                                'are implemented.')


class JE(JumpInstruction):
  """Implements the JE instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x74 },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F84 }
      }

  def __init__(self, target=None):
    """Constructs the JE instruction.
    """
    super(JE,self).__init__()


class JG(JumpInstruction):
  """Implements the JG instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x7F },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F8F }
      }

  def __init__(self, target=None):
    """Constructs the JG instruction.
    """
    super(JG,self).__init__()


class JGE(JumpInstruction):
  """Implements the JGE instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x7D },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F8D }
      }

  def __init__(self, target=None):
    """Constructs the JGE instruction.
    """
    super(JGE,self).__init__()


class JL(JumpInstruction):
  """Implements the JL instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x7C },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F8C }
      }

  def __init__(self, target=None):
    """Constructs the  instruction.
    """
    super(JL,self).__init__()


class JLE(JumpInstruction):
  """Implements the JLE instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x7E },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F8E }
      }

  def __init__(self, target=None):
    """Constructs the JLE instruction.
    """
    super(JLE,self).__init__()


class JMP(JumpInstruction):
  """Implements the JMP instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0xEB },
      'rel32': { 'REX': 0x0, 'OPCODE': 0xE9 }
      }

  def __init__(self, target=None):
    """Constructs the JMP instruction.
    """
    super(JMP,self).__init__()

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if not self.target:
      self.target = 0x0

    if not isinstance(self.target, int):
      raise InvalidInstructionException(
          'Target of branch instruction is not integer. The value given is '
          '%s.' % self.target)

    self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT,
                               self.OPCODE_TABLE['rel32']['OPCODE'])

    self.binary += struct.pack('%si' % BYTE_ORDERING_FMT, self.target)


class JNE(JumpInstruction):
  """Implements the JNE instruction.
  """

  OPCODE_TABLE = {
      'rel8': { 'REX': 0x0, 'OPCODE': 0x75 },
      'rel32': { 'REX': 0x0, 'OPCODE': 0x0F85 }
      }

  def __init__(self, target=None):
    """Constructs the JNE instruction.
    """
    super(JNE,self).__init__()


class MOV(Instruction):
  """Implements the MOV instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'): { 'REX': 0x48, 'OPCODE': 0x8B },
      ('rm64', 'reg64'): { 'REX': 0x48, 'OPCODE': 0x89 },
      ('rm64', 'imm32'): { 'REX': 0x48, 'OPCODE': 0xC7 } # For initializations
      }

  def __init__(self, destination, source):
    """Constructs the MOV instruction.
    """
    super(MOV, self).__init__(destination, source)


class POP(Instruction):
  """Implements the POP instruction.
  """

  OPCODE_TABLE = {
      'reg64': { 'REX': 0x00, 'OPCODE': 0x58 },
      }

  def __init__(self, operand):
    """Constructs the POP instruction.
    """
    self.operand = operand
    super(POP, self).__init__()

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if not isinstance(self.operand, Register):
      raise InvalidInstructionException(
          'Operand of pop instruction is not register. The value given is '
          '%s.' % self.operand)

    opcode = self.OPCODE_TABLE['reg64']['OPCODE']
    opcode = opcode | REGISTER_COLOR_TO_CODE_MAP[self.operand.color]['REG']
    self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, opcode)


class PUSH(Instruction):
  """Implements the PUSH instruction.
  """

  OPCODE_TABLE = {
      'reg64': { 'REX': 0x00, 'OPCODE': 0x50 },
      }

  def __init__(self, operand):
    """Constructs the PUSH instruction.
    """
    self.operand = operand
    super(PUSH, self).__init__()

  def build(self):
    """Builds the instruction bytes.
    """
    self.binary = ''

    if not isinstance(self.operand, Register):
      raise InvalidInstructionException(
          'Operand of push instruction is not register. The value given is '
          '%s.' % self.operand)

    opcode = self.OPCODE_TABLE['reg64']['OPCODE']
    opcode = opcode | REGISTER_COLOR_TO_CODE_MAP[self.operand.color]['REG']
    self.binary += struct.pack('%sB' % BYTE_ORDERING_FMT, opcode)


class RET(Instruction):
  """Implements the RET instruction.
  """

  OPCODE_TABLE = {
      ('noperand'): { 'REX': 0x00, 'OPCODE': 0xCB },
      }

  def __init__(self):
    """Constructs the RET instruction.
    """
    super(RET, self).__init__(destination=None, source=None)


class SUB(Instruction):
  """Implements the SUB instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'): { 'REX': 0x48, 'OPCODE': 0x2B },
      ('rm64', 'reg64'): { 'REX': 0x48, 'OPCODE': 0x29 },
      ('rm64', 'imm32'): { 'REX': 0x48, 'OPCODE': 0x81 }
      }

  def __init__(self, destination, source):
    """Constructs the SUB instruction.
    """
    super(SUB, self).__init__(destination, source)


class XCHG(Instruction):
  """Implements the XCHG instruction.
  """

  OPCODE_TABLE = {
      ('reg64', 'rm64'): { 'REX': 0x48, 'OPCODE': 0x87 },
      ('rm64', 'reg64'): { 'REX': 0x48, 'OPCODE': 0x87 },
      }

  def __init__(self, destination, source):
    """Constructs the XCHG instruction.
    """
    super(XCHG, self).__init__(destination, source)


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

  
  elf = ELF(filename)
  elf.build()

  return elf


if __name__ == '__main__':
  elf = bootstrap()
