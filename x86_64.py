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


# Module level logger object
LOGGER = logging.getLogger(__name__)


# The following two constants holds for everything within x86_64 architecture.
# This won't change within the architecture. And across architectures when
# one changes the other should also change. The first is the name of the
# byte ordering, second is the format required by Python's struct module to
# define byte ordering.
ENDIANNESS = 'little'
BYTE_ORDERING_FMT = '<'


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

  def __init__(self):
    """Constructs the instruction object for x86_64.
    """
    self.rex = None
    self.opcode = None
    self.modregrm = None
    self.displacement = None
    self.immediate = None

  def __str__(self):
    """Returns the binary string for the instruction.
    """
    binary = ''
    if self.rex:
      binary += struct.pack('%sc' % (BYTE_ORDERING_FMT), self.rex)

    if self.rex:
      binary += struct.pack('%sc' % (BYTE_ORDERING_FMT), self.rex)

    if not self.opcode:
      raise InvalidInstructionException('No opcode defined.')

    binary += struct.pack('%sc' % (BYTE_ORDERING_FMT), self.rex)

    if not self.modrm:
      raise InvalidInstructionException('No modregrm defined.')

    binary += struct.pack('%sc' % (BYTE_ORDERING_FMT), self.modregrm)

    # Note this is an "unsigned" 64-bit (8 byte) displacement.
    if self.displacement:
      binary += struct.pack('%sq' % (BYTE_ORDERING_FMT), self.displacement)

    # Note this is an "unsigned" 64-bit (8 byte) immediate operand.
    if self.immediate:
      binary += struct.pack('%sq' % (BYTE_ORDERING_FMT), self.immediate)



class ADD(Instruction):
  """Implements the ADD instruction.
  """

  def __init__(self, source, destination):
    """Constructs the ADD instruction.
    """
    super(ADD, self).__init__()


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
