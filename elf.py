# Copyright 2012 Madhusudan C.S.
#
# This file elf.py is part of PL241-MCS compiler.
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
# IMPORTANT! The structure of this module is inspired by the elf package in
# the tool called pydevtools available for download from:
# http://code.google.com/p/pydevtools/ However this code was written completely
# independently by the authors of PL241-MCS compiler project and hence the
# copyright notice of pydevtools is not included here since this work is
# produced in entirety by the authors of PL241-MCS compiler. However the
# authors would like to thank the authors of pydevtools for the ideas.

"""This file contains all the definitions that ELF specifies.

This is basically just the translation of ELF spec sheet to Python
datastructures.

Few important notes. All the enumerations are defined as classes with the
enumerated names and values as the class attributes of the enumeration class.
The classes have been explicitly defined for each enumeration, even though
it is a tedious work because to keep up with the Python's philosophy of
"Explicit is better than implicit". The enumerated classes follow the naming
convention of all letters in uppercase so that they are easily distinguishable.
"""


import logging
import struct

from argparse import ArgumentParser


# Module level logger object
LOGGER = logging.getLogger(__name__)


class ELFMetaclass(type):
  """The metaclass abstracting all the datastructures required across ELF.
  """

  def __new__(cls, name, bases, scope_dict):
    """Constructor for the new ELF class instances.

    Args:
      Defined by Python protocol for metaclass and __new__ method definitions.
    """
    return super(ELFMetaclass, cls).__new__(cls, name, bases, scope_dict)

  def elf64_addr(cls, data):
    """Returns the packed binary whose size is addr as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_byte(cls, data):
    """Returns the packed binary whose size is byte as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sc' % cls.byte_ordering_fmt
    return struct.pack(format_str, chr(data))

  def elf64_half(cls, data):
    """Returns the packed binary whose size is half as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sH' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_off(cls, data):
    """Returns the packed binary whose size is off as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_word(cls, data):
    """Returns the packed binary whose size is word as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sI' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_xword(cls, data):
    """Returns the packed binary whose size is word as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)


# x86_64 architecture doesn't define any machine-specific flags, so it is
# set to 0 for now. Later this can be changed to enumerations based on
# the machine architecture.
ELF_FLAGS          = 0x0


ELF_EHSIZE         = 0xFF      # Holds the ELF's header size in bytes.
                               # Hard-coded to 64 bytes for now.

ELF_PHENTSIZE      = 0x0       # Holds the size in bytes of one entry in the
                               # file's program header table. All entries must
                               # be of the same size. Should be calculated
                               # dynamically.

ELF_PHNUM          = 0x0       # Holds the number of entries in the program
                               # header table. If the file has no program
                               # header table, this holds the value zero.
                               # Should be calculated dymamically.

ELF_SHENTSIZE      = 0x0       # This member holds a section header's size in
                               # bytes. A section header is one entry in the
                               # the section header table. All entries are the
                               # same size. Should be calculated dymamically.

ELF_SHNUM          = 0x0       # Holds the number of entries in the section
                               # header table. If the file has no section
                               # header table, this holds the value zero.
                               # Should be calculated dymamically.

ELF_SHSTRNDX       = 0x0       # This member holds the section header table
                               # index of the entry associated with the section
                               # name string table. If the file has no section
                               # name string table, this member holds the value
                               # SHN_UNDEF. For now it is hard coded to 0.
                               # But can be calculated dynamically if needed.


class ELF(object):
  """Builds the ELF binary file for the given inputs.
  """


  def __init__(self, filename, elf_class=64, endianness='little',
               architecture='x86_64'):
    """Constructs the ELF object required to generate ELF binaries.
    """
    self.filename = filename
    self.filepointer = open(filename, 'wb')

    if elf_class == 64:
      self.elf_class = ELFCLASS.ELFCLASS64
    elif elf_class == 32:
      self.elf_class = ELFCLASS.ELFCLASS32
    else:
      raise TypeError('Invalid elf type "%s".' % elf_class)

    if endianness == 'little':
      self.byte_ordering_fmt = '<'
      self.elf_data = ELFDATA.ELFDATA2LSB
    elif endianness == 'big':
      self.byte_ordering_fmt = '>'
      self.elf_data = ELFDATA.ELFDATA2MSB
    else:
      raise TypeError('Invalid byte-order type "%s".' % endianness)

    if architecture == 'x86':
      self.elf_machine = ELFMACHINE.EM_386
    elif architecture == 'x86_64':
      self.elf_machine = ELFMACHINE.EM_X86_64
    else:
      raise TypeError('Architecture %s not supported.' % architecture)

  def __del__(self):
    """Ensure that the write file is closed.
    """
    if not self.filepointer.closed:
      self.filepointer.close()

  def build(self):
    """Builds the binary file for the given input.
    """
    elf_header = self.build_elf_header()
    self.filepointer.write(elf_header)

  def build_elf_header(self):
    """Builds the ELF header.
    """
    magic = ''.join([
        self.elf64_byte(ELF_IDENT_MAG0),
        self.elf64_byte(ELF_IDENT_MAG1),
        self.elf64_byte(ELF_IDENT_MAG2),
        self.elf64_byte(ELF_IDENT_MAG3),
        self.elf64_byte(ELF_IDENT_CLASS),
        self.elf64_byte(ELF_IDENT_DATA),
        self.elf64_byte(ELF_IDENT_VERSION),
        self.elf64_byte(0x00) * ELF_IDENT_PAD
        ])

    rest_of_elf_header = ''.join([
        self.elf64_half(ELFTYPE.ET_EXEC),
        self.elf64_half(self.elf_machine),
        self.elf64_word(ELFVERSION.EV_CURRENT),
        self.elf64_addr(ELF_ENTRY),
        self.elf64_off(ELF_PHOFF),
        self.elf64_off(ELF_SHOFF),
        self.elf64_word(ELF_FLAGS),
        self.elf64_half(ELF_EHSIZE),
        self.elf64_half(ELF_PHENTSIZE),
        self.elf64_half(ELF_PHNUM),
        self.elf64_half(ELF_SHENTSIZE),
        self.elf64_half(ELF_SHNUM),
        self.elf64_half(ELF_SHSTRNDX),
        ])
    return ''.join([magic, rest_of_elf_header])

  def elf64_addr(self, data):
    """Returns the packed binary whose size is addr as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % self.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_byte(self, data):
    """Returns the packed binary whose size is byte as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sc' % self.byte_ordering_fmt
    return struct.pack(format_str, chr(data))

  def elf64_half(self, data):
    """Returns the packed binary whose size is half as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sH' % self.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_off(self, data):
    """Returns the packed binary whose size is off as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % self.byte_ordering_fmt
    return struct.pack(format_str, data)

  def elf64_word(self, data):
    """Returns the packed binary whose size is word as specified by ELF64.

    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % self.byte_ordering_fmt
    return struct.pack(format_str, data)



def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-o', '--output', metavar="Output", type=str,
                      nargs='?', const=True,
                      help='The name of the output file. If the name of '
                      'the file is not supplied, it will be same as the '
                      'first input program file name.')
  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  
  if args.output and isinstance(args.output, str):
    filename = args.output
  else:
    first_input_filename = args.filenames[0]
    if first_input_filename[-6:] == '.pl241':
      filename = first_input_filename[:-6]
    else:
      filename = first_input_filename


  elf = ELF(filename)
  elf.build()

  return elf


if __name__ == '__main__':
  elf = bootstrap()
