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


class ELFHeader(object):
  """Abstracts the ELF header.
  """

  __metaclass__ = ELFMetaclass

  class ELFCLASS(object):
    """Enumeration elf classes.
    """
    ELFCLASSNONE = 0x00     # Invalid class
    ELFCLASS32   = 0x01     # 32-bit object file
    ELFCLASS64   = 0x02     # 64-bit object file


  class ELFDATA(object):
    """Enumeration of elf data byte ordering.
    """
    ELFDATANONE = 0x00      # Invalid data encoding
    ELFDATA2LSB = 0x01      # Little-endian encoding
    ELFDATA2MSB = 0x02      # Big-endian encoding


  class ELFTYPE(object):
    """Enumeration of elf file types.
    """
    ET_NONE   = 0x0        # No file type
    ET_REL    = 0x1        # Relocatable file
    ET_EXEC   = 0x2        # Executable file
    ET_DYN    = 0x3        # Shared object file
    ET_CORE   = 0x4        # Core file
    ET_LOPROC = 0xFF00     # Processor-specific
    ET_HIPROC = 0xFFFF     # Processor-specific


  class ELFMACHINE(object):
    """Enumeration of machine types.

    Contains only the types that are needed. Should be expanded when new
    machine types are needed.
    """
    EM_386    = 0x3         # Intel 80386
    EM_X86_64 = 0x3E        # AMD64


  class ELFVERSION(object):
    """Enumeration of object file version.
    """
    EV_NONE    = 0x0        # Invalid version
    EV_CURRENT = 0x1        # Current version

  ELF_IDENT_PAD_SIZE = 0x9                    # Note this is the number of
                                              # padding bytes, not the value

  ELF_STD_SIZE = 0x40

  def __init__(self, elf_class=64, endianness='little', architecture='x86_64',
               entry=None, phoff=None, shoff=None, flags=None, ehsize=None,
               phentsize=None, phnum=None, shentsize=None, shnum=None,
               shstrndx=None):
    """Constructs the ELFHeader object required to generate ELF binaries.

    Args:
      elf_class: The elf file class. Must be one of 32 or 64 for ELF32 and
          ELF64 respectively.
      endianness: The endianness can be either little or big.
      architecture: The CPU architecture we are building this ELF binary for.
      Others: Look at the docstrings for the indvidual properties.
    """

    if elf_class == 64:
      self.elf_class = self.ELFCLASS.ELFCLASS64
    elif elf_class == 32:
      self.elf_class = self.ELFCLASS.ELFCLASS32
    else:
      raise TypeError('Invalid elf type "%s".' % elf_class)

    if endianness == 'little':
      self.byte_ordering_fmt = '<'
      self.elf_data = self.ELFDATA.ELFDATA2LSB
    elif endianness == 'big':
      self.byte_ordering_fmt = '>'
      self.elf_data = self.ELFDATA.ELFDATA2MSB
    else:
      raise TypeError('Invalid byte-order type "%s".' % endianness)

    # Make this available at class level too.
    self.__class__.byte_ordering_fmt = self.byte_ordering_fmt

    if architecture == 'x86':
      self.elf_machine = self.ELFMACHINE.EM_386
    elif architecture == 'x86_64':
      self.elf_machine = self.ELFMACHINE.EM_X86_64
    else:
      raise TypeError('Architecture %s not supported.' % architecture)

    self.entry = entry
    self.phoff = phoff
    self.shoff = shoff
    self.flags = flags
    self.ehsize = ehsize
    self.phentsize = phentsize
    self.phnum = phnum
    self.shentsize = shentsize
    self.shnum = shnum
    self.shstrndx = shstrndx

    # The actual byte encoded program header to be built for this object.
    self.header = None

  @property
  def ELF_IDENT_MAG0(self):
    """Returns the ELF_IDENT_MAG0 byte.
    """
    return self.__class__.elf64_byte(0x7F)

  @property
  def ELF_IDENT_MAG1(self):
    """Returns the ELF_IDENT_MAG1 byte.
    """
    return self.__class__.elf64_byte(0x45)

  @property
  def ELF_IDENT_MAG2(self):
    """Returns the ELF_IDENT_MAG2 byte.
    """
    return self.__class__.elf64_byte(0x4C)

  @property
  def ELF_IDENT_MAG3(self):
    """Returns the ELF_IDENT_MAG3 byte.
    """
    return self.__class__.elf64_byte(0x46)

  @property
  def ELF_IDENT_CLASS(self):
    """Returns the ELF_IDENT_CLASS byte.
    """
    return self.__class__.elf64_byte(self.ELFCLASS.ELFCLASS64)

  @property
  def ELF_IDENT_DATA(self):
    """Returns the ELF_IDENT_DATA byte.
    """
    return self.__class__.elf64_byte(self.ELFDATA.ELFDATA2LSB)

  @property
  def ELF_IDENT_VERSION(self):
    """Returns the ELF_IDENT_VERSION byte.
    """
    return self.__class__.elf64_byte(self.ELFVERSION.EV_CURRENT)

  @property
  def ELF_IDENT_PAD(self):
    """Returns the ELF_IDENT's padding byte.
    """
    return self.__class__.elf64_byte(0x00) * self.ELF_IDENT_PAD_SIZE

  @property
  def ELF_VERSION(self):
    """Returns the ELF version byte.
    """
    return self.__class__.elf64_word(self.ELFVERSION.EV_CURRENT)

  @property
  def ELF_TYPE(self):
    """Returns the ELF type byte.
    """
    return self.__class__.elf64_half(self.ELFTYPE.ET_EXEC)

  @property
  def ELF_MACHINE(self):
    """Returns the ELF machine type byte.
    """
    return self.__class__.elf64_half(self.elf_machine)

  @property
  def ELF_ENTRY(self):
    """Returns the offset to the program entry point.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_addr(self.entry if self.entry else 0)

  @property
  def ELF_PHOFF(self):
    """Returns the offset to the program header table in bytes.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_off(self.phoff if self.phoff else 0)

  @property
  def ELF_SHOFF(self):
    """Returns the offset to the section header table in bytes.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_off(self.shoff if self.shoff else 0)

  ELF_SHSTRNDX       = 0x0       # This member holds the
  @property
  def ELF_FLAGS(self):
    """Returns the ELF_IDENT_MAG3 byte.

    x86_64 architecture doesn't define any machine-specific flags, so it is
    set to 0 for now. Later this can be changed to enumerations based on
    the machine architecture.
    """
    if not self.flags:
      self.flags = 0
    return self.__class__.elf64_word(self.flags)

  @property
  def ELF_EHSIZE(self):
    """Returns the the ELF's header size in bytes.

    Hard-coded to 64 bytes for now.
    """
    if not self.ehsize:
      self.ehsize = self.ELF_STD_SIZE

    return self.__class__.elf64_half(self.ehsize)

  @property
  def ELF_PHENTSIZE(self):
    """Returns the holds the size in bytes of one entry in the file's program
    header table.

    All entries must be of the same size. Should be calculated dynamically.
    """
    return self.__class__.elf64_half(self.phentsize if self.phentsize else 0)

  @property
  def ELF_PHNUM(self):
    """Returns the number of entries in the program header table.

    If the file has no program header table, this holds the value zero. Should
    be calculated dynamically.
    """
    return self.__class__.elf64_half(self.phnum if self.phnum else 0)

  @property
  def ELF_SHENTSIZE(self):
    """Returns the section header's size in bytes.

    A section header is one entry in the section header table. All entries are
    the same size. Should be calculated dymamically.
    """
    return self.__class__.elf64_half(self.shentsize if self.shentsize else 0)

  @property
  def ELF_SHNUM(self):
    """Returns the number of entries in the section header table.

    If the file has no section header table, this holds the value zero.
    Should be calculated dymamically.
    """
    return self.__class__.elf64_half(self.shnum if self.shnum else 0)

  @property
  def ELF_SHSTRNDX(self):
    """Returns the section header table index of the entry associated with the
    section name string table.

    If the file has no section name string table, this member holds the value
    SHN_UNDEF. For now it is hard coded to 0. But can be calculated
    dynamically if needed.
    """
    return self.__class__.elf64_half(self.shstrndx if self.shstrndx else 0)

  def build(self):
    """Returns the ELF header as bytes.
    """
    magic = ''.join([
        self.ELF_IDENT_MAG0,
        self.ELF_IDENT_MAG1,
        self.ELF_IDENT_MAG2,
        self.ELF_IDENT_MAG3,
        self.ELF_IDENT_CLASS,
        self.ELF_IDENT_DATA,
        self.ELF_IDENT_VERSION,
        self.ELF_IDENT_PAD
        ])

    rest_of_elf_header = ''.join([
        self.ELF_TYPE,
        self.ELF_MACHINE,
        self.ELF_VERSION,
        self.ELF_ENTRY,
        self.ELF_PHOFF,
        self.ELF_SHOFF,
        self.ELF_FLAGS,
        self.ELF_EHSIZE,
        self.ELF_PHENTSIZE,
        self.ELF_PHNUM,
        self.ELF_SHENTSIZE,
        self.ELF_SHNUM,
        self.ELF_SHSTRNDX,
        ])
    self.header = ''.join([magic, rest_of_elf_header])

    # So that they can be chained.
    return self

  def __len__(self):
    """Returns the size of the byte-encoded string of this object.
    """
    return len(self.header)

  def __str__(self):
    """Returns the byte-encoded string of this program header object.
    """
    return self.header

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
