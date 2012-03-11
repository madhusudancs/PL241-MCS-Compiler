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
      Others: Look at the docstrings for the indvidual properties and ELF
      man page.
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
  def e_magic(self):
    """Returns the ELF Magic constructed from the individual bytes!

    Yes this is really magic! Don't try to understand why!
    """
    return ''.join([
        self.ELF_IDENT_MAG0,
        self.ELF_IDENT_MAG1,
        self.ELF_IDENT_MAG2,
        self.ELF_IDENT_MAG3,
        self.ELF_IDENT_CLASS,
        self.ELF_IDENT_DATA,
        self.ELF_IDENT_VERSION,
        self.ELF_IDENT_PAD
        ])

  @property
  def e_type(self):
    """Returns the ELF type byte.
    """
    return self.__class__.elf64_half(self.ELFTYPE.ET_EXEC)

  @property
  def e_machine(self):
    """Returns the ELF machine type byte.
    """
    return self.__class__.elf64_half(self.elf_machine)

  @property
  def e_version(self):
    """Returns the ELF version byte.
    """
    return self.__class__.elf64_word(self.ELFVERSION.EV_CURRENT)

  @property
  def e_entry(self):
    """Returns the offset to the program entry point.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_addr(self.entry if self.entry else 0)

  @property
  def e_phoff(self):
    """Returns the offset to the program header table in bytes.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_off(self.phoff if self.phoff else 0)

  @property
  def e_shoff(self):
    """Returns the offset to the section header table in bytes.

    Should be calculated dynamically.
    """
    return self.__class__.elf64_off(self.shoff if self.shoff else 0)

  @property
  def e_flags(self):
    """Returns the ELF_IDENT_MAG3 byte.

    x86_64 architecture doesn't define any machine-specific flags, so it is
    set to 0 for now. Later this can be changed to enumerations based on
    the machine architecture.
    """
    if not self.flags:
      self.flags = 0
    return self.__class__.elf64_word(self.flags)

  @property
  def e_ehsize(self):
    """Returns the the ELF's header size in bytes.

    Hard-coded to 64 bytes for now.
    """
    if not self.ehsize:
      self.ehsize = self.ELF_STD_SIZE

    return self.__class__.elf64_half(self.ehsize)

  @property
  def e_phentsize(self):
    """Returns the holds the size in bytes of one entry in the file's program
    header table.

    All entries must be of the same size. Should be calculated dynamically.
    """
    return self.__class__.elf64_half(self.phentsize if self.phentsize else 0)

  @property
  def e_phnum(self):
    """Returns the number of entries in the program header table.

    If the file has no program header table, this holds the value zero. Should
    be calculated dynamically.
    """
    return self.__class__.elf64_half(self.phnum if self.phnum else 0)

  @property
  def e_shentsize(self):
    """Returns the section header's size in bytes.

    A section header is one entry in the section header table. All entries are
    the same size. Should be calculated dymamically.
    """
    return self.__class__.elf64_half(self.shentsize if self.shentsize else 0)

  @property
  def e_shnum(self):
    """Returns the number of entries in the section header table.

    If the file has no section header table, this holds the value zero.
    Should be calculated dymamically.
    """
    return self.__class__.elf64_half(self.shnum if self.shnum else 0)

  @property
  def e_shstrndx(self):
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
    self.header = ''.join([
        self.e_magic,
        self.e_type,
        self.e_machine,
        self.e_version,
        self.e_entry,
        self.e_phoff,
        self.e_shoff,
        self.e_flags,
        self.e_ehsize,
        self.e_phentsize,
        self.e_phnum,
        self.e_shentsize,
        self.e_shnum,
        self.e_shstrndx
        ])

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


class ProgramHeader(object):
  """Abstracts the Program header.
  """

  __metaclass__ = ELFMetaclass

  class TYPE(object):
    """Enumeration of program header type.
    """
    PT_NULL    = 0x0
    PT_LOAD    = 0x1
    PT_DYNAMIC = 0x2
    PT_INTERP  = 0x3
    PT_NOTE    = 0x4
    PT_SHLIB   = 0x5
    PT_PHDR    = 0x6
    PT_LOPROC  = 0x70000000
    PT_HIPROC  = 0x7fffffff

  class FLAGS(object):
    """Enumeration of program header flags.
    """
    PF_N = 0x0   # Invalid flag.
    PF_X = 0x1
    PF_W = 0x2
    PF_R = 0x4

  counter = 0

  @classmethod
  def reset_counter(cls):
    """Resets the counter for new Program Header table entry.
    """
    cls.counter = 0

  def __init__(self, endianness='little', ph_type=None, offset=None,
               vaddr=None, paddr=None, filesz=None, memsz=None, flags=None,
               align=None):
    """Constructs the ProgramHeader object required to generate them.

    Args:
      Look at elf manpage.
    """
    self.index = self.__class__.counter
    self.__class__.counter += 1

    if endianness == 'little':
      self.__class__.byte_ordering_fmt = '<'
    elif endianness == 'big':
      self.__class__.byte_ordering_fmt = '>'
    else:
      raise TypeError('Invalid byte-order type "%s".' % endianness)

    self.type = ph_type
    self.offset = offset
    self.vaddr = vaddr
    self.paddr = paddr
    self.filesz = filesz
    self.memsz = memsz
    self.flags = flags
    self.align = align

    # The actual byte encoded program header to be built for this object.
    self.header = None

  @property
  def p_type(self):
    if not self.type:
      ph_type = self.TYPE.PT_NULL
    elif self.type == 'LOAD':
      ph_type = self.TYPE.PT_LOAD

    return self.__class__.elf64_word(ph_type)

  @property
  def p_offset(self):
    return self.__class__.elf64_off(self.offset if self.offset else 0)

  @property
  def p_vaddr(self):
    return self.__class__.elf64_addr(self.vaddr if self.vaddr else 0)

  @property
  def p_paddr(self):
    return self.__class__.elf64_addr(self.paddr if self.paddr else 0)

  @property
  def p_filesz(self):
    return self.__class__.elf64_xword(self.filesz if self.filesz else 0)

  @property
  def p_memsz(self):
    return self.__class__.elf64_xword(self.memsz if self.memsz else 0)

  @property
  def p_flags(self):
    flags = self.FLAGS.PF_N
    if 'X' in self.flags:
      flags += self.FLAGS.PF_X
    if 'W' in self.flags:
      flags += self.FLAGS.PF_W
    if 'R' in self.flags:
      flags += self.FLAGS.PF_R

    return self.__class__.elf64_word(flags)

  @property
  def p_align(self):
    return self.__class__.elf64_xword(self.align if self.align else 0)

  def build(self):
    """Returns the program header as bytes.
    """
    self.header = ''.join([
        self.p_type,
        self.p_flags,
        self.p_offset,
        self.p_vaddr,
        self.p_paddr,
        self.p_filesz,
        self.p_memsz,
        self.p_align
        ])

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



    Args:
      data: The data that should be formatted.
    """
    format_str = '%sQ' % self.byte_ordering_fmt
    return struct.pack(format_str, data)



class ELF(object):
  """Builds the ELF binary file for the given inputs.
  """

  __metaclass__ = ELFMetaclass

  def __init__(self, filename, elf_class=64, endianness='little',
               architecture='x86_64', instructions=''):
    """Constructs the ELF object required to generate ELF binaries.

    Args:
      filename: The name of the binary file the ELF should be written to.
      elf_class: The class of the ELF file, can be 32 or 64.
      endianness: The byte ordering. Can be little or big
      architecture: The architecture for which this binary should be generated.
      instructions: The byte-encoded instructions for which the binary is
          being generated.

    VERY IMPORTANT: The instructions are already assumed to be byte-encoded in
        the required byte order. It is just dumped into the ELF file as it is
        supplied to this constructor!!!
    """
    self.filename = filename
    self.elf_class = elf_class
    self.endianness = endianness
    self.architecture = architecture
    self.instructions = instructions

    # Open the file for binary writing.
    self.filepointer = open(filename, 'wb')

  def build(self):
    """Builds the binary file for the given input.
    """
    elf_header = ELFHeader()

    phnum = 1
    ph_load_header = ProgramHeader(ph_type='LOAD', offset=0x0, vaddr=0x400000,
                                   paddr=0x400000, filesz=0x0, memsz=0x0,
                                   flags='RX', align=0x200000)
    ph_load_header.build()

    phentsize = len(ph_load_header)

    phsize = (phnum * phentsize)

    # A phheader padding is added to align to 64 bytes.
    phpaddingsize = 0x40 - (phsize % 0x40)
    phpadding = '0x0' * phpaddingsize

    elf_header.phentsize = phentsize
    elf_header.phnum = phnum

    # program header offset hard-coded to end of the elf header.
    elf_header.phoff = elf_header.ELF_STD_SIZE

    # The program starts at virtual address followed by the program headers
    elf_header.entry = elf_header.phoff + ph_load_header.vaddr + \
        phsize + phpaddingsize

    headers = ''.join([
        str(elf_header.build()), str(ph_load_header), phpadding])

    self.filepointer.write(headers)

  def __del__(self):
    """Ensure that the write file is closed.
    """
    if not self.filepointer.closed:
      self.filepointer.close()


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
