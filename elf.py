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


import collections
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
    format_str = '%sB' % cls.byte_ordering_fmt
    return struct.pack(format_str, data)

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
      self.type = self.TYPE.PT_NULL

    return self.__class__.elf64_word(self.type)

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
    if not self.flags:
      self.flags = self.FLAGS.PF_N

    return self.__class__.elf64_word(self.flags)

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


class SectionHeader(object):
  """Abstracts the Section header.
  """

  __metaclass__ = ELFMetaclass

  class SPECIAL_INDEXES(object):
    """Enumeration of Section Header's special indexes.
    """
    SHN_UNDEF     = 0x0
    SHN_LORESERVE = 0xFF00
    SHN_LOPROC    = 0xFF00
    SHN_HIPROC    = 0xFF1F
    SHN_ABS       = 0xFFF1
    SHN_COMMON    = 0xFFF2
    SHN_HIRESERVE = 0xFFFF

  class TYPE(object):
    """Enumeration of Section Header types.
    """
    SHT_NULL     = 0x0
    SHT_PROGBITS = 0x1
    SHT_SYMTAB   = 0x2
    SHT_STRTAB   = 0x3
    SHT_RELA     = 0x4
    SHT_HASH     = 0x5
    SHT_DYNAMIC  = 0x6
    SHT_NOTE     = 0x7
    SHT_NOBITS   = 0x8
    SHT_REL      = 0x9
    SHT_SHLIB    = 0xA
    SHT_DYNSYM   = 0xB
    SHT_LOPROC   = 0x70000000
    SHT_HIPROC   = 0x7FFFFFFF
    SHT_LOUSER   = 0x80000000
    SHT_HIUSER   = 0xFFFFFFFF

  class FLAGS(object):
    """Enumeration of Section Header flags.
    """
    SHF_UNDEF     = 0x0
    SHF_WRITE     = 0x1
    SHF_ALLOC     = 0x2
    SHF_EXECINSTR = 0x4
    SHF_MASKPROC  = 0xF0000000

  SHN_UNDEF = 0x0

  counter = 0

  @classmethod
  def reset_counter(cls):
    """Resets the counter for new Program Header table entry.
    """
    cls.counter = 0

  def __init__(self, endianness='little', name=None, sh_type=None, flags=None,
               addr=None, offset=None, size=None, link=None, info=None,
               addralign=None, entsize=None):
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

    self.name = name
    self.type = sh_type
    self.flags = flags
    self.addr = addr
    self.offset = offset
    self.size = size
    self.link = link
    self.info = info
    self.addralign = addralign
    self.entsize = entsize

    # The actual byte encoded program header to be built for this object.
    self.header = None

  @property
  def sh_name(self):
    return self.__class__.elf64_word(self.name if self.name else 0)

  @property
  def sh_type(self):
    if not self.type:
      self.type = self.TYPE.SHT_NULL

    return self.__class__.elf64_word(self.type)

  @property
  def sh_flags(self):
    if not self.flags:
      self.flags = self.FLAGS.SHF_UNDEF

    return self.__class__.elf64_xword(self.flags)

  @property
  def sh_addr(self):
    return self.__class__.elf64_addr(self.addr if self.addr else 0)

  @property
  def sh_offset(self):
    return self.__class__.elf64_off(self.offset if self.offset else 0)

  @property
  def sh_size(self):
    return self.__class__.elf64_xword(self.size if self.size else 0)

  @property
  def sh_link(self):
    return self.__class__.elf64_word(
        self.link if self.link else self.SHN_UNDEF)

  @property
  def sh_info(self):
    return self.__class__.elf64_word(self.info if self.info else 0)

  @property
  def sh_addralign(self):
    return self.__class__.elf64_xword(self.addralign if self.addralign else 0)

  @property
  def sh_entsize(self):
    return self.__class__.elf64_xword(self.entsize if self.entsize else 0)

  def build(self):
    """Returns the Section Header as bytes.
    """
    self.header = ''.join([
        self.sh_name,
        self.sh_type,
        self.sh_flags,
        self.sh_addr,
        self.sh_offset,
        self.sh_size,
        self.sh_link,
        self.sh_info,
        self.sh_addralign,
        self.sh_entsize
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


class SYMTAB(object):
  """Abstracts the Symbol Table for the ELF binaries.
  """

  __metaclass__ = ELFMetaclass

  class BIND(object):
    """Enumeration of Symbol Table's bind values.
    """
    STB_LOCAL  = 0x0
    STB_GLOBAL = 0x1
    STB_WEAK   = 0x2
    STB_LOPROC = 0xD
    STB_HIPROC = 0xF

  class TYPE(object):
    """Enumeration of Symbol Table's type values.
    """
    STT_NOTYPE  = 0x0
    STT_OBJECT  = 0x1
    STT_FUNC    = 0x2
    STT_SECTION = 0x3
    STT_FILE    = 0x4
    STT_LOPROC  = 0xD
    STT_HIPROC  = 0xF


  STN_UNDEF = 0x0

  counter = 0

  @classmethod
  def reset_counter(cls):
    """Resets the counter for new Program Header table entry.
    """
    cls.counter = 0

  def __init__(self, endianness='little', name=None, bind=None, info_type=None,
               other=None, shndx=None, value=None, size=None):
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

    self.name = name
    self.bind = bind
    self.type = info_type
    self.other = other
    self.shndx = shndx
    self.value = value
    self.size = size

    # The actual byte encoded program header to be built for this object.
    self.entry = None

  @property
  def st_name(self):
    return self.__class__.elf64_word(self.name if self.name else 0)

  @property
  def st_info(self):
    if not self.bind:
      self.bind = 0x0
    if not self.type:
      self.type = 0x0
    info = (self.bind << 4 | (self.type & 0xF))
    return self.__class__.elf64_byte(info)

  @property
  def st_other(self):
    return self.__class__.elf64_byte(self.other if self.other else 0)

  @property
  def st_shndx(self):
    return self.__class__.elf64_half(self.shndx if self.shndx else 0)

  @property
  def st_value(self):
    return self.__class__.elf64_addr(self.value if self.value else 0)

  @property
  def st_size(self):
    return self.__class__.elf64_xword(self.size if self.size else 0)

  def build(self):
    """Returns the Symbol Table entry as bytes.
    """
    self.entry = ''.join([
        self.st_name,
        self.st_info,
        self.st_other,
        self.st_shndx,
        self.st_value,
        self.st_size,
        ])

    # So that they can be chained.
    return self

  def __len__(self):
    """Returns the size of the byte-encoded string of this object.
    """
    return len(self.entry)

  def __str__(self):
    """Returns the byte-encoded string of this program header object.
    """
    return self.entry


class STRTAB(object):
  """Abstracts the string table.

  There really isn't much in this class. String Table is just a set of null
  terminated strings. The String Table always starts with a null terminated
  string and ends with a null terminated string.

  NOTE: This class doesn't include the ELFMetaclass since it is not required.
  This is a table not a header, so there is no real need for the functionality
  that the ELFMetaclass provides since the length's of the entities and their
  byte ordering here is dicated by the table and not by the ELF rules.
  """

  def __init__(self):
    """Construct the String Table.
    """
    # A dictionary containing the string as the key and its position in the
    # string table as the value for fast look-ups.
    self.strings = {
        '\x00': 0
        }

    # A dictionary containing the position of the string as the key and the
    # string itself as the value for look up by position.
    self.positions = collections.OrderedDict({
        0: '\x00'
        })

    # The index of the next position in the string table. 0 is already
    # taken up.
    self.next_position = 1

  def append(self, string):
    """Appends the string to the string table.

    Args:
      string: The string that should be appended to the string table.
    """
    string = string + '\x00'
    self.strings[string] = self.next_position
    self.positions[self.next_position] = string
    self.next_position += len(string)

  def __getitem__(self, key):
    """Returns the index of the given string in the string table.

    Args:
      key: The string whose position should be retrieved.
    """
    key += '\x00'
    return self.strings[key]

  def __len__(self):
    """Returns the size of the byte-encoded string of this object.
    """
    length = 0
    for i in self.positions:
      length += len(self.positions[i])

    return length

  def __str__(self):
    """Returns the byte-encoded string of this program header object.
    """
    string = ''
    for i in self.positions:
      string += self.positions[i]

    return string


class ELF(object):
  """Builds the ELF binary file for the given inputs.
  """

  __metaclass__ = ELFMetaclass

  # Hard-coded data start memory virtual address and physical address
  # FIXME: Think of a way to make it dynamic.
  DATA_VADDR        = 0x600000
  DATA_PADDR        = 0x600000

  # Hard-coded program start memory virtual address and physical address
  # FIXME: Think of a way to make it dynamic.
  PROGRAM_VADDR     = 0x400000
  PROGRAM_PADDR     = 0x400000

  # Hard-coded section alignment value.
  # FIXME: Think of a way to make it dynamic.
  SECTION_ALIGNMENT = 0x200000

  STRTAB_SHNDX = 0x4

  ELF_HEADER_SHSTRNDX = 0x3

  def __init__(self, filename, linker, global_memory_size, elf_class=64,
               endianness='little', architecture='x86_64'):
    """Constructs the ELF object required to generate ELF binaries.

    Args:
      filename: The name of the binary file the ELF should be written to.
      linker: The linker object
      global_memory_size: The size of the global memory required. The whole
          global memory is assumed to be uninitialized and hence will be loaded
          in the .bss segment.
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
    self.linker = linker
    self.global_memory_size = global_memory_size
    self.elf_class = elf_class
    self.endianness = endianness
    self.architecture = architecture

    # Final binary to be generated.
    self.binary = None

    if endianness == 'little':
      self.__class__.byte_ordering_fmt = '<'
    elif endianness == 'big':
      self.__class__.byte_ordering_fmt = '>'
    else:
      raise TypeError('Invalid byte-order type "%s".' % endianness)

    # All the intermediate objects built

    # ELF Header object
    self.elf_header = None

    # Section header string table related objects.
    self.shstrtaboff = None
    self.shstrtab = None
    self.shstrtabsize = None

    # String table related objects.
    self.strtaboff = None
    self.strtab = None
    self.strtabsize = None

    # Symbol table related objects.
    self.symtaboff = None
    self.null_sym_entry = None
    self.symtab_start_entry = None
    self.function_symtab_entries = None
    self.symtab_locals = None
    self.symtabnum = None
    self.symtabsize = None

    # Program header table related objects.
    self.phoff = None
    self.ph_load_header = None
    self.ph_bss_header = None
    self.phnum = None
    self.phentsize = None
    self.phsize = None
    self.phpaddingsize = None

    # Instructions offsets related objects
    self.instructions_size = None
    self.instructionsoff = None
    self.instructionsvoff = None
    self.instructionspaddingsize = None

    # Section header related objects.
    self.sh_null_header = None
    self.sh_text_header = None
    self.sh_bss_header = None
    self.sh_shstrtab_header = None
    self.sh_strtab_header = None
    self.sh_symtab_header = None
    self.shnum = None
    self.shentsize = None
    self.shsize = None
    self.shpaddingsize = None

  def padding(self, num_bytes):
    """Returns the padding string.

    Args:
      num_bytes: Number of bytes that must be padded.
    """
    return self.__class__.elf64_byte(0x00) * num_bytes

  def build_elf_header(self):
    """Builds the initial elf header object.
    """
    self.elf_header = ELFHeader()

  def build_shstrtab(self):
    """Builds the section header string table containing the section names.
    """
    # Start of Section String Table building.

    # Hard coded to the end of ELF header for now.
    self.shstrtaboff = self.elf_header.ELF_STD_SIZE

    self.shstrtab = STRTAB()
    self.shstrtab.append('.shstrtab')
    self.shstrtab.append('.strtab')
    self.shstrtab.append('.symtab')
    self.shstrtab.append('.text')
    if self.global_memory_size:
      self.shstrtab.append('.bss')

    self.shstrtabsize = len(self.shstrtab)

  def build_strtab(self):
    """Builds the string table.
    """
    # Start of Instructions String Table building.
    self.strtaboff = self.shstrtaboff + self.shstrtabsize

    self.strtab = STRTAB()
    self.strtab.append('_start')
    for function_name in self.linker.function_offset_map:
      self.strtab.append('%s' % function_name)

    self.strtabsize = len(self.strtab)

  def build_symtab(self):
    """Builds the symbol table.
    """
    # Start of Instructions String Table building.
    self.symtaboff = self.strtaboff + self.strtabsize

    self.null_sym_entry = SYMTAB()
    self.null_sym_entry.build()

    # Defines the starting point of the program
    # Value which is the memory offset needs to be recalculated again.
    self.symtab_start_entry = SYMTAB(name=self.strtab['_start'],
                                     bind=SYMTAB.BIND.STB_GLOBAL,
                                     info_type=SYMTAB.TYPE.STT_FUNC, other=0x0,
                                     shndx=0x1, value=0x0,
                                     size=0x0)
    self.symtab_start_entry.build()

    self.function_symtab_entries = []
    # NOTE: For shndx the index into the section header table, we always
    # put .text section as the second entry, which has the index 1, this
    # is the convention followed. So we just set shndx to 1.

    # NOTE: The value entry will be reset again when we calculate the
    # instructionsoffset
    for function_name, entry in self.linker.function_offset_map.iteritems():
      sym_entry = SYMTAB(name=self.strtab[function_name],
                         bind=SYMTAB.BIND.STB_GLOBAL,
                         info_type=SYMTAB.TYPE.STT_FUNC, other=0x0,
                         shndx=0x1, value=entry['offset'],
                         size=entry['size'])
      sym_entry.build()
      self.function_symtab_entries.append(sym_entry)

    self.symtab_locals = 0
    self.symtabnum = SYMTAB.counter
    self.symtabsize = len(self.null_sym_entry) * self.symtabnum

  def build_program_headers(self):
    """Builds the program headers.
    """
    # Start of program header building.
    self.phoff = self.symtaboff + self.symtabsize

    # Virtual address, physical address and alignments are hard-coded for now.
    # FIXME: Virtual address, physical address and alignment should be made
    # dynamic if necessary.
    self.ph_load_header = ProgramHeader(
        ph_type=ProgramHeader.TYPE.PT_LOAD, offset=0x0,
        vaddr=self.PROGRAM_VADDR, paddr=self.PROGRAM_PADDR,
        filesz=self.instructions_size, memsz=self.instructions_size,
        flags=ProgramHeader.FLAGS.PF_X | ProgramHeader.FLAGS.PF_R,
        align=self.SECTION_ALIGNMENT)

    self.ph_load_header.build()

    if self.global_memory_size:
      self.ph_bss_header = ProgramHeader(
        ph_type=ProgramHeader.TYPE.PT_LOAD, offset=0x0,
        vaddr=self.DATA_VADDR, paddr=self.DATA_PADDR,
        filesz=0x0, memsz=self.global_memory_size,
        flags=ProgramHeader.FLAGS.PF_R | ProgramHeader.FLAGS.PF_W,
        align=self.SECTION_ALIGNMENT)

      self.ph_bss_header.build()

    self.phnum = ProgramHeader.counter
    self.phentsize = len(self.ph_load_header)

    self.phsize = (self.phnum * self.phentsize)

    # A phheader padding is added to align to 16 bytes.
    self.phpaddingsize = 0x10 - (self.phsize % 0x10) if \
        (self.phsize % 0x10) else 0x0

  def compute_instructions_offset(self):
    """Computes the offsets for the instructions starting point.
    """
    # Start of instructions
    # The program starts at virtual address followed by the program headers
    self.instructionsoff = self.phoff + self.phsize + self.phpaddingsize

    self.instructionsvoff = self.instructionsoff + self.ph_load_header.vaddr

  def build_section_headers(self):
    """Builds the section headers.
    """
    # Start of section headers building
    self.sh_null_header = SectionHeader()
    self.sh_null_header.build()


    # Address alignment is hard-coded for now since ELF64 wants 16 bytes
    # alignment.
    # FIXME: Make the address alignment dynamic.
    self.sh_text_header = SectionHeader(
        name=self.shstrtab['.text'], sh_type=SectionHeader.TYPE.SHT_PROGBITS,
        addr=self.instructionsvoff, offset=self.instructionsoff,
        size=self.instructions_size,
        flags=(SectionHeader.FLAGS.SHF_ALLOC | \
            SectionHeader.FLAGS.SHF_EXECINSTR),
        link=SectionHeader.SHN_UNDEF, info=0x0, addralign=0x10, entsize=0x0)

    self.sh_text_header.build()

    if self.global_memory_size:
      # Address alignment is hard-coded for now since ELF64 wants 16 bytes
      # alignment.
      # FIXME: Make the address alignment dynamic.
      self.sh_bss_header = SectionHeader(
          name=self.shstrtab['.bss'], sh_type=SectionHeader.TYPE.SHT_NOBITS,
          addr=self.DATA_VADDR, offset=0x0,
          size=self.global_memory_size,
          flags=(SectionHeader.FLAGS.SHF_ALLOC | \
              SectionHeader.FLAGS.SHF_WRITE),
          link=SectionHeader.SHN_UNDEF, info=0x0, addralign=0x10, entsize=0x0)

      self.sh_bss_header.build()

    # Since this is byte aligned, since it is a string, the address alignment
    # of 1 is fixed always.
    self.sh_shstrtab_header = SectionHeader(
        name=self.shstrtab['.shstrtab'], sh_type=SectionHeader.TYPE.SHT_STRTAB,
        addr=0x0, offset=self.shstrtaboff, size=self.shstrtabsize,
        link=SectionHeader.SHN_UNDEF, info=0x0, addralign=0x1, entsize=0x0)

    self.sh_shstrtab_header.build()

    # Since this is byte aligned, since it is a string, the address alignment
    # of 1 is fixed always.
    self.sh_strtab_header = SectionHeader(
        name=self.shstrtab['.strtab'], sh_type=SectionHeader.TYPE.SHT_STRTAB,
        addr=0x0, offset=self.strtaboff, size=self.strtabsize,
        link=SectionHeader.SHN_UNDEF, info=0x0, addralign=0x1, entsize=0x0)

    self.sh_strtab_header.build()

    # Since this is byte aligned, since it is a string, the address alignment
    # of 1 is fixed always.
    self.sh_symtab_header = SectionHeader(
        name=self.shstrtab['.symtab'], sh_type=SectionHeader.TYPE.SHT_SYMTAB,
        addr=0x0, offset=self.symtaboff, size=self.symtabsize,
        link=self.STRTAB_SHNDX if self.global_memory_size else \
            self.STRTAB_SHNDX - 1,
        info=self.symtab_locals + 1, addralign=0x8,
        entsize=len(self.null_sym_entry))

    self.sh_symtab_header.build()

    self.shnum = SectionHeader.counter
    self.shentsize = len(self.sh_null_header)

    self.shsize = (self.shnum * self.shentsize)

    # A shheader padding is added to align to 64 bytes.
    self.shpaddingsize = 0x10 - (self.shsize % 0x10) if \
        (self.shsize % 0x10) else 0x0

  def build_upto_instructions_offset(self):
    """Builds upto computing the instructions offset for linking globals.
    """
    self.instructions_size = len(self.linker.binary)

    self.build_elf_header()

    self.build_shstrtab()

    self.build_strtab()

    self.build_symtab()

    self.build_program_headers()

    self.compute_instructions_offset()

  def build(self):
    """Builds the final binary file by interlinking the headers.

    When this function is called, it is assumed that
    self.build_upto_instructions_offset() is already called from elsewhere
    and hence begins by calling the method that builds section headers.
    """
    # An instructions padding is added to align to 16 bytes.
    self.instructionspaddingsize = 0x10 - (self.instructions_size % 0x10) if (
        self.instructions_size % 0x10) else 0x0

    self.build_section_headers()

    # Interlinking updates

    # Update the value attribute of symbol table entries for start and
    # function names and rebuild
    self.symtab_start_entry.value += self.instructionsvoff
    self.symtab_start_entry.build()

    for entry in self.function_symtab_entries:
      entry.value += self.instructionsvoff
      entry.build()

    # Complete populate the ELF header.
    self.elf_header.phentsize = self.phentsize
    self.elf_header.phnum = self.phnum

    # Program Header offset hard-coded to end of the elf header.
    self.elf_header.phoff = self.phoff

    # Entry point to the program.
    self.elf_header.entry = self.instructionsvoff

    # Section Header offset.
    self.elf_header.shoff = self.instructionsoff + self.instructions_size + \
        self.instructionspaddingsize

    self.elf_header.shentsize = self.shentsize
    self.elf_header.shnum = self.shnum

    # This is hard-coded for now since we will only have limited sections for
    # now.
    # FIXME: Should be made dynamic when we start adding sections dynamically.
    self.elf_header.shstrndx = self.ELF_HEADER_SHSTRNDX if \
        self.global_memory_size else self.ELF_HEADER_SHSTRNDX - 1


    # Final binary dumping.
    self.binary = ''.join([
        str(self.elf_header.build()),
        str(self.shstrtab),
        str(self.strtab),
        str(self.null_sym_entry), str(self.symtab_start_entry),
        ''.join([str(e) for e in self.function_symtab_entries]),
        str(self.ph_load_header),
        str(self.ph_bss_header) if self.ph_bss_header else '',
        self.padding(self.phpaddingsize),
        str(self.linker.binary), self.padding(self.instructionspaddingsize),
        str(self.sh_null_header), str(self.sh_text_header),
        str(self.sh_bss_header) if self.sh_bss_header else '',
        str(self.sh_shstrtab_header), str(self.sh_strtab_header),
        str(self.sh_symtab_header),
        self.padding(self.shpaddingsize),
        ])

  def __str__(self):
    """Returns the final binary string.
    """
    return self.binary


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
