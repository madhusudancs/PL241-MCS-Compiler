# Copyright 2012 Madhusudan C.S.
#
# This file linker.py is part of PL241-MCS compiler.
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

"""This file contains the code that links together all the functions compiled.
"""


from precompiled import entry


class Linker(object):
  """Links all functions together and supplies the symbols for symbol table.
  """

  def __init__(self, functions):
    """Constructs the linker for the linking functions.

    Args:
      functions: list of all the function's code generator objects.
    """
    self.functions = functions

    # A dictionary containing a mapping from the function name to the function
    # start byte offset.
    self.function_offset_map = {
        # Hard code them to 0 for now.
        'OutputNum': {
            'offset': 0,
            'size': 0
            },
        'InputNum': {
            'offset': 0,
            'size': 0
            },
        'OutputNewLine': {
            'offset': 0,
            'size': 0
            },
        }

    # Sums the offsets of the functions as they get added
    self.function_offset = 0

  def compute_offsets(self):
    """Computes the offsets for the functions once they are put together.
    """
    for function in self.functions:
      size = len(function.binary)
      self.function_offset_map[function.ir.function_name] = {
          'offset': self.function_offset,
          'size': size
          }
      self.function_offset += size

  def build(self):
    """Build the binary string for the entire code.
    """
    self.binary = ''.join(
        [self.start_call.binary] + [f.binary for f in self.functions])

  def link(self):
    """Links all the functions together by linking the calls.
    """
    self.start_call = entry()
    start_size = len(self.start_call.binary)
    self.function_offset += start_size

    self.compute_offsets()

    next_offset = start_size
    main_offset = self.function_offset_map['main']['offset']
    jump_offset = main_offset - next_offset
    self.start_call.set_target(jump_offset)

    for function in self.functions:
      for instruction, target_function_name in function.calls_to_link:
        target_offset = self.function_offset_map[
            target_function_name]['offset']
        function_offset = self.function_offset_map[
            function.ir.function_name]['offset']
        next_offset = function.instruction_offsets_map[
            instruction]['end_offset']
        jump_offset = target_offset - (function_offset + next_offset)
        instruction.set_target(jump_offset)

    self.build()

  def __str__(self):
    """
    """
    linked_binary = '\n'
    for f in self.functions:
      linked_binary += "\nFunction name: %s\n" % (f.ir.function_name)
      for i in f.instructions:
        for char in i.binary:
          linked_binary += '%02x ' % (ord(char))
        linked_binary += '%30s\n' % i.__class__.__name__

    return linked_binary
