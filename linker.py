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
        'OutputNum': 0,
        'InputNum': 0
        }

    # Sums the offsets of the functions as they get added
    self.function_offset = 0

  def compute_offsets(self):
    """Computes the offsets for the functions once they are put together.
    """
    for function in self.functions:
      self.function_offset_map[function.ir.function_name] = \
          self.function_offset
      self.function_offset += len(function.binary)

  def build(self):
    """Build the binary string for the entire code.
    """
    self.binary = ''.join([f.binary for f in self.functions])
    for f in self.functions:
      print "Function name: ", f.ir.function_name
      print
      for i in f.instructions:
        for char in i.binary:
          print '%02x' % (ord(char),),
        print


  def link(self):
    """Links all the functions together by linking the calls.
    """
    self.compute_offsets()
    for function in self.functions:
      for instruction, target_function_name in function.calls_to_link:
        target_offset = self.function_offset_map[target_function_name]
        function_offset = self.function_offset_map[function.ir.function_name]
        next_offset = function.instruction_offsets_map[
            instruction]['end_offset']
        jump_offset = target_offset - (function_offset + next_offset)
        instruction.set_target(jump_offset)

    self.build()