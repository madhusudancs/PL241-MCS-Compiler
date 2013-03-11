# Copyright 2013 Madhusudan C.S.
#
# This file ipra.py is part of PL241-MCS compiler.
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

"""Perform interprocedural register allocation trying to optimize the
registers used.
"""


from regalloc import RegisterAllocator
from x86_64 import FUNCTION_ARGUMENTS_COLORS
from x86_64 import REGISTERS_COLOR_SET


# We use one register as a scratch register which 13 which is not used for
# register allocation and in x86 %rbp, %rsp and %rip are used for specific
# purporses so they cannot be used for register allocation either.
AVAIL_REGS = REGISTERS_COLOR_SET - set([13])


class IPRA(object):
  """Interprocedural register allocator.
  """

  def __init__(self, compiling_functions, call_graph):
    """Initializes the interprocedural register allocator.
    """
    self.compiling_functions = compiling_functions
    self.call_graph = call_graph

    # Map of allocated function names, to used_registers to make sure
    # functions are not register allocated more than once.
    self.function_used_reg_map = {}

  def allocate(self):
    """Begin register allocation on 
    """
    self.dfs(self.call_graph['main'])

  def dfs(self, node):
    """Perform depth first search on the given subtree's root node.
    """
    if node.value in self.function_used_reg_map:
      return self.function_used_reg_map[node.value]

    used_regs = set()
    func_name = node.value
    if func_name in ['InputNum', 'OutputNum', 'OutputNewLine']:
      return set(FUNCTION_ARGUMENTS_COLORS)

    if node.children:
      for child in node.children:
          used_regs |= self.dfs(child)

      regalloc = RegisterAllocator(
          self.compiling_functions[func_name]['ssa'],
          max(len(AVAIL_REGS) - len(used_regs), 8))

    else:
      regalloc = RegisterAllocator(
          self.compiling_functions[func_name]['ssa'], len(AVAIL_REGS))

    regalloc.allocate()
    self.compiling_functions[func_name]['regalloc'] = regalloc

    updated_used_regs = self.remap_registers(
        used_regs, regalloc.function_parameters,
        set(regalloc.used_physical_registers.keys()))

    self.compiling_functions[func_name]['used_physical_registers'] = \
        updated_used_regs

    self.function_used_reg_map[node.value] = updated_used_regs

    return updated_used_regs

  def remap_registers(self, used_child_regs, function_parameters, used_regs):
    """Remaps the assigned register colors to the lowest valued colors.
    """
    used_func_arg_regs = set(
        [regnum for regnum, param in zip(
            FUNCTION_ARGUMENTS_COLORS, function_parameters)])

    used_regs |= used_child_regs | used_func_arg_regs

    return used_regs
