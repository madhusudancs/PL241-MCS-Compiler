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

  def allocate(self):
    """Begin register allocation on 
    """
    self.dfs(self.call_graph['main'])

  def dfs(self, node):
    """Perform depth first search on the given subtree's root node.
    """
    used_regs = set()
    func_name = node.value
    if func_name in ['InputNum', 'OutputNum', 'OutputNewline']:
      return used_regs

    for child in node.children:
        used_regs |= self.dfs(child)

    regalloc = RegisterAllocator(
        self.compiling_functions[func_name]['ssa'],
        len(AVAIL_REGS) - len(used_regs))

    regalloc.allocate()
    self.compiling_functions[func_name]['regalloc'] = regalloc
    
    used_regs = self.remap_registers(
        used_regs, regalloc.used_physical_registers,
        regalloc.function_parameters)

    return used_regs

  def remap_registers(self, used_child_regs, assigned_regs, function_parameters):
    """Remaps the assigned register colors to the lowest valued colors.
    """
    used_regs = used_child_regs

    assigned_reg_colors = assigned_regs.keys()
    for color in assigned_reg_colors:
      if color in AVAIL_REGS:
        AVAIL_REGS.remove(color)
        used_regs.add(color)
        assigned_regs.pop(color)

    for color in assigned_regs:
      new_color = AVAIL_REGS.pop()
      assigned_regs[color].color = new_color
      used_regs.add(color)

    return used_regs

