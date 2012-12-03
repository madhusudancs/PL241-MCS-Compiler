# Copyright 2012 Madhusudan C.S.
#
# This file callgraph.py is part of PL241-MCS compiler.
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

"""Perform a call graph analysis on the functions.
"""


from datastructures import CallGraph
from datastructures import Node


class CallGraphAnalysis(object):
  """Peform call graph analysis on the given functions.
  """

  def __init__(self, compiling_functions):
    """Initializes the call graph analysis.

    Args:
      compiling_functions: A mapping containing the function name and the
          various representations of the function as it is getting compiled.
    """
    self.compiling_functions = compiling_functions
    self.call_graph = CallGraph()

  def analyze(self):
    """Analyze the given functions to build the call graph.
    """
    for name, stages in self.compiling_functions.iteritems():
      call_sites = self.find_callsites(stages['ssa'])
      self.extend_call_graph(name, call_sites)

  def find_callsites(self, ssa):
    """Find the call sites in the given SSA representation of the function.

    Args:
      ssa: The SSA representation of the program.
    """
    call_sites = set()
    for instruction in ssa.optimized():
      if instruction.instruction == 'call':
        call_sites.add(instruction.function_name)

    return call_sites

  def extend_call_graph(self, name, call_sites):
    """Extend the call graph for the given function and its callsites.

    Args:
      name: The name of the function.
      call_sites: The call_sites in the function.
    """
    node = self.call_graph.get(name, None)
    if not node:
      node = Node(value=name)
      self.call_graph[name] = node

    for cs_name in call_sites:
      cs_node = self.call_graph.get(cs_name, None)
      if not cs_node:
        cs_node = Node(value=name)
        self.call_graph[cs_name] = cs_node

      if cs_node not in node.children:
        node.append_children(cs_node)
