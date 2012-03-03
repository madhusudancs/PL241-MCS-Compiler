# Copyright 2012 Madhusudan C.S.
#
# This file test_regalloc.py is part of PL241-MCS compiler.
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

"""Contains the tests for the register allocation.
"""

import math
import datastructures
import subprocess


def test_bool_sat():
  """Tests if the boolean satisfiability is working as expected.
  """
  n1 = datastructures.InterferenceNode('r1', (10, 20))
  n2 = datastructures.InterferenceNode('r2', (15, 25))
  n3 = datastructures.InterferenceNode('r3', (18, 30))
  n4 = datastructures.InterferenceNode('r4', (28, 40))

  n1.append_edges(n2, n3)
  n2.append_edges(n3)

  registers = 2

  ifg = datastructures.InterferenceGraph([n1, n2, n3, n4])

  # Dictionary holding the 
  processed = {}

  conflicting_registers = {}

  last_pattern = bin(registers - 1)[2:]
  last_pattern_length = len(last_pattern)

  bit_template = [last_pattern]
  for i in range(registers - 2, -1, -1):
    pattern = ('%s' % bin(i)[2:]).zfill(last_pattern_length)
    bit_template.append(pattern)

  clauses = []

  for node in ifg:
    node_reg = node.register[1:]
    if node.edges:
      conflicting_registers[node_reg] = True

    for edge in node.edges:
      edge_reg = edge.register[1:]
      if ((edge_reg, node_reg) in processed) or (
          (node_reg, edge_reg) in processed):
        continue

      processed[(node_reg, edge_reg)] = True

      conflicting_registers[edge_reg] = True

      for template in bit_template:
        clause = ''
        for bit_position, bit in enumerate(template):
          if bit == '0':
            clause += ('-%(node_reg)s%(bit_position)d '
                '-%(edge_reg)s%(bit_position)d ') % {
                    'node_reg': node_reg,
                    'edge_reg': edge_reg,
                    'bit_position': bit_position,
                }
          else:
            clause += ('%(node_reg)s%(bit_position)d '
                '%(edge_reg)s%(bit_position)d ') % {
                    'node_reg': node_reg,
                    'edge_reg': edge_reg,
                    'bit_position': bit_position,
                    }
        clauses.append(clause)

  cnf = 'p cnf %d %d\n%s0' % (
       len(conflicting_registers) * math.ceil(math.log(registers)),
       len(clauses),
       '0\n'.join(clauses))

  process = subprocess.Popen('glucose_static', stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE)
  output = process.communicate(cnf)
  # Read last 4 lines
  lines = output.rsplit('\n', 4)
  if lines[0] == 's UNSATISFIABLE':
    print "Not Satisfiable"
  elif lines[1] == 's SATISFIABLE':
    print "Allocation: %s"

if __name__ == '__main__':
  test_output = test_bool_sat()
