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
import regalloc
import subprocess


def test_sat_solver():
  """Tests if the boolean satisfiability is working as expected.
  """
  r1 = regalloc.Register()
  r2 = regalloc.Register()
  r3 = regalloc.Register()
  r4 = regalloc.Register()

  n1 = datastructures.InterferenceNode(r1, (10, 20))
  n2 = datastructures.InterferenceNode(r2, (15, 25))
  n3 = datastructures.InterferenceNode(r3, (18, 30))
  n4 = datastructures.InterferenceNode(r4, (28, 40))

  n1.append_edges(n2, n3)
  n2.append_edges(n3)

  ifg = datastructures.InterferenceGraph([n1, n2, n3, n4])
  ra = regalloc.RegisterAllocator(None, 4)
  is_solved, allocation = ra.sat_solve(ifg)
  assert is_solved
  print "Test Case 1 Passed! Allocation: %s" % (allocation)

  r5 = regalloc.Register()

  n1 = datastructures.InterferenceNode(r1, (10, 20))
  n2 = datastructures.InterferenceNode(r2, (15, 25))
  n3 = datastructures.InterferenceNode(r3, (18, 30))
  n4 = datastructures.InterferenceNode(r4, (19, 40))
  n5 = datastructures.InterferenceNode(r5, (14, 18))

  n1.append_edges(n2, n3, n4, n5)
  n2.append_edges(n3, n4, n5)
  n3.append_edges(n4)

  ifg = datastructures.InterferenceGraph([n1, n2, n3, n4])
  ra = regalloc.RegisterAllocator(None, 4)
  is_solved, allocation = ra.sat_solve(ifg)
  assert is_solved
  print 'Test Case 2 Passed! Allocation: %s' % (allocation)

  r6 = regalloc.Register()

  n1 = datastructures.InterferenceNode(r1, (10, 20))
  n2 = datastructures.InterferenceNode(r2, (15, 25))
  n3 = datastructures.InterferenceNode(r3, (18, 30))
  n4 = datastructures.InterferenceNode(r4, (19, 40))
  n5 = datastructures.InterferenceNode(r5, (14, 18))
  n6 = datastructures.InterferenceNode(r6, (18, 20))

  n1.append_edges(n2, n3, n4, n5, n6)
  n2.append_edges(n3, n4, n5, n6)
  n3.append_edges(n4, n6)
  n4.append_edges(n6)

  ifg = datastructures.InterferenceGraph([n1, n2, n3, n4])
  ra = regalloc.RegisterAllocator(None, 4)
  is_solved, allocation = ra.sat_solve(ifg)
  assert not is_solved
  print 'Test Case 3 Passed! Allocation failed'


if __name__ == '__main__':
  test_output = test_sat_solver()
