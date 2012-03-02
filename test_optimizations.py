# Copyright 2012 Madhusudan C.S.
#
# This file test_optimizations.py is part of PL241-MCS compiler.
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

"""Contains the tests for the optimizations.
"""


import optimizations
import ir
import ssa


def test_cse_cp():
  # Example 1
  ir.Instruction.reset_counter()

  instructions = [
      ir.Instruction('mul', 'b_0', 'c_0'),
      ir.Instruction('add', 0, 'd_0'),
      ir.Instruction('move', 1, 'd_1'),
      ir.Instruction('mul', 'b_0', 'c_0'),
      ir.Instruction('mul', 3, 'd_1'),
      ir.Instruction('move', 4, 'd_2')
      ]

  # Create a dummy SSA object
  ssa_form = ssa.SSA([], None)
  ssa_form.ssa = instructions

  optimized = optimizations.Optimize(ssa_form)
  optimized.cse_cp()
  optimized.replace_phis()

  expected = [
      ('mul', 'b_0', 'c_0'),
      ('add', 0, 'd_0'),
      ('mul', 0, 1)
      ]
  for ssa_instruction, expected in zip(ssa_form.optimized(), expected):
    assert ssa_instruction.instruction == expected[0]
    assert ssa_instruction.operand1 == expected[1]
    assert ssa_instruction.operand2 == expected[2]

  print "Test Case 1: Passed!!!"

  # Example 2
  ir.Instruction.reset_counter()

  instructions = [
      ir.Instruction('load', '[ret]'),
      ir.Instruction('move', 0, 'b_2'),
      ir.Instruction('move', 'b_2', 'c_3'),
      ir.Instruction('add', 'b_2', 'c_3'),
      ir.Instruction('move', 3, 'd_5'),
      ir.Instruction('add', 0, 'b_2'),
      ir.Instruction('move', 5, 'e_7'),
      ir.Instruction('cmp', 0, '#0'),
      ir.Instruction('bge', 7, 14),
      ir.Instruction('add', 'd_5', 'e_7'),
      ir.Instruction('move', 9, 'd_11'),
      ir.Instruction('move', 'd_11', 'a_12'),
      ir.Instruction('bra', 15),
      ir.Instruction('move', 'e_7', 'd_14'),
      ir.Instruction('phi', 'd_15', 'd_11', 'd_5'),
      ir.Instruction('phi', 'a_16', 'a_12', 0),
      ir.Instruction('write', 'a_16'),
      ]

  # Create a dummy SSA object
  ssa_form = ssa.SSA([], None)
  ssa_form.ssa = instructions

  optimized = optimizations.Optimize(ssa_form)
  optimized.cse_cp()
  optimized.replace_phis()

  expected = [
      ('load', '[ret]', None),
      ('add', 0, 0),
      ('cmp', 0, '#0'),
      ('bge', 7, 14),
      ('add', 3, 3),
      ('bra', 15, None),
      ('phi', 'd_15', 9, 3),
      ('phi', 'a_16', 9, 0),
      ('write', 'a_16', None)
      ]
  for ssa_instruction, expected in zip(ssa_form.optimized(), expected):
    assert ssa_instruction.instruction == expected[0]
    assert ssa_instruction.operand1 == expected[1]
    assert ssa_instruction.operand2 == expected[2]
    for op, exp_op in zip(ssa_instruction.operands, expected[3:]):
      assert op == exp_op

  print "Test Case 2: Passed!!!"

  # Example 2
  ir.Instruction.reset_counter()

  instructions = [
      ir.Instruction('.begin_sum'),
      ir.Instruction('add', '!FP', '#0'),
      ir.Instruction('load', 1),
      ir.Instruction('move', 2, '[framesize]'),
      ir.Instruction('add', 1, '#4'),
      ir.Instruction('load', 4),
      ir.Instruction('move', 5, '[ret]'),
      ir.Instruction('add', 4, '#4'),
      ir.Instruction('move', 7, '[sum]'),
      ir.Instruction('add', 7, '#4'),
      ir.Instruction('load', 9),
      ir.Instruction('move', 10, 'sum/a_0'),
      ir.Instruction('add', 9, '#4'),
      ir.Instruction('load', 12),
      ir.Instruction('move', 13, 'sum/len_0'),
      ir.Instruction('move', '#0', 'sum/count_0'),
      ir.Instruction('move', '#0', 'sum/temp_0'),
      ir.Instruction('phi', 'sum/count_1', 'sum/count_0', 'sum/count_4'),
      ir.Instruction('phi', 'sum/temp_1', 'sum/temp_0', 'sum/temp_2'),
      ir.Instruction('cmp', 'sum/count_0', 'sum/len_0'),
      ir.Instruction('bge', 19, 38),
      ir.Instruction('mul', 'sum/count_1', '#4'),
      ir.Instruction('add', '!FP', '#sum/a'),
      ir.Instruction('adda', 21, 22),
      ir.Instruction('load', 23),
      ir.Instruction('add', 'sum/temp_1', 24),
      ir.Instruction('move', 25, 'sum/temp_2'),
      ir.Instruction('add', 'sum/count_1', '#1'),
      ir.Instruction('move', 27, 'sum/count_2'),
      ir.Instruction('cmp', 'sum/temp_2', 'sum/len_0'),
      ir.Instruction('bge', 29, 34),
      ir.Instruction('add', 'sum/count_4', '#1'),
      ir.Instruction('move', 31, 'sum/count_5'),
      ir.Instruction('bra', 37),
      ir.Instruction('sub', 'sum/count_2', '#1'),
      ir.Instruction('move', 34, 'sum/count_3'),
      ir.Instruction('phi', 'sum/count_4', 'sum/count_5', 'sum/count_3'),
      ir.Instruction('bra', 19),
      ir.Instruction('store', 'sum/temp_1', '[sum]'),
      ir.Instruction('bra', '[ret]'),
      ir.Instruction('.end_sum')
      ]

  # Create a dummy SSA object
  ssa_form = ssa.SSA([], None)
  ssa_form.ssa = instructions

  optimized = optimizations.Optimize(ssa_form)
  optimized.cse_cp()
  optimized.replace_phis()

  expected = [
      ('.begin_sum', None, None),
      ('add', '!FP', '#0'),
      ('load', 1, None),
      ('move', 2, '[framesize]'),
      ('add', 1, '#4'),
      ('load', 4, None),
      ('move', 5, '[ret]'),
      ('add', 4, '#4'),
      ('move', 7, '[sum]'),
      ('add', 7, '#4'),
      ('load', 9, None),
      ('add', 9, '#4'),
      ('load', 12, None),
      ('phi', 'sum/count_1', '#0', 'sum/count_4'),
      ('phi', 'sum/temp_1', '#0', 25),
      ('cmp', '#0', 13),
      ('bge', 19, 38),
      ('mul', 'sum/count_1', '#4'),
      ('add', '!FP', '#sum/a'),
      ('adda', 21, 22),
      ('load', 23, None),
      ('add', 'sum/temp_1', 24),
      ('add', 'sum/count_1', '#1'),
      ('cmp', 25, 13),
      ('bge', 29, 34),
      ('add', 'sum/count_4', '#1'),
      ('bra', 37, None),
      ('sub', 27, '#1'),
      ('phi', 'sum/count_4', 31, 34),
      ('bra', 19, None),
      ('store', 'sum/temp_1', '[sum]'),
      ('bra', '[ret]', None),
      ('.end_sum', None, None)
      ]

  for ssa_instruction, expected in zip(ssa_form.optimized(), expected):
    assert ssa_instruction.instruction == expected[0]
    assert ssa_instruction.operand1 == expected[1]
    assert ssa_instruction.operand2 == expected[2]
    for op, exp_op in zip(ssa_instruction.operands, expected[3:]):
      assert op == exp_op

  print "Test Case 3: Passed!!!"

if __name__ == '__main__':
  test_output = test_cse_cp()
