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

  new_ssa = optimizations.cse_cp(ssa_form)

  expected = [
      ('mul', 'b_0', 'c_0'),
      ('add', 0, 'd_0'),
      ('mul', 0, 1),
      ('move', 4, 'd_2')
      ]
  for ssa_instruction, expected in zip(new_ssa.optimized(), expected):
    assert ssa_instruction.instruction == expected[0]
    assert ssa_instruction.operand1 == expected[1]
    assert ssa_instruction.operand2 == expected[2]


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

  new_ssa = optimizations.cse_cp(ssa_form)

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
  for ssa_instruction, expected in zip(new_ssa.optimized(), expected):
    assert ssa_instruction.instruction == expected[0]
    assert ssa_instruction.operand1 == expected[1]
    assert ssa_instruction.operand2 == expected[2]
    for op, exp_op in zip(ssa_instruction.operands, expected[3:]):
      assert op == exp_op


if __name__ == '__main__':
  test_cse_cp()
