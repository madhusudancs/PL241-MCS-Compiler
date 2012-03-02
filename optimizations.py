# Copyright 2012 Madhusudan C.S. 
#
# This file optimizations.py is part of PL241-MCS compiler.
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

"""Algorithms to perform optimizations on the input source program.
"""


import logging

from argparse import ArgumentParser

from ir import Instruction
from ir import IntermediateRepresentation
from parser import LanguageSyntaxError
from parser import Parser
from ssa import SSA


# Module level logger object
LOGGER = logging.getLogger(__name__)


def cse_cp(ssa):
  """Performs common sub-expression elimination and copy propogation.

  Args:
    ssa: The SSA object.
  """
  # FIXME: Don't copy-propagate main function variables since they are global
  # variables!!! That is the most dangerous thing because any function can
  # use the global variables! So we cannot copy-propagate it in the function
  # since it changes for every function call whose exact value cannot be
  # determined statically (i.e. until runtime)

  # For each statement stores its replacement for either the statement
  # itself or the result of the statement if the statement was removed
  # because of Common Sub-expression elimination or Copy propogation.
  operand_replacements = {}

  instruction_replacements = {}

  for instruction in ssa.ssa:
    # Check if the operands exist in the replacements dictionary, if so
    # replace them with the value of the operand key in the replacements
    # dictionary.
    if instruction.operand1 in operand_replacements:
      instruction.operand1 = operand_replacements[instruction.operand1]
    if instruction.operand2 in operand_replacements:
      instruction.operand2 = operand_replacements[instruction.operand2]

    # Remove all move instructions by copy propagation and record
    # the replacement.
    if (instruction.instruction == 'move' and
        instruction.is_variable(instruction.operand2)):
      operand_replacements[instruction.operand2] = instruction.operand1

      ssa.optimized_removal[instruction.label] = True

      # Done with this instruction since we don't want to add it, move on
      # to next instruction.
      continue

    # This loop occurs only in the case of phi instructions since all other
    # instructions have exactly one or two operands.
    new_operands = []
    for operand in instruction.operands:
      if operand in operand_replacements:
        new_operands.append(operand_replacements[operand])
      else:
        new_operands.append(operand)

    instruction.operands = tuple(new_operands)

    # FIXME: This code may become buggy if two values (unrelated
    # instructions) compute to the same hash.
    if hash(instruction) in instruction_replacements:
      # This instruction should be a common sub-expression, record the
      # replacement for it and remove it (effectively do not add it to
      # new SSA list)
      instruction_replacements[instruction.label] = \
          instruction_replacements[hash(instruction)]
      ssa.optimized_removal[instruction.label] = True
    else:
      # We need this instruction, so copy to the new SSA and record it in the
      # replacements dictionary.
      instruction_replacements[hash(instruction)] = instruction.label

    # FIXME: This may not be required at all.
    # Additionally if an instruction is a phi instruction update update
    # the corresponding phi functions in the corresponding basic blocks
    if instruction.instruction == 'phi':
      original_variable = instruction.operand1.rsplit('_', 1)[0]
      node = ssa.label_nodes[instruction.label]
      phi_function = node.phi_functions[original_variable]
      if phi_function['LHS'] in operand_replacements:
        phi_function['LHS'] = operand_replacements[phi_function['LHS']]

      new_operands = []
      for operand in phi_function['RHS']:
        if operand in operand_replacements:
          new_operands.append(operand_replacements[operand])
        else:
          new_operands.append(operand)
      phi_function['RHS'] = new_operands

  return ssa


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-t', '--trace', metavar="Trace Mode", type=str,
                      nargs='?', const=True,
                      help='Enable trace mode for optimizations.')
  parser.add_argument('-g', '--vcg', metavar="VCG", type=str,
                      nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph output.')
  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  try:
    p = Parser(args.file_names[0])
    ir = IntermediateRepresentation(p)

    ir.generate()
    cfg = ir.build_cfg()
    cfg.compute_dominance_frontiers()

    ssa = SSA(ir, cfg)
    ssa.construct()

    ssa = cse_cp(ssa)

    if args.vcg:
      vcg_file = open(args.vcg, 'w') if isinstance(args.vcg, str) else \
          sys.stdout
      vcg_file.write(p.root.generate_vcg())
      vcg_file.close()

    return ssa

  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ssa = bootstrap()
