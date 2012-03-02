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
import sys

from argparse import ArgumentParser

from ir import Instruction
from ir import IntermediateRepresentation
from parser import LanguageSyntaxError
from parser import Parser
from ssa import SSA


# Module level logger object
LOGGER = logging.getLogger(__name__)


class Optimize(object):
  """Implements different types of optimizations for the compiler
  """

  def __init__(self, ssa):
    """Initializes the optimizations class.

    Args:
      ssa: The ssa object on which optimizations must be performed
    """
    self.ssa = ssa

  def replace_phi_functions(self):
    """Replace all phi functions
    """
    for node in set(self.phi_nodes):
      for phi_function in node.phi_functions.values():
        new_operands = []
        for operand in phi_function['RHS']:
          if operand in self.operand_replacements:
            new_operands.append(self.operand_replacements[operand])
          else:
            new_operands.append(operand)
        phi_function['RHS'] = new_operands

  def replace_phis(self):
    """Replace the operands of the phi instructions with the optimized code.

    This also replaces the phi functions in the nodes.
    ALERT! Don't touch the LHS of the phi functions and hence the first
    operand of the phi instructions! It doesn't make sense to replace them!
    """
    self.phi_nodes = []
    for label in self.phi_instructions:
      instruction = self.ssa.ssa[label]
      # We should not touch the first operand of the phi function since it
      # is the result of the phi function. We only start replacing from the
      # second operand which are the RHS of the phi-functions.
      if instruction.operand2 in self.operand_replacements:
        instruction.operand2 = self.operand_replacements[instruction.operand2]

      new_operands = []
      for operand in instruction.operands:
        if operand in self.operand_replacements:
          new_operands.append(self.operand_replacements[operand])
        else:
          new_operands.append(operand)

      instruction.operands = tuple(new_operands)

      original_variable = instruction.operand1.rsplit('_', 1)[0]

      self.phi_nodes.append(self.ssa.label_nodes.get(label))

  def cse_cp(self):
    """Performs common sub-expression elimination and copy propogation.
    """
    # FIXME: Don't copy-propagate main function variables since they are global
    # variables!!! That is the most dangerous thing because any function can
    # use the global variables! So we cannot copy-propagate it in the function
    # since it changes for every function call whose exact value cannot be
    # determined statically (i.e. until runtime)

    # For each statement stores its replacement for either the statement
    # itself or the result of the statement if the statement was removed
    # because of Common Sub-expression elimination or Copy propogation.
    self.operand_replacements = {}

    self.instruction_replacements = {}

    self.phi_instructions = []

    for instruction in self.ssa.ssa:
      # We will process all the phi functions in the end because in some cases
      # like loops, the phi operands appear before they are actually defined.
      # We need to take care of those too, so let us process them separately.
      if instruction.instruction == 'phi':
        self.phi_instructions.append(instruction.label)
        continue

      # If end of function is reached reset the instruction_replacements
      # since we cannot optimize it anyway.
      # Also we need not reset operand_replacements because they don't
      # affect each other anyway because of the scoped-representation of
      # the symbols we have chosen and instruction labels are different
      # anyway.
      # FIXME: Get rid of this when functions are compiled independently
      if instruction.instruction.startswith('.end_'):
        self.instruction_replacements = {}

      # Check if the operands exist in the replacements dictionary, if so
      # replace them with the value of the operand key in the replacements
      # dictionary.
      if instruction.operand1 in self.operand_replacements:
        instruction.operand1 = self.operand_replacements[instruction.operand1]
      if instruction.operand2 in self.operand_replacements:
        instruction.operand2 = self.operand_replacements[instruction.operand2]

      new_operands = []
      for operand in instruction.operands:
        if operand in self.operand_replacements:
          new_operands.append(self.operand_replacements[operand])
        else:
          new_operands.append(operand)

      instruction.operands = tuple(new_operands)

      # Remove all move instructions by copy propagation and record
      # the replacement.
      if (instruction.instruction == 'move' and
          instruction.is_variable(instruction.operand2)):
        self.operand_replacements[instruction.operand2] = instruction.operand1

        self.ssa.optimized_removal[instruction.label] = True

        # Done with this instruction since we don't want to add it, move on
        # to next instruction.
        continue

      # FIXME: This code may become buggy if two values (unrelated
      # instructions) compute to the same hash.
      if hash(instruction) in self.instruction_replacements:
        # This instruction should be a common sub-expression, record the
        # replacement for it and remove it (effectively do not add it to
        # new SSA list)
        self.operand_replacements[instruction.label] = \
            self.instruction_replacements[hash(instruction)]
        self.ssa.optimized_removal[instruction.label] = True
      else:
        # We need this instruction, so copy to the new SSA and record it in the
        # replacements dictionary.
        self.instruction_replacements[hash(instruction)] = instruction.label

  def optimize(self):
    """Bootstraps the whole optimization process
    """
    self.cse_cp()
    self.replace_phis()
    self.replace_phi_functions()


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-t', '--trace', metavar="Trace Mode", type=str,
                      nargs='?', const=True,
                      help='Enable trace mode for optimizations.')
  parser.add_argument('-o', '--optimized', metavar="Optimized", type=str,
                      nargs='?', const=True,
                      help='Generates the optimized output.')
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

    optimize = Optimize(ssa)
    optimize.optimize()

    if args.optimized:
      external_file = isinstance(args.optimized, str)
      optimized_file = open(args.optimized, 'w') if external_file \
          else sys.stdout
      optimized_file.write('\n'.join([str(s) for s in ssa.optimized()]))
      if external_file:
        optimized_file.close()

    if args.vcg:
      vcg_file = open(args.vcg, 'w') if isinstance(args.vcg, str) else \
          sys.stdout
      vcg_file.write(ssa.ssa_cfg.generate_vcg())
      vcg_file.close()

    return ssa

  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ssa = bootstrap()
