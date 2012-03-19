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

  def cse_cp(self, root, operand_replacements,
                         instruction_replacements):
    """Performs the common sub-expression elimination and copy propogation.

    Args:
      root: The root of the dominator subtree.
      operand_replacements: The operand replacements passed on by the parent.
      instruction_replacements: The instruction replacements passed on by the
          parent.
    """
    block_operand_replacements = {}
    block_instruction_replacements = {}

    for instruction in self.ssa.ir.ir[root.value[0]:root.value[1] + 1]:
      # Check if the operands exist in the replacements dictionary, if so
      # replace them with the value of the operand key in the replacements
      # dictionary.
      if instruction.operand1 in block_operand_replacements:
        instruction.operand1 = block_operand_replacements[instruction.operand1]
      elif instruction.operand1 in operand_replacements:
        instruction.operand1 = operand_replacements[instruction.operand1]

      if instruction.operand2 in block_operand_replacements:
        instruction.operand2 = block_operand_replacements[instruction.operand2]
      elif instruction.operand2 in operand_replacements:
        instruction.operand2 = operand_replacements[instruction.operand2]


      new_operands = []
      for operand in instruction.operands:
        if operand in block_operand_replacements:
          new_operands.append(block_operand_replacements[operand])
        elif operand in operand_replacements:
          new_operands.append(operand_replacements[operand])
        else:
          new_operands.append(operand)

      instruction.operands = tuple(new_operands)

      # Leave the prologue and epilogue instructions alone
      if (instruction.instruction == '.begin_' or
          instruction.instruction == '.end_'):
        continue

      # Remove all move instructions by copy propagation and record
      # the replacement.
      if (instruction.instruction == 'move' and
          instruction.is_variable(instruction.operand2)):
        block_operand_replacements[instruction.operand2] = instruction.operand1

        self.ssa.optimized_removal[instruction.label] = True

        # Done with this instruction since we don't want to add it, move on
        # to next instruction.
        continue

      # FIXME: This code may become buggy if two values (unrelated
      # instructions) compute to the same hash.
      if hash(instruction) in block_instruction_replacements:
        # This instruction should be a common sub-expression, record the
        # replacement for it and remove it (effectively do not add it to
        # new SSA list)
        block_operand_replacements[instruction.label] = \
            block_instruction_replacements[hash(instruction)]
        self.ssa.optimized_removal[instruction.label] = True
      elif hash(instruction) in instruction_replacements:
        # This instruction should be a common sub-expression, record the
        # replacement for it and remove it (effectively do not add it to
        # new SSA list)
        block_operand_replacements[instruction.label] = \
            instruction_replacements[hash(instruction)]
        self.ssa.optimized_removal[instruction.label] = True        
      else:
        # We need this instruction, so copy to the new SSA and record it in the
        # replacements dictionary.
        block_instruction_replacements[hash(instruction)] = instruction.label


    # ALERT IMPORTANT EVERYTHING: It will be disastrous if you don't create
    # a new dictionary here since Python updates the same object
    merged_operand_replacements = {}
    merged_operand_replacements.update(operand_replacements)
    merged_operand_replacements.update(block_operand_replacements)
    merged_instruction_replacements = {}
    merged_instruction_replacements.update(instruction_replacements)
    merged_instruction_replacements.update(block_instruction_replacements)

    children_operand_replacements = {}
    for child in root.dom_children:
      op_repl, ins_repl = self.cse_cp(child, merged_operand_replacements, 
                  merged_instruction_replacements)
      children_operand_replacements.update(op_repl)

    # We will process all the phi functions in the end because in some cases
    # like loops, the phi operands appear before they are actually defined.
    # We need to take care of those too, so let us process them separately.
    for phi_function in root.phi_functions.values():
      for i, operand in enumerate(phi_function['RHS']):
        if operand in block_operand_replacements:
          phi_function['RHS'][i] = block_operand_replacements[operand]
        elif operand in operand_replacements:
          phi_function['RHS'][i] = operand_replacements[operand]
        elif operand in children_operand_replacements:
          phi_function['RHS'][i] = children_operand_replacements[operand]

    return merged_operand_replacements, merged_instruction_replacements

  def optimize(self):
    """Bootstraps the whole optimization process
    """
    self.cse_cp(self.ssa.cfg[0], {}, {})

  def __str__(self):
    """Prints the SSA stored for the program
    """
    bfs_queue = [self.ssa.cfg[0]]
    visited = set([])
    ssa_blocks = []

    start_labels_to_blocks = {}

    while bfs_queue:
      ssa = ''
      node = bfs_queue.pop(0)
      if node in visited:
        continue

      visited.add(node)
      bfs_queue.extend(node.out_edges[::-1])

      for phi_function in node.phi_functions.values():
        ssa += '%4s: %5s' % ('', 'phi')

        ssa += '%50s' % phi_function['LHS']
        for operand in phi_function['RHS']:
          ssa += '%50s' % operand

        ssa += '\n'

      start_labels_to_blocks[len(ssa_blocks)] = node.value[0]
      for instruction in self.ssa.optimized(node.value[0], node.value[1] + 1):
        ssa += '%s\n' % (instruction)

      ssa_blocks.append(ssa)

    # Sort the basic blocks according to their start instruction label
    sorted_blocks = sorted(
        enumerate(ssa_blocks), key=lambda k: start_labels_to_blocks[k[0]])

    # Ditch the last block since that is a repeatition of the end instruction.
    return '\n'.join([b[1] for b in sorted_blocks])


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
