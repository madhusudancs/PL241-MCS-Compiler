# Copyright 2012 Madhusudan C.S. 
#
# This file ssa.py is part of PL241-MCS compiler.
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

"""Transforms the Intermediate Representation (IR) to Static Single Assignment
(SSA) form for the PL241 compiler.
"""


import collections
import copy
import logging
import sys

from argparse import ArgumentParser

from datastructures import CFG
from datastructures import CFGNode
from datastructures import Stack
from ir import Instruction
from ir import IntermediateRepresentation
from parser import LanguageSyntaxError
from parser import Parser


# Module level logger object
LOGGER = logging.getLogger(__name__)


class SSA(object):
  """Constructs the Static Single Assignment form of the Intermediate
  Representation of the source program.

  This class implements the minimal SSA construction algorithm based on
  the paper by Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman,
  and Z. Kenneth Zadeck in their landmark paper on minimal-SSA construction,
  "Efficiently Computing Static Single Assignment Form and the Control
  Dependence Graph" available at:

  http://dl.acm.org/citation.cfm?doid=115372.115320
  """

  def __init__(self, ir, cfg):
    """Intializes the datastructures required for SSA construction.
    """
    self.ir = ir
    self.cfg = cfg

    # Like an inverted index, for every label as key its value is the
    # CFG node it corresponds to.
    # Note this can be constructed in linear time in the size of the number
    # of instructions in the IR and this is not a problem because we MUST
    # do a linear scan to determine the assignment and usage nodes. The
    # time we spend here in building this map will be of a great help later
    # since we can check the CFG node any instruction belongs to in constant
    # time using this dictionary.
    self.label_nodes = {}

    # Again like an inverted index, for every variable, stores the nodes
    # where an assignment happens.
    self.variable_assignments = collections.defaultdict(list)

    # Again like an inverted index, for every variable, stores the nodes
    # where it is mentioned.
    self.variable_mentions = collections.defaultdict(list)

    # Contains a mapping of labels from IR to SSA
    self.labels_ir_to_ssa = {}

    # Dictionary of SSA instruction labels that were removed during
    # optimization. Should generate the machine code by not generating
    # the code for these instructions.
    self.optimized_removal = {}

  def populate_labels(self):
    """Populates the begin labels for each CFG Node.
    """
    for node in self.cfg:
      for label in range(node.value[0], node.value[1] + 1):
        self.label_nodes[label] = node

  def identify_assignment_and_usage_nodes(self):
    """Linearly scans the IR to determine assignment and usage nodes in CFG.
    """
    for i in self.ir.ir:
      if i.is_variable(i.operand1):
        node = self.label_nodes[i.label]
        node.mentions[i.operand1] = True
        self.variable_mentions[i.operand1].append(node)

      if i.is_variable(i.operand2):
        node = self.label_nodes[i.label]
        if i.instruction == 'move':
          node.assignments[i.operand2] = True
          self.variable_assignments[i.operand2].append(node)
        else:
          node.mentions[i.operand2] = True
          self.variable_mentions[i.operand2].append(node)

      for op in i.operands:
        if i.is_variable(op):
          node = self.label_nodes[i.label]
          node.mentions[op] = True
          self.variable_mentions[op].append(node)

  def place_phi(self):
    """Places the phi functions for the nodes in the control flow graph.
    """
    iter_count = 0
    has_already = collections.defaultdict(lambda: 0)
    work = collections.defaultdict(lambda: 0)

    current_work_list = []
    for variable, assignment_nodes in self.variable_assignments.items():
      iter_count += 1
      for node in assignment_nodes:
        work[node] = iter_count
        current_work_list.append(node)

      while current_work_list:
        node = current_work_list.pop(0)
        for frontier_node in node.dominance_frontier:
          if has_already[frontier_node] < iter_count:
            # We are just placing the phi functions here.
            # We will fill up the values, when renaming the
            # variables.
            frontier_node.phi_functions[variable] = {
                'LHS': None,
                'RHS': [0] * len(frontier_node.in_edges),
                }
            has_already[frontier_node] = iter_count
            if work[frontier_node] < iter_count:
              work[frontier_node] = iter_count
              current_work_list.append(frontier_node)

  def search(self, root, stacks, count):
    """Performs a top-down search of dominator tree for renaming variables.

    Args:
      stacks: contains a stack for every variable.
      count: contains a count of the number of assignments to a variable.
    """
    for variable in root.phi_functions:
      i = count[variable]
      root.phi_functions[variable]['LHS'] = '%s_%d' % (variable, i)
      stacks[variable].push(i)
      count[variable] = i + 1

    for label in range(root.value[0], root.value[1] + 1):
      instruction = self.ir.ir[label]
      if instruction.instruction == 'move':
        variable = instruction.operand1
        if instruction.is_variable(variable):
          i = stacks[variable].top()
          instruction.operand1 = '%s_%d' % (variable, i)

        variable = instruction.operand2
        if instruction.is_variable(variable):
          i = count[variable]
          instruction.operand2 = '%s_%d' % (variable, i)
          stacks[variable].push(i)
          count[variable] = i + 1

          # This is transient, but we want this for later use in renaming
          # so store it as a variable although not explicitly mentioned
          # in the class.
          instruction.old_operand2 = variable
      else:
        # For the prologue make dummy assignments or pretend to make
        # assignments so that phi function results won't get the same
        # subscript as the formal parameter subscripts.
        if instruction.instruction == '.begin_':
          operands = []
          for op in instruction.operands:
            if instruction.is_variable(op):
              i = count[op]
              stacks[op].push(i)
              count[op] = i + 1

        # After this continue normally for all instructions

        variable1 = instruction.operand1
        if instruction.is_variable(variable1):
          i = stacks[variable1].top()
          instruction.operand1 = '%s_%d' % (variable1, i)

        variable2 = instruction.operand2
        if instruction.is_variable(variable2):
          i = stacks[variable2].top()
          instruction.operand2 = '%s_%d' % (variable2, i)

        operands = []
        for op in instruction.operands:
          if instruction.is_variable(op):
            i = stacks[op].top()
            operands.append('%s_%d' % (op, i))
          else:
            operands.append(op)

        instruction.operands = operands

    for successor in root.out_edges:
      # My implementation of WhichPred(Y, X)
      j = successor.in_edges.index(root)
      for variable in successor.phi_functions:
        i = stacks[variable].top()
        successor.phi_functions[variable]['RHS'][j] = '%s_%d' % (variable, i)

    for child in root.dom_children:
      self.search(child, stacks, count)

    for label in range(root.value[0], root.value[1] + 1):
      instruction = self.ir.ir[label]
      if instruction.instruction == 'move' and hasattr(
          instruction, 'old_operand2'):
        stacks[instruction.old_operand2].pop()
      for phi_variable in root.phi_functions:
        stacks[phi_variable].pop()

  def rename(self):
    """Rename all the variables for SSA representation.
    """
    for tree in self.cfg.dom_trees:
      # Dictionary containing all the variables as the keys and stack
      # for each variable as the value.
      stacks = collections.defaultdict(lambda: Stack([0]))

      # Dictionary containing all the variables as the keys and the count
      # of how many times they are assigned.
      count = collections.defaultdict(lambda: 0)

      self.search(tree, stacks, count)

  def construct(self):
    """Constructs the SSA form of the IR.
    """
    self.populate_labels()
    self.identify_assignment_and_usage_nodes()

    self.place_phi()
    self.rename()

  def optimized(self, start=None, stop=None, reversed=False):
    """Yields the optimized SSA form of the IR as an iterator.
    """
    i = start if start else 0
    stop = stop if stop else len(self.ir.ir)
    if reversed:
      while i > stop:
        i -= 1
        if i + 1 in self.optimized_removal:
          continue

        yield self.ir.ir[i + 1]
    else:
      while i < stop:
        i += 1
        if i - 1 in self.optimized_removal:
          continue

        yield self.ir.ir[i - 1]

  def __str__(self):
    """Prints the SSA stored for the program
    """
    bfs_queue = [self.cfg[0]]
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
      for instruction in self.ir.ir[node.value[0]:node.value[1] + 1]:
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
  parser.add_argument('-g', '--vcg', metavar="VCG", type=str,
                      nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph output.')
  parser.add_argument('-r', '--ir', metavar="IR", type=str,
                      nargs='?', const=True,
                      help='Generate the Intermediate Representation.')
  parser.add_argument('-s', '--ssa', metavar="SSA", type=str,
                      nargs='?', const=True,
                      help='Generate the Static Single Assignment.')
  parser.add_argument('-t', '--dom', metavar="DominatorTree", type=str,
                      nargs='?', const=True,
                      help='Generate the Dominator Tree VCG output.')

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

    if args.vcg:
      external_file = isinstance(args.vcg, str)
      vcg_file = open(args.vcg, 'w') if external_file else \
          sys.stdout
      vcg_file.write(ssa.ssa_cfg.generate_vcg(ir=ssa.ssa))
      if external_file:
        vcg_file.close()

    if args.ir:
      external_file = isinstance(args.ir, str)
      ir_file = open(args.ir, 'w') if external_file else \
          sys.stdout
      ir_file.write(str(ir))
      if external_file:
        ir_file.close()

    if args.dom:
      external_file = isinstance(args.dom, str)
      dom_file = open(args.dom, 'w') if external_file else \
          sys.stdout
      dom_file.write(str(cfg.generate_dom_vcg()))
      if external_file:
        dom_file.close()

    if args.ssa:
      external_file = isinstance(args.ssa, str)
      ssa_file = open(args.ssa, 'w') if external_file else \
          sys.stdout
      ssa_file.write(str(ssa))
      if external_file:
        ssa_file.close()

    return ssa
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ssa = bootstrap()
