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
import logging
import sys

from argparse import ArgumentParser

from datastructures import CFG
from datastructures import CFGNode
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

  # Anything that begins with this is not a valid variable in the source
  # program and hence can be ignored during SSA construction.
  NON_VARIABLE_OPERANDS_STARTSWITH = {
      '#': True,
      '.': True,
      '!': True,
      '[': True,
      }

  def __init__(self, ir, cfg):
    """Intializes the datastructures required for SSA construction.
    """
    self.ir = ir
    self.cfg = cfg

    # Stores the SSA form the IR as a list.
    self.ssa = []

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

  def populate_labels(self):
    """Populates the begin labels for each CFG Node.
    """
    for node in self.cfg:
      for label in range(node.value[0], node.value[1] + 1):
        self.label_nodes[label] = node

  def identify_assignment_and_usage_nodes(self):
    """Linearly scans the IR to determine assignment and usage nodes in CFG.
    """
    for i in self.ir:
      if i.operand1 and isinstance(i.operand1, str) and (
            i.operand1[0] not in self.NON_VARIABLE_OPERANDS_STARTSWITH):
        node = self.label_nodes[i.label]
        node.mentions[i.operand1] = True
        self.variable_mentions[i.operand1].append(node)

      if i.operand2 and isinstance(i.operand2, str) and (
          i.operand2[0] not in self.NON_VARIABLE_OPERANDS_STARTSWITH):
        node = self.label_nodes[i.label]
        if i.instruction == 'move':
          node.assignments[i.operand2] = True
          self.variable_assignments[i.operand2].append(node)
        else:
          node.mentions[i.operand2] = True
          self.variable_mentions[i.operand2].append(node)

  def construct(self):
    """Constructs the SSA form of the IR.
    """
    self.populate_labels()
    self.identify_assignment_and_usage_nodes()

  def __str__(self):
    """Prints the SSA stored for the program
    """
    pass


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
      vcg_file.write(cfg.generate_vcg(ir=ir))
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

    return ssa
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  ssa = bootstrap()
