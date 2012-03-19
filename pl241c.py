#! /usr/bin/python

# Copyright 2012 Madhusudan C.S.
#
# This file pl241c.py is part of PL241-MCS compiler.
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
# 

"""The main compiler file that bootstraps the whole compiler.
"""


import argparse
import os
import sys

from codegen import CodeGenerator
from ir import IntermediateRepresentation
from optimizations import Optimize
from parser import GLOBAL_SCOPE_NAME
from parser import LanguageSyntaxError
from parser import Parser
from ssa import SSA
from regalloc import RegisterAllocator


def bootstrap():
  parser = argparse.ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-a', '--dumpall', action='store_true',
                      help='Dump all the intermediate files and graphs'
                      'generated.')
  parser.add_argument('--cfg', action='store_true',
                      help='Generate the Visualization Compiler Graph output '
                      'of the Control Flow Graph.')
  parser.add_argument('--interferencevcg', action='store_true',
                      help='Generate the Visualization Compiler Graph '
                           'of the interference graph.')
  parser.add_argument('--optimized', action='store_true',
                      help='Generates the ouput of the optimizer.')
  parser.add_argument('--optimizedvcg', action='store_true',
                      help='Generate the Visualization Compiler Graph output '
                      'of the optimized SSA.')
  parser.add_argument('-p', '--parsetreevcg', action='store_true',
                      help='Generate the Visualization Compiler Graph '
                      'output of the parse tree.')
  parser.add_argument('-r', '--ir', action='store_true',
                      help='Generate the Intermediate Representation.')
  parser.add_argument('--regassigned', action='store_true',
                      help='Generate the instructions with registers '
                      'and phi functions resolved.')
  parser.add_argument('-s', '--ssa', action='store_true',
                      help='Generate the Static Single Assignment.')
  parser.add_argument('--ssavcg', action='store_true',
                      help='Generate the Visualization Compiler Graph output '
                      'of the SSA form.')
  parser.add_argument('-t', '--dom', action='store_true',
                      help='Generate the Dominator Tree VCG output.')
  parser.add_argument('--virtualreg', action='store_true',
                      help='Allocate registers in the virtual space of '
                          'infinite registers.')
  parser.add_argument('--virtualregvcg', action='store_true',
                      help='Generate the Visualization Compiler Graph '
                           'for the virtual registers allocated and liveness '
                           'computed for the subgraphs.')

  args = parser.parse_args()
  filename, fileext = os.path.splitext(args.file_names[0])


  try:
    p = Parser(args.file_names[0])
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

  if args.parsetreevcg or args.dumpall:
    global_graph = p.trees['globals'].generate_vcg(
        'TREE-GLOBALS')
    function_definitions_graph = p.trees['function_definitions'].generate_vcg(
        'TREE-FUNCTIONDEFINITIONS')

    graph = """graph: { title: "PARSER TREES"
    port_sharing: no
    %s
    %s
    }
    """ % (global_graph, function_definitions_graph)

    parser_vcg_file = open('%s.parser.vcg' % (
        filename), 'w')
    parser_vcg_file.write(graph)
    parser_vcg_file.close()

  global_symbol_table = p.symbol_table[GLOBAL_SCOPE_NAME]
  for function_tree in p.trees['function_definitions'].children:
    # The first child of the function root is the "ident" node which
    # is the function name. So get the symbol table entries from that
    # name of the function.
    function_name = function_tree.children[0].value
    symbol_table = p.symbol_table[function_name]
    ir = IntermediateRepresentation(function_tree, symbol_table,
                                    global_symbol_table)
    ir.generate()


    cfg = ir.build_cfg()
    cfg.compute_dominance_frontiers()

  if args.cfg or args.dumpall:
    ir_cfg_vcg_file = open('%s.ir.cfg.vcg' % filename, 'w')
    ir_cfg_vcg_file.write(cfg.generate_vcg(ir=ir.ir))
    ir_cfg_vcg_file.close()

  if args.ir or args.dumpall:
    ir_file = open('%s.ir' % filename, 'w')
    ir_file.write(str(ir))
    ir_file.close()

  if args.dom or args.dumpall:
    dom_file = open('%s.domtree.vcg' % filename, 'w')
    dom_file.write(str(cfg.generate_dom_vcg()))
    dom_file.close()


  ssa = SSA(ir, cfg)
  ssa.construct()

  if args.ssavcg or args.dumpall:
    ssa_cfg_vcg_file = open('%s.ssa.cfg.vcg' % filename, 'w')
    ssa_cfg_vcg_file.write(ssa.ssa_cfg.generate_vcg(ir=ssa.ssa))
    ssa_cfg_vcg_file.close()

  if args.ssa or args.dumpall:
    ssa_file = open('%s.ssa' % filename, 'w')
    ssa_file.write(str(ssa))
    ssa_file.close()


  optimize = Optimize(ssa)
  optimize.optimize()

  if args.optimized or args.dumpall:
    optimized_file = open('%s.optimized.ssa' % filename, 'w')
    optimized_file.write('\n'.join([str(s) for s in ssa.optimized()]))
    optimized_file.close()

  if args.optimizedvcg or args.dumpall:
    ssa_after_optimized_vcg_file = open('%s.optimized.ssa.vcg' % filename, 'w')
    ssa_after_optimized_vcg_file.write(ssa.ssa_cfg.generate_vcg())
    ssa_after_optimized_vcg_file.close()


  regalloc = RegisterAllocator(ssa)
  is_allocated, failed_subgraph = regalloc.allocate()

  if args.virtualregvcg or args.dumpall:
    virtualreggraph_file = open('%s.virtualreg.vcg' % filename, 'w')
    virtualreggraph_file.write(ssa.ssa_cfg.generate_virtual_reg_vcg(ssa=ssa))
    virtualreggraph_file.close()

  if args.virtualreg or args.dumpall:
    virtualreg_file = open('%s.virtualreg' % filename, 'w')
    virtualreg_file.write(regalloc.str_virtual_register_allocation())
    virtualreg_file.close()

  if args.interferencevcg or args.dumpall:
    interference_vcg_file = open('%s.virtualreg.interference.vcg' % filename,
                                 'w')
    for graph in regalloc.interference_graphs:
      interference_vcg_file.write('%s\n' % graph.generate_vcg())

    interference_vcg_file.close()


  regalloc.deconstruct_ssa()

  if args.regassigned or args.dumpall:
    reg_assigned_file = open('%s.reg.assigned' % filename, 'w')
    reg_assigned_file.write('%s\n' % regalloc.almost_machine_instructions())
    reg_assigned_file.close()


  cg = CodeGenerator(ssa)
  cg.generate()

  return cg

if __name__ == '__main__':
  cg = bootstrap()
