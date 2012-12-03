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

from callgraph import CallGraphAnalysis
from codegen import allocate_global_memory
from codegen import CodeGenerator
from elf import ELF
from ir import IntermediateRepresentation
from linker import Linker
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
  parser.add_argument('--callgraphvcg', action='store_true',
                      help='Generate the Visualization Compiler Graph '
                           'for the whole program call graph.')

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

  # A dictionary whose key is the function name and the value is another
  # dictionary with the functions compilation stages
  compilation_stages = {}

  for function_tree in p.trees['function_definitions'].children:
    # The first child of the function root is the "ident" node which
    # is the function name. So get the symbol table entries from that
    # name of the function.
    function_name = function_tree.children[0].value
    if function_name == 'main':
      symbol_table = {}
    else:
      symbol_table = p.symbol_table[function_name]
    ir = IntermediateRepresentation(
        function_name, function_tree, symbol_table, global_symbol_table)
    ir.generate()

    cfg = ir.build_cfg()
    cfg.compute_dominance_frontiers()

    compilation_stages[function_name] = {
      'name': function_name,
      'ir': ir,
      'cfg': cfg
      }

  if args.cfg or args.dumpall:
    ir_cfg_vcg_file = open('%s.ir.cfg.vcg' % filename, 'w')

    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      graph += function['cfg'].generate_vcg(title=function['name'],
                                            ir=function['ir'].ir)
      graph += '\n'

    graph += '}'
    ir_cfg_vcg_file.write(graph)
    ir_cfg_vcg_file.close()

  if args.ir or args.dumpall:
    ir_file = open('%s.ir' % filename, 'w')
    for function in compilation_stages.values():
      ir_file.write(str(function['ir']) + '\n\n')

    ir_file.close()


  if args.dom or args.dumpall:
    dom_file = open('%s.domtree.vcg' % filename, 'w')

    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      graph += str(function['cfg'].generate_dom_vcg(title=function['name']))
      graph += '\n'

    graph += '}'
    dom_file.write(graph)
    dom_file.close()


  for function_name in compilation_stages:
    ir = compilation_stages[function_name]['ir']
    cfg = compilation_stages[function_name]['cfg']
    ssa = SSA(ir, cfg)
    ssa.construct()
    compilation_stages[function_name]['ssa'] = ssa

  if args.ssa or args.dumpall:
    ssa_file = open('%s.ssa' % filename, 'w')
    for function in compilation_stages.values():
      ssa_file.write(str(function['ssa']) + '\n\n')

    ssa_file.close()

  if args.ssavcg or args.dumpall:
    ssa_cfg_vcg_file = open('%s.ssa.cfg.vcg' % filename, 'w')

    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      graph += function['ir'].cfg.generate_vcg(title=function['name'],
                                               ir=function['ir'].ir)
      graph += '\n'

    graph += '}'
    ssa_cfg_vcg_file.write(graph)
    ssa_cfg_vcg_file.close()

  for function_name in compilation_stages:
    optimize = Optimize(compilation_stages[function_name]['ssa'])
    optimize.optimize()
    compilation_stages[function_name]['optimize'] = optimize

  if args.optimized or args.dumpall:
    optimized_file = open('%s.optimized.ssa' % filename, 'w')
    for function in compilation_stages.values():
      optimized_file.write(str(function['optimize']) + '\n\n')

    optimized_file.close()

  if args.optimizedvcg or args.dumpall:
    ssa_after_optimized_vcg_file = open('%s.optimized.ssa.vcg' % filename, 'w')

    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      graph += function['ssa'].cfg.generate_vcg(
          title=function['name'], ir=function['ssa'].ir.ir,
          optimized=function['ssa'].optimized_removal)
      graph += '\n'

    graph += '}'
    ssa_after_optimized_vcg_file.write(graph)
    ssa_after_optimized_vcg_file.close()


  cga = CallGraphAnalysis(compilation_stages)
  cga.analyze()

  if args.callgraphvcg or args.dumpall:
    callgraph_file = open('%s.callgraph.vcg' % filename, 'w')

    graph = """graph: { title: "CallGraph"
    port_sharing: no
    """
    graph += cga.call_graph.generate_vcg()
    graph += '\n}'

    callgraph_file.write(graph)
    callgraph_file.close()

  for function_name in compilation_stages:
    regalloc = RegisterAllocator(
        compilation_stages[function_name]['ssa'])
    compilation_stages[function_name]['is_allocated'] = regalloc.allocate()
    compilation_stages[function_name]['regalloc'] = regalloc

  if args.virtualregvcg or args.dumpall:
    virtualreggraph_file = open('%s.virtualreg.vcg' % filename, 'w')

    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      graph += function['ssa'].cfg.generate_virtual_reg_vcg(
          title=function['name'], ssa=function['ssa'])
      graph += '\n'

    graph += '}'

    virtualreggraph_file.write(graph)
    virtualreggraph_file.close()

  if args.virtualreg or args.dumpall:
    virtualreg_file = open('%s.virtualreg' % filename, 'w')

    for function in compilation_stages.values():
      virtualreg_file.write(
        function['regalloc'].str_virtual_register_allocation() + '\n\n')

    virtualreg_file.close()

  if args.interferencevcg or args.dumpall:
    interference_vcg_file = open('%s.virtualreg.interference.vcg' % filename,
                                 'w')
    graph = """graph: { title: "CFG"
    port_sharing: no
    """

    for function in compilation_stages.values():
      if function['regalloc'].interference_graph:
        graph += function['regalloc'].interference_graph.generate_vcg(
            title=function['name'],)
        graph += '\n'

    graph += '}'
    interference_vcg_file.write(graph)
    interference_vcg_file.close()

  for function_name in compilation_stages:
    compilation_stages[function_name]['regalloc'].deconstruct_ssa()

  if args.regassigned or args.dumpall:
    reg_assigned_file = open('%s.reg.assigned' % filename, 'w')

    for function in compilation_stages.values():
      reg_assigned_file.write(
          function['regalloc'].almost_machine_instructions() + '\n\n')

    reg_assigned_file.close()


  global_memory_size = allocate_global_memory(global_symbol_table)

  generated_functions = []
  for function_name in compilation_stages:
    cg = CodeGenerator(compilation_stages[function_name]['ir'])
    cg.generate()
    generated_functions.append(cg)
    compilation_stages[function_name]['cg'] = cg

  # There is a cyclic dependency between the ELF builder the linker to resolve
  # the addresses of globals for each instruction. So we break this cyclic
  # dependency by first linking and building the ELF header as far as we
  # can i.e. upto linking functions and build ELF upto instructions offset
  # and then call the linker again with the ELF object to continue with globals
  # linking and finally call the ELF's final binary builder.
  linker = Linker(generated_functions)
  linker.link_functions()

  elf = ELF(filename, linker, global_memory_size)
  elf.build_upto_instructions_offset()

  linker.link_globals(elf)

  elf.build()

  linked_file = open('%s.linked.binary' % filename, 'w')
  linked_file.write(str(linker))
  linked_file.close()



  executable_file = open('%s' % filename, 'w')
  executable_file.write(str(elf))

  # Make the file executable.
  os.fchmod(executable_file.fileno(), 0755)

  executable_file.close()

  return elf


if __name__ == '__main__':
  elf = bootstrap()
