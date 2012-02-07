# Copyright 2012 Madhusudan C.S.
#
# This file test_datastructures.py is part of PL241-MCS compiler.
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

"""Contains the tests for the datastructures.
"""


import datastructures


def test_tree():
  root = datastructures.Node(node_type='value', value=1)
  nodes = [datastructures.Node(node_type='value', value=i) for i in range(2, 11)]
  root.append_children(nodes[0], nodes[1])
  nodes[0].append_children(nodes[2], nodes[3], nodes[4])
  nodes[1].append_children(nodes[5], nodes[6])
  nodes[3].append_children(nodes[7])
  nodes[7].append_children(nodes[8])


def test_dominator():
  nodes = [datastructures.CFGNode(i) for i in range(8)]
  nodes[0].append_out_edges(nodes[1])
  nodes[1].append_out_edges(nodes[2], nodes[3])
  nodes[2].append_out_edges(nodes[7])
  nodes[3].append_out_edges(nodes[4])
  nodes[4].append_out_edges(nodes[5], nodes[6])
  nodes[5].append_out_edges(nodes[7])
  nodes[6].append_out_edges(nodes[4])
  dom = datastructures.Dominator(nodes)
  dom_tree = dom.construct()
  assert set(dom_tree.dom_children) == set([nodes[1]])
  assert set(nodes[1].dom_children) == set([nodes[2], nodes[3], nodes[7]])
  assert set(nodes[2].dom_children) == set([])
  assert set(nodes[3].dom_children) == set([nodes[4]])
  assert set(nodes[4].dom_children) == set([nodes[5], nodes[6]])
  assert set(nodes[5].dom_children) == set([])
  assert set(nodes[6].dom_children) == set([])
  assert set(nodes[7].dom_children) == set([])


def test_dominance_frontier():
  entry = datastructures.CFGNode('entry')
  exit_node = datastructures.CFGNode('exit')

  nodes = [datastructures.CFGNode(i) for i in range(0, 13)]

  entry.append_out_edges(nodes[1], exit_node)

  nodes[1].append_out_edges(nodes[2])
  nodes[2].append_out_edges(nodes[3], nodes[7])
  nodes[3].append_out_edges(nodes[4], nodes[5])
  nodes[4].append_out_edges(nodes[6])
  nodes[5].append_out_edges(nodes[6])
  nodes[6].append_out_edges(nodes[8])
  nodes[7].append_out_edges(nodes[8])
  nodes[8].append_out_edges(nodes[9])
  nodes[9].append_out_edges(nodes[10], nodes[11])
  nodes[10].append_out_edges(nodes[11])
  nodes[11].append_out_edges(nodes[9], nodes[12])
  nodes[12].append_out_edges(nodes[2], exit_node)

  dom = datastructures.Dominator([entry] + nodes[1:] + [exit_node])
  dom_tree = dom.construct()
  df = datastructures.DominanceFrontier(dom_tree)
  df.compute_dominance_frontier()

  assert set(dom_tree.dom_children) == set([nodes[1], exit_node])
  assert set(nodes[1].dom_children) == set([nodes[2]])
  assert set(nodes[2].dom_children) == set([nodes[3], nodes[7], nodes[8]])
  assert set(nodes[3].dom_children) == set([nodes[4], nodes[5], nodes[6]])
  assert set(nodes[8].dom_children) == set([nodes[9]])
  assert set(nodes[9].dom_children) == set([nodes[10], nodes[11]])
  assert set(nodes[11].dom_children) == set([nodes[12]])

  assert set(entry.dominance_frontier) == set([])
  assert set(exit_node.dominance_frontier) == set([])
  assert set(nodes[1].dominance_frontier) == set([exit_node])
  assert set(nodes[2].dominance_frontier) == set([exit_node, nodes[2]])
  assert set(nodes[3].dominance_frontier) == set([nodes[8]])
  assert set(nodes[4].dominance_frontier) == set([nodes[6]])
  assert set(nodes[5].dominance_frontier) == set([nodes[6]])
  assert set(nodes[6].dominance_frontier) == set([nodes[8]])
  assert set(nodes[7].dominance_frontier) == set([nodes[8]])
  assert set(nodes[8].dominance_frontier) == set([exit_node, nodes[2]])
  assert set(nodes[9].dominance_frontier) == set([exit_node, nodes[2],
                                                  nodes[9]])
  assert set(nodes[10].dominance_frontier) == set([nodes[11]])
  assert set(nodes[11].dominance_frontier) == set([exit_node, nodes[2],
                                                   nodes[9]])
  assert set(nodes[12].dominance_frontier) == set([exit_node, nodes[2]])


if __name__ == '__main__':
  test_tree()
  test_dominator()
  test_dominance_frontier()
