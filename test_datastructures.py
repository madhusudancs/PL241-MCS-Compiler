import datastructures

def test_tree():
  root = datastructures.Node(node_type='value', name=1)
  nodes = [datastructures.Node(node_type='value', name=i) for i in range(2, 11)]
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
  dom_tree.generate_dom_vcg()

