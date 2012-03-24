# Copyright 2012 Madhusudan C.S.
#
# This file datastructures.py is part of PL241-MCS compiler.
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

"""Collection of all the datastructures required for PL241-MCS Compiler.
"""


import heapq
import collections


class StackUnderflowException(Exception):
  """Exception to represent the stack underflow operation.
  """
  def __init__(self, msg=None):
    self._msg = msg if msg else "There was a stack underflow."

  def __str__(self):
    return self._msg

  def __repr__(self):
    return self.__str__()


class NodeProcessedException(Exception):
  """Exception to indicate that the node is already processed
  """
  def __init__(self, msg=None):
    self._msg = msg if msg else "The node is already processed."

  def __str__(self):
    return self._msg

  def __repr__(self):
    return self.__str__()


class Stack(list):
  """Implements the stack datastructure.

  NOTE: pop operation is not implemented because the pop method provided
  by the list type which this class inherits by default removes the last
  element which is the top of the stack.
  """

  def push(self, element):
    """Pushes a single element to the top of the stack.

    Args:
      elements: An element that should be pushed to the top of the stack.
    """
    self.append(element)

  def top(self):
    """Returns the element at the top of the stack without removing it.
    """
    try:
      return self[-1]
    except IndexError:
      raise StackUnderflowException

  def pop(self, *args, **kwargs):
    """Pops the element out of the top of stack.
    """
    try:
      return super(Stack, self).pop(*args, **kwargs)
    except IndexError:
      raise StackUnderflowException


class Node(object):
  """Represents a node in the parse tree.
  """

  def __init__(self, node_type=None, value=None, parent=None, children=None):
    """Initializes a node in the parse tree along with its pointers.

    Args:
      node_type: The type of the node, can take various values, see below.
      value: The value to be stored along with the node in the parse tree.
      parent: The optional parent of this node in the parse tree.
      children: The optional children of this node in the parse tree.

    node_type values:
      abstract: The node does not represent any real grammar element but an
          abtract intermediate type in the grammar like varDecl, funcDecl. The
          value for this type of node indicates the abstract name used.
      keyword: The node represents the program keyword, the value stores the
          name of the keyword.
      ident: The node represents the ident type of data, the value stores the
          name of the ident.
      number: The node contains the numerical value resulting from an
          expression. This stores the actual numerical value as the value.
          Since our grammar supports only integer values, we always store the
          value of the Node with type value as an integer type.
      control: The node represents the control character in the grammar. The
          value will be one of the control characters.
      operator: The node represents one of the operators in the grammar. It
          can be either relational operator or any other operator. The value of
          the node will just be the operator itself.
    """
    self.type = node_type
    self.value = value
    self.parent = parent
    # cast the children argument passed as any type to list before storing
    # it as class attributes
    self.children = list(children) if children else []

    if self.parent:
      self.parent.append_children(self)

    self.vcg_output = None

  def append_children(self, *children):
    """Appends children to the end of the list of children.

    Args:
      children: tuple of children that must be appended to this node in order.
    """
    self.children.extend(children)
    for child in children:
      child.parent = self

  def compress(self):
    """Compress the tree path as up as there is only one child up the hierarchy.
    """
    parent = self.parent

    if not parent:
      return

    parent_children = parent.children
    if len(parent_children) != 1:
      return

    index = parent.parent.children.index(parent)
    self.parent = parent.parent
    self.parent.children[index] = self

    self.parent.compress()

  def generate_tree_for_vcg(self, tree):
    """Recursively visit nodes of the tree with the given argument as the root.

    Args:
      tree: The root of the sub-tree whose nodes we must visit recursively.
    """
    self.vcg_output.append(str(tree))
    for child in tree.children:
      self.generate_tree_for_vcg(child)
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (id(tree), id(child)))

  def generate_vcg(self, title="TREE"):
    """Generate the Visualization of Compiler Graphs for this node as the root.
    """
    self.vcg_output = []
    self.generate_tree_for_vcg(self)

    return """graph: { title: "%(title)s"
    height: 700
    width: 700
    x: 30
    y: 30
    color: lightcyan
    stretch: 7
    shrink: 10
    layoutalgorithm: tree
    layout_downfactor: 10
    layout_upfactor:   1
    layout_nearfactor: 0
    manhattan_edges: yes
    %(nodes_edges)s
}""" % {
        'title': title,
        'nodes_edges': '\n    '.join(self.vcg_output)
        }

  def __str__(self):
    return 'node: { title: "%(id)s" label: "%(type)s: %(value)s" }' % {
        'id': id(self),
        'value': self.value,
        'type': self.type
        }

  def __repr__(self):
    return self.__str__()


class CFGNode(Node):
  """Represents a node in the control flow graph.

  Contains some additional attributes like dominator for the node etc. and
  parent and children are renamed to in_edges and out_edges.

  Attributes:
    value: The value to be stored in the current node.
    in_edges: The set of nodes which have an edge to this node.
    out_edges: The set of nodes to which this node has an edge.
    dom_children: The set of children in the dominator tree for this node.
    dom_parent: The parent in the dominator tree for this node.
  """

  def __init__(self, value=None, in_edges=None,
               out_edges=None, entry=False):
    """Constructs a node of the Control Flow Graph.

    Args:
      value: The value to be stored in the current node.
      in_edges: The set of nodes which have an edge to this node.
      out_edges: The set of nodes to which this node has an edge.
      entry: True if this node is the entry node else False.
    """
    self.value = value

    # Cast the in_edges and out_edges argument passed as any type to list
    # before storing it as class attributes.
    self.in_edges = list(in_edges) if in_edges else []
    self.out_edges = list(out_edges) if out_edges else []

    # True if this node is an entry node, a beginning for a function.
    # This is needed to build the connected components and the start
    # of the graph for computing dominator trees.
    self.entry = entry

    # The node which is the parent of this node in the dominator tree.
    self.dom_parent = None

    # The nodes that are the children of this node in the dominator tree.
    self.dom_children = []

    # The dominance frontier for this node
    self.dominance_frontier = []

    # Dictionary storing every variable to which there is an assignment in
    # this node as key and a dummy value "True" as its value.
    self.assignments = {}

    # Dictionary storing every variable which is used in this node as
    # key and a dummy value "True" as its value.
    self.mentions = {}

    # Stores the dictionary of phi-functions added to this node where the
    # key is the variable name and the value is the dictionary containing
    # the LHS and RHS of the phi function, and a pointer to the
    # instruction in SSA.
    self.phi_functions = {}

    # Dictionary of the live-in variables for each basic block where the
    # keys of the dictionary are the variable names and values are just
    # the dummy True values.
    self.live_in = {}

    # Dictionary of the live variables intervals for only the start node
    # where the keys of the dictionary are the variable names and values
    # two tuples where first element of the tuple is the label of the
    # instruction where the variable goes live in the subgraph. The second
    # element of the tuple is the label of the instruction where the
    # variable goes dead in this subgraph.
    self.live_intervals = {}

    # Dictionary whose keys are the in_edges and the values are list of
    # variables that must be included only on that path
    self.live_include = {}

    # The start node contains all the phi-nodes for the entire program
    # function.
    self.phi_nodes = []

    # A list of all the instructions that belongs to this basic block,
    # this is used only after the register allocation phase because until
    # then the instructions are linearly ordered in IntermediateRepresentation
    # object's ir list according to their labels. However allocating registers
    # and resolving phi instructions introduce additional instructions that
    # breaks this ordering. So we store the entire list of instructions
    # explicitly from the point of allocating registers and deconstructing
    # SSA.
    self.instructions = []


  def append_in_edges(self, *in_edges):
    """Add the in-edges for this node and also update the out-edges.

    If we call this method, one should not call out_edges method for the same
    pair of nodes since this method already updates the out_edges too.
    """
    self.in_edges.extend(in_edges)
    for i in in_edges:
      i.out_edges.append(self)

  def append_out_edges(self, *out_edges):
    """Add the out-edges for this node and also update the in-edges.

    If we call this method, one should not call in_edges method for the same
    pair of nodes since this method already updates the in_edges too.
    """
    self.out_edges.extend(out_edges)
    for i in out_edges:
      i.in_edges.append(self)

  def append_dom_children(self, *children):
    """Appends the children to this node in the dominator tree.

    Args:
      children: tuple of children that must be appended to this node in order.
    """
    self.dom_children.extend(children)
    for child in children:
      child.dom_parent = self

  def __contains__(self, instruction_label):
    """Checks if instruction label belongs to this basic block.
    """
    return self.value[0] <= instruction_label <= self.value[1]

  def __str__(self):
    return 'node: { title: "%(id)s" label: "CFGNode: %(value)s" }' % {
        'id': id(self),
        'value': self.value,
        }

  def plain_str(self):
    """Plain string representations to be used within a VCG node.
    """
    return "'CFGNode: %s'" % (self.value,)


class CFG(list):
  """Stores the Control Flow Graph for the given source.
  """

  def __init__(self, *args, **kwargs):
    """Initializes the datastructures required for the graph.

    Delegates most of the construction to the list class which is the parent
    class for this class.
    """
    super(CFG, self).__init__(*args, **kwargs)

    # Stores a list of root nodes pointing to each of the dominator tree
    # in the control flow graph.
    self.dom_trees = []

    # Contains the lists of connected components. Each entry, i.e. connected
    # components are in turn lists.
    self.connected_components = []

  def dfs_connected_components(self, root, graph_copy, current_component):
    """Implements the DFS for finding connected components.

    This is different from general DFS in that it gives up when the node
    is already identified as part of the component.

    Args:
      root: The root node on which the DFS should be performed/
      graph_copy: a copy of the GRAPH as a dictionary with nodes as keys
         and all values are True
      current_component: The dictionary of currently found connected
          components.
    """
    if root in current_component or root not in graph_copy:
      return

    current_component[root] = True
    graph_copy.pop(root)
    for child in root.out_edges:
      self.dfs_connected_components(child, graph_copy, current_component)

    for child in root.in_edges:
      self.dfs_connected_components(child, graph_copy, current_component)

  def compute_connected_components(self):
    """Identifies the connected components in the graph.
    """
    self.connected_components = []
    graph_copy = dict([(n, True) for n in self])
    for node in graph_copy.keys():
      component = {}
      self.dfs_connected_components(node, graph_copy, component)

      nodes = component.keys()
      if not nodes:
        continue

      for i, comp in enumerate(nodes):
        if comp.entry:
          break

      node = nodes.pop(i)
      nodes.insert(0, node)

      self.connected_components.append(nodes)

    return self.connected_components

  def compute_dominators(self):
    """Computes the dominator tree for the given graph.
    """
    self.compute_connected_components()
    self.dom_trees = []
    for component in self.connected_components:
      dom = Dominator(component)
      self.dom_trees.append(dom.construct())

    return self.dom_trees

  def compute_dominance_frontiers(self):
    """Computes the dominance frotiers for the control flow graph.
    """
    self.compute_dominators()
    for tree in self.dom_trees:
      df = DominanceFrontier(tree)
      df.compute_dominance_frontier()

    return self.dom_trees

  def generate_graph_for_vcg(self, node, ir=None, optimized={}):
    """Generate this node and all the outward edges from this node.

    Args:
      node: The node for which we need to generate the VCG output.
      ir: The Intermediate Representation object, if specified adds that to
          the node description.
    """
    node_str = ('node: { title: "%(id)s" color: lightgrey '
        'label: "CFGNode: %(value)s' % {
        'id': id(node),
        'value': node.value,
        })

    if node.dominance_frontier:
      dominance_edges = []
      node_str += '\n\nDominance Frontier: '
      for n in node.dominance_frontier:
        node_str += '%s' % (str(id(n)))
        dominance_edges.append('edge: {sourcename: "%s" color: green '
            'targetname: "%s" }' % (id(node), id(n)))

      node_str += '\n'

      self.vcg_output.extend(dominance_edges)

    if node.assignments:
      node_str += '\nAssignment Variables: %s\n' % (node.assignments.keys())

    if node.mentions:
      node_str += '\nMention Variables: %s\n' % (node.mentions.keys())

    if node.phi_functions:
      node_str += '\nPhi Functions\n'
      for v, phi_function in node.phi_functions.items():
        node_str += '\n%s = phi(' % (v)
        node_str += '%s, ' % phi_function['LHS']
        node_str += ', '.join([str(op) for op in phi_function['RHS']])
        node_str += ')'

      node_str += '\n\n'

    if ir:
      instructions = []
      instructions = [str(ir[i]) for i in range(
          node.value[0], node.value[1] + 1) if i not in optimized]

      node_str += '\n'.join(instructions)

    node_str += '" }'

    self.vcg_output.append(node_str)
    for out_edge in node.out_edges:
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (
              id(node), id(out_edge)))

  def generate_vcg(self, title="Control Flow Graph", ir=None, optimized={}):
    """Generate the Visualization of Compiler Graphs for this node as the root.
    """
    self.vcg_output = []
    for node in self:
      self.generate_graph_for_vcg(node, ir, optimized)

    return """graph: { title: "%(title)s"
    folding: 1
    hidden: 2
    height: 700
    width: 700
    x: 30
    y: 30
    stretch: 7
    shrink: 10
    orientation: top_to_bottom
    layout_downfactor: 10
    layout_upfactor:   1
    layout_nearfactor: 0
    manhattan_edges: yes
    %(nodes_edges)s
}""" % {
        'title': title,
        'nodes_edges': '\n    '.join(self.vcg_output)
        }

  def generate_virtual_reg_graph_for_vcg(self, node, ssa=None):
    """Generate this node and all the outward edges from this node.

    Args:
      node: The node for which we need to generate the VCG output.
      ssa: The SSA object required to generate optimized output.
    """
    node_str = ('node: { title: "%(id)s" color: lightgrey '
        'label: "CFGNode: %(value)s\nLive-In:\n' % {
        'id': id(node),
        'value': node.value,
        })

    for variable in sorted(node.live_intervals, key=lambda k: k.name):
      liveness = node.live_intervals[variable]
      node_str += '%s: (%s...%s)\n' % (
          variable,
          liveness[0] if liveness[0] else 'None',
          liveness[1] if liveness[1] else 'None')

    for variable in sorted(node.live_in):
      node_str += '%s\n' % (variable)

    if node.phi_functions:
      node_str += '\nPhi Functions\n'
      for v, phi_function in node.phi_functions.items():
        node_str += '\n%s <- phi(' % (phi_function['LHS'])
        node_str += ', '.join([str(op) for op in phi_function['RHS']])
        node_str += ')'

      node_str += '\n\n'

    for instruction in ssa.optimized(node.value[0],
                                     node.value[1] + 1):
      node_str += '\n%10s <-%s' % (
          instruction.result if instruction.result else '', instruction)

    node_str += '" }'

    self.vcg_output.append(node_str)
    for out_edge in node.out_edges:
      if ((node, out_edge) in self.edge_processed or
          (out_edge, node) in self.edge_processed):
        continue
      self.edge_processed[(node, out_edge)] = True
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (
              id(node), id(out_edge)))

  def generate_virtual_reg_vcg(
      self, title="Control Flow Graph For Optimized SSA After Virtual "
      "Register Allocation", ssa=None):
    """Generate the Visualization of Compiler Graphs for this node as the root.

    Args:
      title: Title of the graph
      ssa: The SSA object needed to generate this graph.
    """
    self.edge_processed = {}
    self.vcg_output = []
    for node in self:
      self.generate_virtual_reg_graph_for_vcg(node, ssa)

    return """graph: { title: "%(title)s"
    folding: 1
    hidden: 2
    height: 700
    width: 700
    x: 30
    y: 30
    stretch: 7
    shrink: 10
    orientation: top_to_bottom
    layout_downfactor: 10
    layout_upfactor:   1
    layout_nearfactor: 0
    manhattan_edges: yes
    %(nodes_edges)s
}""" % {
        'title': title,
        'nodes_edges': '\n    '.join(self.vcg_output)
        }

  def generate_dom_tree_for_vcg(self, tree):
    """Recursively visit nodes of the tree with the given argument as the root.

    Args:
      tree: The root of the sub-tree whose nodes we must visit recursively.
    """
    node_str = 'node: { title: "%(id)s" label: "CFGNode: %(value)s" }' % {
        'id': id(tree),
        'value': tree.value,
        }
    self.vcg_output.append(node_str)
    for child in tree.dom_children:
      self.generate_dom_tree_for_vcg(child)
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (id(tree), id(child)))

  def generate_dom_vcg(self, title="DOMINATOR TREE"):
    """Generate the Visualization of Compiler Graphs for this node as the root.
    """
    self.vcg_output = []
    for dom_tree in self.dom_trees:
      self.generate_dom_tree_for_vcg(dom_tree)

    return """graph: { title: "%(title)s"
    height: 700
    width: 700
    x: 30
    y: 30
    color: lightcyan
    stretch: 7
    shrink: 10
    layoutalgorithm: tree
    layout_downfactor: 10
    layout_upfactor:   1
    layout_nearfactor: 0
    manhattan_edges: yes
    %(nodes_edges)s
}""" % {
        'title': title,
        'nodes_edges': '\n    '.join(self.vcg_output)
        }


class Dominator(object):
  """Stores the dominator tree for a given Graph.

  This class implements dominator tree construction algorithm proposed by
  Thomas Lengauer and Robert Endre Tarjan in their landmark paper
  "Fast Algorithm for Finding Dominators in a Flowgraph" available at:

  http://dl.acm.org/citation.cfm?doid=357062.357071
  """

  def __init__(self, graph):
    """Initializes the dominator tree and all other datastructures required.

    NOTE: This class assumes that the whole dominator tree construction is done
    in memory. If we ever do construction out of memory the semi dominator
    dictionary used must be completely reimplemented since its keys are memory
    dependent.

    Args:
      graph: The control flow graph which is the input for this class supplied
          as AdjacencyList of vertices. The first element of the list is
          assumed to be the start vertex of the graph.
    """
    self.graph = graph

    # This dictionary contains a key value pair for every dictionary where
    # the keys are the vertex itself and the value is another dictionary.
    # The value dictionary corresponding to the vertex key contains exactly
    # 5 key-value pairs:
    #     'semi': Stores the semi-dominator information for the given vertex.
    #     'parent':the parent on the given vertex in the spanning tree.
    #     'pred': List of predecessors for the vertex.
    #     'bucket': List of vertices whose semi-dominator is this vertex.
    #     'dom': The dominator of this vertex.
    #     'ancestor': Required for path compression during link and eval.
    #     'label': Required for path compression during link and eval.
    #
    # NOTE: The paper uses the variable name vertex for the list of vertices.
    # We will avoid using this datastructure by using the OrderedDict 
    # datastructure from the collections module which is a dictionary with
    # the ordering for keys. The keys are ordered in the order in which they
    # were inserted into the dictionary.
    self.vertices = None

  def construct(self):
    """Constructs the dominator tree for this object.
    """
    self.vertices = collections.OrderedDict()

    # Step 1 in the paper
    self.dfs(self.graph[0])

    vertices_order = self.vertices.keys()

    # Combined Steps 2 and 3 in the paper.
    # -1:1:-1 ensures that we start with the last vertex and go on up to the
    # first vertex which is the root, but not including the first vertex
    # in steps of -1 which is the reverse ordering.
    for w in vertices_order[-1:0:-1]:
      # Step 2 in the paper
      for v in self.vertices[w]['pred']:
        u = self.eval_dom(v)
        if self.vertices[u]['semi'] < self.vertices[w]['semi']:
          self.vertices[w]['semi'] = self.vertices[u]['semi']

      bucket_vertex = vertices_order[self.vertices[w]['semi']]
      self.vertices[bucket_vertex]['bucket'].append(w)
      self.link(self.vertices[w]['parent'], w)

      # Step 3 in the paper
      for v in self.vertices[self.vertices[w]['parent']]['bucket']:
        u = self.eval_dom(v)
        if self.vertices[u]['semi'] < self.vertices[v]['semi']:
          self.vertices[v]['dom'] = u
        else:
          self.vertices[v]['dom'] = self.vertices[w]['parent']

      # We do this as an alternative to removing each entry in the parent's
      # bucket because of the way Python's for construct works. This is an
      # implementation level change.
      self.vertices[self.vertices[w]['parent']]['bucket'] = []

    # Step 4 in the paper.
    for w in vertices_order[1:]:
      if self.vertices[w]['dom'] != vertices_order[self.vertices[w]['semi']]:
        self.vertices[w]['dom'] = self.vertices[self.vertices[w]['dom']]['dom']

    # As one final step we construct the dominator tree for the datastructure
    # that we have chosen. This is not part of the paper, but we need it for
    # our custom datastructure.
    self.construct_dom_tree()

    return self.graph[0]

  def number(self, vertex):
    """Assigns a number for the given vertex and updates the datastructures.

    It updates those datastructures that depend on the vertex numbering.

    Args:
      vertex: The vertex which should be numbered.
    """
    if vertex in self.vertices:
      raise NodeProcessedException

    self.vertices[vertex] = {
        'semi': len(self.vertices),
        'parent': None,
        'pred': [],
        'bucket': [],
        'dom': None,
        'ancestor': None,
        'label': vertex,
    }

  def dfs(self, vertex):
    """Perform depth-first search on the input graph and enumerate the nodes.

    Args:
      vertex: The vertex which is under processing during the depth-first
          search.
    """
    try:
      self.number(vertex)
    except NodeProcessedException:
      return

    for w in vertex.out_edges:
      if w not in self.vertices:
        self.dfs(w)
        # Note the order of this operation is swapped from the one given in
        # the paper. This really doesn't change the algorithm since parent
        # of w can be set to vertex before doing a dfs of w or after since
        # dfs looks only at the children of a node, not its parents. But in
        # case of our implementation this is important because until we do a
        # dfs on w, w will not be numbered which means that the entry of w
        # is still not created in self.vertices and hence
        # self.vertices[w]['parent'] will give us a KeyError. Hence the order
        # swapping.
        self.vertices[w]['parent'] = vertex

      self.vertices[w]['pred'].append(vertex)

  def link(self, v, w):
    """Creates an edge from the vertex v to vertex w in the dominator tree.

    Args:
      v: vertex which forms the tail for the edge we are adding.
      w: vertex which forms the head for the edge we are adding.
    """
    self.vertices[w]['ancestor'] = v

  def eval_dom(self, v):
    """Evaluates the dominator or the semi-dominator for the given vertex.

    Returns a vertex "u" among vertices numbered greater than "w" satisfying
    "u" has a path to "v" whose semidominator has the minimum number. To know
    what is "w" look at the construct method above.

    NOTE: The naming is a bit weird with _dom because eval is a keyword
    in Python.

    Args:
      v: vertex which we must evaluate for now.
    """
    # The order of the if-else is swapped from the way it is presented in the
    # paper just for better readability.
    if self.vertices[v]['ancestor']:
      self.compress(v)
      return self.vertices[v]['label']
    else:
      return v

  def compress(self, v):
    """Perform path compression for performing eval according to the paper.

    IMPORTANT: This method assumes self.vertices[v]['ancestor'] is not None,
    i.e. v is not the root of any forest.
    """
    # The order of the if-else is swapped from the way it is presented in the
    # paper just for better readability.
    if not self.vertices[self.vertices[v]['ancestor']]['ancestor']:
      return

    self.compress(self.vertices[v]['ancestor'])
    if (self.vertices[self.vertices[self.vertices[v][
        'ancestor']]['label']]['semi'] <
        self.vertices[self.vertices[v]['label']]['semi']):
      self.vertices[v]['label'] = \
          self.vertices[self.vertices[v]['ancestor']]['label']

    self.vertices[v]['ancestor'] = \
        self.vertices[self.vertices[v]['ancestor']]['ancestor']

  def construct_dom_tree(self):
    """Constructs the dominator tree in the CFGNode objects.
    """
    for v in self.vertices:
      if self.vertices[v]['dom']:
        self.vertices[v]['dom'].append_dom_children(v)


class DominanceFrontier(object):
  """Implements the dominance frontier algorithm for the control flow graph.

  NOTE: We don't store DF local and DF up separately since we can
  directly combine them together in our program and they are anyway
  not required separately after we compute the dominance frontier.

  This class implements dominance frontier construction as presented in
  the paper by Ron Cytron, Jeanne Ferrante, Barry K. Rosen, Mark N. Wegman,
  and Z. Kenneth Zadeck in their landmark paper on minimal-SSA construction,
  "Efficiently Computing Static Single Assignment Form and the Control
  Dependence Graph" available at:

  http://dl.acm.org/citation.cfm?doid=115372.115320
  """

  def __init__(self, domtree):
    """Initalizes the datastructures required for computing dominance frontier.
    """
    self.domtree = domtree

  def compute_df_local(self, node):
    """Computes the DF local part of the algorithm for the given node.

    NOTE: The parameter used for this function is called "node" and not
    "root" unlike other methods because here we are working with the node
    in the control flow graph by computing values for its successors not
    with the dominator tree.

    Args:
      node: The node of the control flow graph for which the DF local should
          be calculated.
    """
    for out_node in node.out_edges:
      if self.idom(out_node) != node:
        node.dominance_frontier.append(out_node)

  def compute_df_up(self, root):
    """Computes the DF up part for the given root of the subtree.

    Args:
      root: The root of the dominator sub-tree for which the DF up should
          be computed.
    """
    for child in root.dom_children:
      for domnode in child.dominance_frontier:
        if self.idom(domnode) != root:
          root.dominance_frontier.append(domnode)

  def post_order(self, root):
    """Does the post order traversal of the dominator tree.

    Args:
      The root of the subtree that must be traversed in post order fashion.
    """
    for child in root.dom_children:
      self.post_order(child)

    self.compute_df_local(root)
    self.compute_df_up(root)

  def compute_dominance_frontier(self):
    """Computes the dominance frontier for the given tree.
    """
    self.post_order(self.domtree)

  def idom(self, root):
    """Returns the immediate dominator of the given root in the dominator tree.

    Args:
      root: The node for which the immediate dominator should be found
    """
    return root.dom_parent


class InterferenceNode(object):
  """Represents a node in the interference graph.
  """

  def __init__(self, register, instructions=None):
    """Initializes a node in the parse tree along with its pointers.

    Args:
      register: The name of the register
      instructions: A two tuple representing the range of the instructions
          that the register covers.
    """
    self.register = register
    self.instructions = instructions

    # cast the edges argument passed as any type to set before storing
    # it as class attributes
    self.edges = set([])

    self.vcg_output = None

  def append_edges(self, *edges):
    """Appends children to the end of the list of children.

    Args:
      edges: tuple/set of edges that must be appended to this node.
    """
    self.edges.update(edges)
    for edge in edges:
      edge.edges.add(self)

  def __str__(self):
    return 'node: { title: "%s" label: "%s: (%d..%d)" }' % (
        id(self), self.register, self.instructions[0], self.instructions[1])

  def __repr__(self):
    return self.__str__()


class InterferenceGraph(list):
  """The interference graph for the interfering registers.

  Essentially contains a list of nodes, so it inherits from the list type.
  """

  def __init__(self, *args, **kwargs):
    """Constructs the Interference graph datastructures.

    Args:
      args: positional arguments to be passed to the super class.
      kwargs: keyword arguments to be passed to the super class.
    """
    super(InterferenceGraph, self).__init__(*args, **kwargs)

  def generate_for_vcg(self, node):
    """Generate the VCG for the given node and all its edges

    Args:
      node: The node of the graph whose nodes we must visit recursively.
    """
    node_str = ('node: { title: "%(id)s" label: '
          ' "InterferenceNode: %(register)s" }' % {
          'id': id(node),
          'register': node.register,
          })

    self.vcg_output.append(node_str)
    for edge in node.edges:
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (id(node), id(edge)))

  def generate_vcg(self, title="INTERFERENCE GRAPH"):
    """Generate the Visualization of Compiler Graphs for this graph.
    """
    self.vcg_output = []
    for node in self:
      self.generate_for_vcg(node)

    return """graph: { title: "%(title)s"
    height: 700
    width: 700
    x: 30
    y: 30
    color: lightcyan
    stretch: 7
    shrink: 10
    layout_downfactor: 10
    layout_upfactor:   1
    layout_nearfactor: 0
    manhattan_edges: no
    smanhattan_edges: no
    %(nodes_edges)s
}""" % {
        'title': title,
        'nodes_edges': '\n    '.join(self.vcg_output)
        }


class LiveIntervalsHeap(dict):
  """Extends the Python's heap implementation to a datastructure.

  This is ridiculous, I don't understand why Python's heapq is not already
  doing this.
  """

  def __init__(self, *args, **kwargs):
    """Constructs the heap datastructure.
    """
    super(LiveIntervalsHeap, self).__init__(*args, **kwargs)

    # This is an inverted map of start of the interval to the register
    self.heap_map = collections.defaultdict(list)

    # Stores the keys of the inverted intervals index for the entire duration
    # of the object.
    self.all_heap_keys = []
    for key in self.keys():
      self.heap_map[self[key][0]].append(key)

      # Sorting on the start of the intervals
      # IMPORTANT: In one of trials we were using self.heap_map.keys()
      # as heap_keys, but this doesn't work because, multiple instructions
      # can have the same start point, so the inverted index will have
      # multiple registers for the same start instruction and for each of
      # them a separate heap key entry needs to be made.
      self.all_heap_keys.append(self[key][0])

    # While the previous all_heap_keys remains as is during the entire
    # duration of the object, this list gets shortened and repopulated
    # during every traversal of the heap.
    self.heap_keys = []
    heapq.heapify(self.heap_keys)

    # List containing the registers that have been popped in the order
    # they have been popped, the last popped register is at the end of
    # this list
    self.popped = []

  def pop(self):
    """Pop an element out of the heap.
    """
    heap_key = heapq.heappop(self.heap_keys)
    popped = self.heap_map[heap_key].pop()
    self.popped.append(popped)
    # Reinsert it back to the map, since we want it.
    self.heap_map[heap_key].insert(0, popped)
    return popped

  def push(self, register, interval):
    """Pushes an item into the heap.

    register: The name of the register that must be pushed to the heap.
    interval: The interval range that must be pushed to the heap.
    """
    self[register] = interval
    self.heap_map[interval[0]].append(register)
    return heapq.heappush(self.heap_keys, interval[0])

  def previous(self):
    """Returns the register that was last popped.
    """
    if len(self.popped) > 1:
      return self.popped[-2]

    return None

  def __iter__(self):
    """Returns this object which implements the iterator protocol.
    """
    # Reload the heap keys for the iteration and heapify them if the heap_keys
    # list is empty
    if not self.heap_keys:
      self.heap_keys = self.all_heap_keys
      heapq.heapify(self.heap_keys)
      self.popped = []

    return self

  def next(self):
    """Gets the next item in the heap, essentially a pop.
    """
    if not self.heap_keys:
      raise StopIteration

    return self.pop()
