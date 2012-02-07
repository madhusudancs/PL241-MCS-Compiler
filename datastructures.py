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


from collections import OrderedDict


class NodeProcessedException(Exception):
  """Exception to indicate that the node is already processed
  """
  def __init__(self, msg=None):
    self._msg = msg if msg else "The node is already processed."

  def __str__(self):
    return self._msg

  def __repr__(self):
    return self.__str__()


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
      number: The node contains the numerical value resulting from an expression.
          This stores the actual numerical value as the value. Since our grammar
          supports only integer values, we always store the value of the Node
          with type value as an integer type.
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

  def __init__(self, value=None, in_edges=None, out_edges=None):
    """Constructs a node of the Control Flow Graph.

    Args:
      value: The value to be stored in the current node.
      in_edges: The set of nodes which have an edge to this node.
      out_edges: The set of nodes to which this node has an edge.
    """
    self.value = value

    # Cast the in_edges and out_edges argument passed as any type to list
    # before storing it as class attributes.
    self.in_edges = list(in_edges) if in_edges else []
    self.out_edges = list(out_edges) if out_edges else []

    self.dom_parent = None
    self.dom_children = []

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

  def generate_dom_tree_for_vcg(self, tree):
    """Recursively visit nodes of the tree with the given argument as the root.

    Args:
      tree: The root of the sub-tree whose nodes we must visit recursively.
    """
    self.vcg_output.append(str(tree))
    for child in tree.dom_children:
      self.generate_dom_tree_for_vcg(child)
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (id(tree), id(child)))

  def generate_dom_vcg(self, title="DOMINATOR TREE"):
    """Generate the Visualization of Compiler Graphs for this node as the root.
    """
    self.vcg_output = []
    self.generate_dom_tree_for_vcg(self)

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
    return 'node: { title: "%(id)s" label: "CFGNode: %(value)s" }' % {
        'id': id(self),
        'value': self.value,
        }


class CFG(object):
  """Stores the Control Flow Graph for the given source.
  """

  def __init__(self, nodes):
    """Initializes the control flow graph.

    Args:
      nodes: The set of nodes which are of CFGNode type that constitute this
          graph
    """
    self.nodes = nodes

    self.num_nodes = len(nodes)

    self.iter_count = 0

  def __iter__(self):
    """The iterator for this class is the object of this class itself.
    """
    self.iter_count = 0
    return self

  def next(self):
    """The next method as required by the iterator protocol.
    """
    if self.iter_count == self.num_nodes:
      raise StopIteration

    node = self.nodes[self.iter_count]
    self.iter_count += 1
    return node


class Dominator(object):
  """Stores the dominator tree for a given Graph.

  This class implements dominator tree construction algorithm proposed by
  Thomas Lengauer and Robert Endre Tarjan in their landmark paper
  "Fast Algorithm for Finding Dominators in a Flowgraph"
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
    self.vertices = OrderedDict()

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
