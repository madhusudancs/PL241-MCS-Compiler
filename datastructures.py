# Copyright 2012 Madhusudan C.S. 
#
# This file parser.py is part of PL241-MCS compiler.
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

  def __init__(self, node_type=None, name=None, parent=None, children=None):
    """Initializes a node in the parse tree along with its pointers.

    Args:
      node_type: The type of the node, can take various values, see below.
      name: The value to be stored along with the node in the parse tree.
      parent: The optional parent of this node in the parse tree.
      children: The optional children of this node in the parse tree.

    node_type values:
      abstract: The node does not represent any real grammar element but an
          abtract intermediate type in the grammar like varDecl, funcDecl. The
          name for this type of node indicates the abstract name used.
      keyword: The node represents the program keyword, the name stores the
          name of the keyword.
      ident: The node represents the ident type of data, the name stores the
          name of the ident.
      number: The node contains the numerical value resulting from an expression.
          This stores the actual numerical value as the name. Since our grammar
          supports only integer values, we always store the name of the Node
          with type value as an integer type.
      control: The node represents the control character in the grammar. The
          name will be one of the control characters.
      operator: The node represents one of the operators in the grammar. It
          can be either relational operator or any other operator. The name of
          the node will just be the operator itself.
    """
    self.type = node_type
    self.name = name
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

  def generate_tree_for_vcg(self, tree):
    self.vcg_output.append(str(tree))
    for child in tree.children:
      self.generate_tree_for_vcg(child)
      self.vcg_output.append(
          'edge: {sourcename: "%s" targetname: "%s" }' % (id(tree), id(child)))

  def generate_vcg(self):
    self.vcg_output = []
    self.generate_tree_for_vcg(self)

    print """graph: { title: "SYNTAXTREE"
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
    %s
}""" % ('\n    '.join(self.vcg_output))

  def __str__(self):
    return 'node: { title: "%(id)s" label: "%(type)s: %(name)s" }' % {
        'id': id(self),
        'name': self.name,
        'type': self.type
        }

  def __repr__(self):
    return self.__str__()


class Dominator(object):
  """Stores the Dominator Tree for a given Graph.

  This class implements Dominator Tree Construction Algorithm proposed by
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
      graph: The Control Flow Graph which is the input for this class supplied
          as Adjacency List of Vertices. The first element of the list is
          assumed to be the start vertex of the graph.
    """
    self.graph = graph

    self.semi = None

    # The paper uses the variable name vertex even for this list and for
    # individual vertices under consideration at each step. However I don't
    # know any programming language that lets us do it ;-)
    self.vertices = None

  def construct(self):
    """Constructs the dominator tree for this object.
    """
    self.semi = {}

    self.vertices = []

    dfs(self.graph[0])

  def number(self, vertex):
    """Assigns a number for the given vertex and updates the datastructures.

    It updates those datastructures that depend on the vertex numbering.

    Args:
      vertex: The vertex which should be numbered.
    """
    if vertex in self.semi:
      raise NodeProcessedException

    self.semi[vertex] = len(self.vertices)

    self.vertices.append(vertex)

  def dfs(self, vertex):
    """Perform Depth-First search on the input graph and enumerate the nodes.

    Args:
      vertex: The vertex which is under processing during the depth-first
          search.
    """
    try:
      self.number(vertex)
    except NodeProcessedException:
      return

    for e in vertex.out_edges:
      if semi(e) = 0:
        parent(e) = vertex
        dfs(e)
      pred(w).append(vertex)

  def link_edge(self, v, w):
    """
    """
    pass

  def eval_dom(self, v):
    """
    """
    pass

