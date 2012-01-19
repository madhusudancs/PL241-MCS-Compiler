import re

from argparse import ArgumentParser


class SyntaxError(Exception):
  def __init__(self, msg):
    self.__msg = msg

  def __str__(self):
    return 'SyntaxError: %s' % self.__msg


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
      value: The node contains the numerical value resulting from an expression.
          This stores the actual numerical value as the name. Since our grammar
          supports only integer values, we always store the name of the Node
          with type value as an integer type.
    """
    # convert the *args passed as tuple to list before storing it as
    # class attributed
    self.type = node_type
    self.name = name
    self.parent = parent
    self.children = list(children) if children else []

  def append_children(self, *children):
    """Appends children to the end of the list of children.

    Args:
      children: tuple of children that must be appended to this node in order.
    """

    self.children.extend(children)
    print self.children

  def __str__(self):
    return 'Node: %s "%s"' % (self.type, self.name)

  def __repr__(self):
    return self.__str__()


class Parser(object):
  """Abstracts the entire grammar parser along with building the parse tree.

  This parser is implemented as a home-brewn recursive descent parser with
  some help from regular expression library only for tokenizing.
  """

  KEYWORDS = ['main', 'let', 'var', 'array', 'if', 'while', 'function',
              'procedure', 'return', 'call']

  def __init__(self, program_file):
    """Initializes by reading the program file and constructing the parse tree.

    Args:
      program_file: the file object that contains the source code to compile.
    """
    self.src = program_file.read()
    self.__tokenize()
    self.__parse()

  def __tokenize(self):
    """Splits the entire source code into tokens using regular expression.
    """
    self.tokens = re.findall("(\d+|\w+|[^\s+])", self.src)

  def __parse(self):
    """Parses the tokens by delegating to appropriate functions and builds tree.
    """
    self.tokens

  def parse_main():
    pass

  def parse_let():
    pass

  def parse_var():
    pass

  def parse_array():
    pass

  def parse_if():
    pass

  def parse_while():
    pass

  def parse_function():
    pass

  def parse_procedure():
    pass

  def parse_return():
    pass

  def parse_call():
    pass



if __name__ == '__main__':
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=file, nargs='+',
                      help='name of the input files.')
  args = parser.parse_args()
  p = Parser(args.file_names[0])

