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

"""Implements the parser for the PL241 compiler.

EBNF of the PL241 grammar is as follows:

letter = "a".."z";
digit = "0".."9";
relOp = "==" | "!=" | "<" | "<=" | ">" | ">=";
ident = letter {letter | digit};
number = digit {digit};
designator = ident{ "[" expression "]" };
factor = designator | number | "(" expression ")" | funcCall;
term = factor { ("*" | "/") factor};
expression = term {("+" | "-") term};
relation = expression relOp expression;
assignment = "let" designator "<-" expression;
funcCall = "call" ident [ "(" [expression { "," expression } ] ")" ];
ifStatement = "if" relation "then" statSequence [ "else" statSequence ] "fi";
whileStatement = "while" relation "do" statSequence "od";
returnStatement = "return" [ expression ];
statement = assignment | funcCall | ifStatement | whileStatement | returnStatement;
statSequence = statement { ";" statement };
typeDecl = "var" | "array" "[" number "]" { "[" number "]" };
varDecl = typeDecl ident { "," ident } ";";
funcDecl = ("function" | "procedure") ident [formalParam] ";" funcBody ";";
formalParam = "(" [ident { "," ident }] ")";
funcBody = { varDecl } "{" [ statSequence ] "}";
computation = "main" { varDecl } { funcDecl } "{" statSequence "}" ".";
"""

import re
import sys

from argparse import ArgumentParser
from copy import deepcopy


class SyntaxError(Exception):
  def __init__(self, msg):
    self.__msg = msg

  def __str__(self):
    return 'SyntaxError: %s' % self.__msg


class StopParsing(Exception):
  def __init__(self, msg):
    self.__msg = msg

  def __str__(self):
    return 'StopParsing: %s' % self.__msg



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
      control: The node represents the control character in the grammar. The
          name will be one of the control characters.
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


class TokenStream(object):
  """Handles the token stream of the source.
  """

  def __init__(self, src):
    """Initializes the token stream from the source.
    """
    self.src = src

    # The stream pointers that keep track of where in the source code our
    # parser currently is.
    self.__stream_pointer = None

  def __tokenize(self):
    """Splits the entire source code stream into tokens using regular expression.
    """
    self.__tokens = re.findall("(\d+|\w+|[^\s+])", self.src)

    # Initializes the stream to the beginning of the tokens list.
    self.__stream_pointer = 0

  def fastforward(self, token):
    """Fast forward the stream upto the point we find the given token.
    """
    try:
      self.__stream_pointer = self.__tokens.index(token)
    except ValueError:
      raise SyntaxError('"%s" not found' % (token))

  def __iter__(self):
    """Setup the iterator protocol.
    """
    return self

  def next(self):
    """Get the next item in the token stream.
    """
    if self.__stream_pointer = None:
      self.__tokenize()

    try:
      next_token = self.__tokens[self.__stream_pointer]
      self.__stream_pointer += 1
    except IndexError:
      raise StopParsing('End of source has been reached while something '
                        'was expected.')

    return next_token


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

    # Close the program file, we do not need that anymore
    program_file.close()

    self.__tokenize()
    self.__parse()

  def __tokenize(self):
    """Splits the entire source code into tokens using regular expression.
    """
    self.__tokens = re.findall("(\d+|\w+|[^\s+])", self.src)

  def __parse(self):
    """Parses the tokens by delegating to appropriate functions and builds tree.
    """
    self.__stream = deepcopy(self.__tokens)
    self.__ff_to_main()
    self.__parse_main()

  def __ff_to_main(self):
    """Fast forward the stream upto the point we find the keyword main.

    We do this because the grammar defines the program entry point as main.
    """
    try:
      main_index = self.__stream.index('main')
      self.__stream = self.__stream[main_index:]
    except ValueError:
      raise SyntaxError('"main" not found')

  def __r_to_period(self):
    """Rewind the stream upto the point we find the period that ends program.

    We do this because the grammar defines the program exit point as period.
    """
    try:
      period_index = self.__stream.index('.')
      self.__stream = self.__stream[:period_index + 1]
    except ValueError:
      raise SyntaxError('"." not found')

    except ValueError:
      raise SyntaxError('"main" not found')

  def __parse_main(self):
    main = self.__stream[0]
    main_node = Node('keyword', main)

    period = self.__stream[-1]
    period_node = Node('keyword', period, main_node)

    main_node.append_children(period_node)

  def __parse_let(self):
    pass

  def __parse_var(self):
    pass

  def __parse_array(self):
    pass

  def __parse_if(self):
    pass

  def __parse_while(self):
    pass

  def __parse_function(self):
    pass

  def __parse_procedure(self):
    pass

  def __parse_return(self):
    pass

  def __parse_call(self):
    pass

  def is_keyword(word):
    return word in KEYWORDS


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=file, nargs='+',
                      help='name of the input files.')
  args = parser.parse_args()
  try:
    p = Parser(args.file_names[0])
  except SyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  bootstrap()

