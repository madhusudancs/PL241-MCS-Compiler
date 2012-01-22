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


import logging
import re
import sys

from argparse import ArgumentParser
from copy import deepcopy


LOGGER = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile('(\d+|\w+|<-|==|!=|<=|>=|[^\s+])')


class ParserBaseException(Exception):
  def __init__(self, msg):
    self._msg = msg

  def __str__(self):
    return '%s: %s' % (self.__class__.__name__, self._msg)


class LanguageSyntaxError(ParserBaseException):
  def __str__(self):
    return 'SyntaxError: %s' % self._msg


class EndControlException(ParserBaseException):
  def __init__(self, begin, end, msg=None):
    msg = msg if msg else ('%s encountered without a matching %s.' % (
        begin, end))
    super(EndControlException, self).__init__(msg)


class ThenFoundException(EndControlException):
  def __init__(self, msg=None):
    super(ThenFoundException, self).__init__('if', 'then', msg)


class ElseFoundException(EndControlException):
  def __init__(self, msg=None):
    super(ElseFoundException, self).__init__('if', 'else', msg)


class FiFoundException(EndControlException):
  def __init__(self, msg=None):
    super(FiFoundException, self).__init__('if', 'fi', msg)


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
      operator: The node represents one of the operators in the grammar. It
          can be either relational operator or any other operator. The name of
          the node will just be the operator itself.
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

    self.__tokenize()

  def __tokenize(self):
    """Splits the entire source code stream into tokens using regular expression.
    """
    self.__tokens = TOKEN_PATTERN.findall(self.src)

    LOGGER.debug('Parsed tokens: %s' % self.__tokens)

    # Initializes the stream to the beginning of the tokens list.
    self.__stream_pointer = 0

  def fastforward(self, token):
    """Fast forward the stream upto the point we find the given token.
    """
    try:
      self.__stream_pointer = self.__tokens.index(token)
    except ValueError:
      raise LanguageSyntaxError('"%s" not found' % (token))

  def __iter__(self):
    """Setup the iterator protocol.
    """
    return self

  def next(self):
    """Get the next item in the token stream.
    """
    if self.__stream_pointer == None:
      self.__tokenize()

    try:
      next_token = self.__tokens[self.__stream_pointer]
      self.__stream_pointer += 1
    except IndexError:
      self.__stream_pointer = None
      raise StopIteration

    return next_token


class Parser(object):
  """Abstracts the entire grammar parser along with building the parse tree.

  This parser is implemented as a home-brewn recursive descent parser with
  some help from regular expression library only for tokenizing.
  """

  KEYWORDS = ['main', 'let', 'var', 'array', 'if', 'while', 'function',
              'procedure', 'return', 'call']

  CONTROL_CHARACTERS_MAP = {
      ',': 'comma',
      ';': 'semicolon',
      '(': 'leftparen',
      ')': 'rightparen',
      '{': 'leftbrace',
      '}': 'rightbrace',
      '[': 'leftbracket',
      ']': 'rightbracket',
      }

  def __init__(self, program_file):
    """Initializes by reading the program file and constructing the parse tree.

    Args:
      program_file: the file object that contains the source code to compile.
    """
    program_file_obj = open(program_file, 'r')

    # Read the entire source in the supplied program file.
    self.src = program_file_obj.read()

    # Close the program file, we do not need that anymore since we have read
    # entire source in the program file.
    program_file_obj.close()

    self.root = self.__parse()

  def __parse(self):
    """Parses the tokens by delegating to appropriate functions and builds tree.
    """
    self.__token_stream = TokenStream(self.src)

    return self.__parse_main()

  def __parse_main(self):
    # We fast forward to "main" in the source code because the grammar defines
    # the program entry point as main.
    self.__token_stream.fastforward('main')

    main = self.__token_stream.next()
    main_node = Node('keyword', main)

    children_nodes = []

    for token in self.__token_stream:
      token_node = self.__parse_next_token(token, main_node)
      children_nodes.append(token_node)

    # The last token, which is essentially the end of the stream must be
    # a period token, otherwise there is a syntax error in the program
    # according to the grammar.
    if token != '.':
      raise LanguageSyntaxError('Program does not end with a "."')

    main_node.append_children(*children_nodes)

    return main_node

  def __parse_next_token(self, token, parent):
    """Parses the next token in the stream and returns the node for it.

    Args:
      token: the token that must be parsed.
    """
    if self.is_keyword(token):
      # Note that if the token is recognized as a keyword it should have
      # a parse method defined, otherwise getattr will raise an exception.
      # This is exception is not handled because this is likely to be a
      # error in the compiler implementation.
      parse_func = getattr(self, '_Parser__parse_%s' % (token))

      return parse_func(parent)

  def __parse_let(self, parent):
    pass

  def __parse_var(self, parent):
    pass

  def __parse_array(self, parent):
    pass

  def __parse_if(self, parent):
    children_nodes = []
    then_found = False
    fi_found = False

    if_node = Node('keyword', 'if', parent)

    current_node = if_node

    for token in self.__token_stream:
      try:
        children_nodes.append(self.__parse_next_token(token, current_node))
      except ThenFoundException:
        then_found = True
        try:
          then_node = Node('keyword', 'then', if_node)
          current_node = then_node
        except ElseFoundException:
          else_node = Node('keyword', 'else', if_node)
          current_node = else_node
        except FiFoundException:
          fi_found = True
          break

    if not then_found:
      raise LanguageSyntaxError('No matching "then" was found for the if.')

    if not fi_found:
      raise LanguageSyntaxError('No matching "fi" was found for the if.')

    node.append_children(*children_nodes)

    return node

  def __parse_then(self, parent):
    raise ThenFoundException('if', 'then')

  def __parse_else(self, parent):
    raise ElseFoundException('if', 'else')

  def __parse_fi(self, parent):
    raise FiFoundException('if', 'fi')

  def __parse_while(self, parent):
    pass

  def __parse_function(self, parent):
    pass

  def __parse_procedure(self, parent):
    pass

  def __parse_return(self, parent):
    pass

  def __parse_call(self, parent):
    pass

  def is_keyword(self, word):
    return word in self.KEYWORDS


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  try:
    p = Parser(args.file_names[0])
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  bootstrap()

