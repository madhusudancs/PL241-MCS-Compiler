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

from datastructures import Node


# Module level logger object
LOGGER = logging.getLogger(__name__)

# Regular expression patterns used for parsing.
IDENT_PATTERN = r'[a-zA-Z][a-zA-Z0-9]*'
NUMBER_PATTERN = r'-?\d+'
IDENT_RE = re.compile(IDENT_PATTERN)
NUMBER_RE = re.compile(NUMBER_PATTERN)
TOKEN_RE = re.compile(r'(%s|%s|<-|==|!=|<=|>=|\+|\-|\*|\/|[\n]|[^\t +])' % (
    NUMBER_PATTERN, IDENT_PATTERN))


class ParserBaseException(Exception):
  def __init__(self, msg):
    self._msg = msg

  def __str__(self):
    return '%s: %s' % (self.__class__.__name__, self._msg)


class LanguageSyntaxError(ParserBaseException):
  def __str__(self):
    return 'SyntaxError:%s' % self._msg


class EndControlException(ParserBaseException):
  def __init__(self, begin, end, msg=None):
    msg = msg if msg else ('%s encountered without a matching %s.' % (
        begin, end))
    super(EndControlException, self).__init__(msg)


class ElseFoundException(EndControlException):
  def __init__(self, msg=None):
    super(ElseFoundException, self).__init__('if', 'else', msg)


class FiFoundException(EndControlException):
  def __init__(self, msg=None):
    super(FiFoundException, self).__init__('if', 'fi', msg)


class OdFoundException(EndControlException):
  def __init__(self, msg=None):
    super(OdFoundException, self).__init__('while', 'do', msg)


class RightBracketFoundException(EndControlException):
  def __init__(self, msg=None):
    super(RightBracketFoundException, self).__init__('[', ']', msg)


class RightBraceFoundException(EndControlException):
  def __init__(self, msg=None):
    super(RightBraceFoundException, self).__init__('{', '}', msg)


class RightParenthesisFoundException(EndControlException):
  def __init__(self, msg=None):
    super(RightParenthesisFoundException, self).__init__('(', ')', msg)


class EndOfStatementFoundException(EndControlException):
  def __init__(self, msg=None):
    super(EndOfStatementFoundException, self).__init__('statement', ';', msg)


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
    self.__tokens = None
    self.__line_track = None

    self.__tokenize()

  def __tokenize(self):
    """Splits the entire source code stream into tokens using regular expression.
    """
    initial_tokens = TOKEN_RE.findall(self.src)

    line_num = 1
    self.__tokens = []
    self.__line_track = []

    for i, token in enumerate(initial_tokens):
      if token == '\n':
        line_num += 1
        continue
      self.__tokens.append(token)
      self.__line_track.append(line_num)

    LOGGER.debug('Parsed tokens: %s' % self.__tokens)

    # Initializes the stream to the beginning of the tokens list.
    self.__stream_pointer = 0

  def linenum(self):
    """Returns the current line number in the source program as seen by stream.
    """
    return self.__line_track[self.__stream_pointer]

  def fastforward(self, token):
    """Fast forward the stream upto the point we find the given token.
    """
    try:
      self.__stream_pointer = self.__tokens.index(token)
    except ValueError:
      raise LanguageSyntaxError('"%s" not found' % (token))

  def look_ahead(self):
    """Return the next token in the stream.

    If the end of stream has already been reached, the IndexError returned by
    list is communicated, without handling. This should be handled outside
    the class. This is a way to indicate that there is nothing more in the
    stream to look ahead.
    """
    return self.__tokens[self.__stream_pointer]

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

  def debug(self):
    """Logs all the debug information about the TokenStream.
    """
    LOGGER.debug('Stream Pointer Value: %s' % (self.__stream_pointer))
    LOGGER.debug('Stream Ahead: %s' % (self.__tokens[self.__stream_pointer:]))


class Parser(object):
  """Abstracts the entire grammar parser along with building the parse tree.

  This parser is implemented as a home-brewn recursive descent parser with
  some help from regular expression library only for tokenizing.
  """

  KEYWORDS = [
      'array', 'call', 'do', 'else', 'fi', 'function', 'if', 'let', 'main',
      'od', 'procedure', 'return', 'then', 'var', 'while']

  CONTROL_CHARACTERS_MAP = {
      '.': 'period',
      ',': 'comma',
      ';': 'semicolon',
      '[': 'leftbracket',
      ']': 'rightbracket',
      '{': 'leftbrace',
      '}': 'rightbrace',
      '(': 'leftparen',
      ')': 'rightparen',
      '<-': 'assignment_operator',
      }

  RELATIONAL_OPERATORS = ['==', '!=', '<', '<=', '>', '>=']

  TERM_OPERATORS = ['*', '/']

  EXPRESSION_OPERATORS = ['+', '-']

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

    # The current scope of the program as the parser, parses through the source.
    self.__current_scope = None

    # The program symbol table, stored as dictionary. The keys of this
    # dictionary represents the scope of the symbol whose value is another
    # dictionary whose keys are the symbols and values are the latest value
    # of the symbol.
    self.symbol_table = {}

    self.root = self.__parse()

  def __update_scope(self, scope):
    """Update the current scope, add it to the symbol table if doesn't exist.
    """
    self.__current_scope = scope

    if self.__current_scope not in self.symbol_table:
      self.symbol_table[self.__current_scope] = {}

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
    node = Node('keyword', main)

    # Since main is getting defined here, update the scope to indicate it.
    self.__update_scope('main')

    while True:
      try:
        self.__parse_abstract_var_decl(node)
      except EndOfStatementFoundException:
        continue
      except LanguageSyntaxError, e:
        if str(e).startswith('SyntaxError:%d: Expected "var" or "array" but "'
            % (self.__token_stream.linenum())):
          break
        else:
          raise

    while True:
      try:
        self.__parse_abstract_func_decl(node)
      except LanguageSyntaxError, e:
        if str(e).startswith(
            'SyntaxError:%d: Expected "function" or "procedure" but "' % (
                self.__token_stream.linenum())):
          break
        else:
          raise

    # Although we won't have any more variable declarations, to be on the
    # safer side set the scope back to main since we are done with all the
    # function declarations.
    self.__update_scope('main')

    main_body = Node('abstract', 'funcBody', node)

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token != '{':
      raise LanguageSyntaxError('%d: Expected "{" but "%s" was found.' % (
          self.__token_stream.linenum(), look_ahead_token))

    self.__token_stream.next()

    try:
      self.__parse_abstract_stat_sequence(main_body)
      if self.__token_stream.look_ahead() == '}':
        self.__token_stream.next()
        self.__parse_rightbrace(main_body)
    except RightBraceFoundException:
      pass

    # The last token, which is essentially the end of the stream must be
    # a period token, otherwise there is a syntax error in the program
    # according to the grammar.
    if self.__token_stream.next() != '.':
      raise LanguageSyntaxError('%d: Program does not end with a "."',
                                self.__token_stream.linenum(), )

    return node

  def __parse_abstract_ident(self, parent):
    look_ahead_token = self.__token_stream.look_ahead()
    if IDENT_RE.match(look_ahead_token):
      next_token = self.__token_stream.next()
      Node('ident', next_token, parent)

      # Symbol table should be updated at this point since we found a new name.
      if next_token not in self.symbol_table[self.__current_scope]:
        self.symbol_table[self.__current_scope][next_token] = None
      return next_token

    self.__token_stream.debug()

    raise LanguageSyntaxError('%d: Expected identifier but "%s" found.' % (
        self.__token_stream.linenum(), look_ahead_token))

  def __parse_abstract_number(self, parent):
    look_ahead_token = self.__token_stream.look_ahead()
    if NUMBER_RE.match(look_ahead_token):
      next_token = int(self.__token_stream.next())
      Node('number', next_token, parent)
      return next_token

    raise LanguageSyntaxError('%d: Expected number but "%s" found.' % (
        self.__token_stream.linenum(), look_ahead_token))

  def __parse_abstract_designator(self, parent):
    node = Node('abstract', 'designator', parent)

    self.__parse_abstract_ident(node)

    while self.__token_stream.look_ahead() == '[':
      try:
        self.__token_stream.next()
        self.__parse_abstract_expression(node)
        if self.__token_stream.look_ahead() == ']':
          self.__token_stream.next()
          self.__parse_rightbracket(node)
      except RightBracketFoundException:
        continue

  def __parse_abstract_factor(self, parent):
    node = Node('abstract', 'factor', parent)

    look_ahead_token = self.__token_stream.look_ahead()

    if look_ahead_token == '(':
      try:
        self.__token_stream.next()
        self.__parse_abstract_expression(node)
        if self.__token_stream.look_ahead() == ')':
          self.__token_stream.next()
          self.__parse_rightparen(node)
      except RightParenthesisFoundException:
        pass
    elif look_ahead_token == 'call':
      self.__token_stream.next()
      self.__parse_call(node)
    else:
      try:
        self.__parse_abstract_number(node)
      except LanguageSyntaxError:
        try:
          self.__parse_abstract_designator(node)
        except LanguageSyntaxError:
          look_ahead_token = self.__token_stream.look_ahead()
          if self.is_control_character(look_ahead_token):
            next_token = self.__token_stream.next()
            parser_method = '_Parser_parser_%s' % (
                self.CONTROL_CHARACTERS_MAP[next_token])
            parser_method(node)
          else:
            # Re-raise the exception back if it is not a control character.
            raise

  def __parse_abstract_term(self, parent):
    node = Node('abstract', 'term', parent)

    self.__parse_abstract_factor(node)

    while self.is_term_operator(self.__token_stream.look_ahead()):
      self.__parse_generic_operator(node, self.__token_stream.next())
      self.__parse_abstract_factor(node)

  def __parse_abstract_expression(self, parent):
    node = Node('abstract', 'expression', parent)

    self.__parse_abstract_term(node)

    while self.is_expression_operator(self.__token_stream.look_ahead()):
      self.__parse_generic_operator(node, self.__token_stream.next())
      self.__parse_abstract_term(node)

  def __parse_abstract_relation(self, parent):
    node = Node('abstract', 'relation', parent)

    self.__parse_abstract_expression(node)

    look_ahead_token = self.__token_stream.look_ahead()
    if self.__token_stream.look_ahead() not in self.RELATIONAL_OPERATORS:
      raise LanguageSyntaxError(
          '%d: Relational operator expected but "%s" was found'
              % (self.__token_stream.linenum(), look_ahead_token))

    next_token = self.__token_stream.next()
    self.__parse_generic_operator(node, next_token)

    self.__parse_abstract_expression(node)

  def __parse_let(self, parent):
    node = Node('keyword', 'let', parent)

    self.__parse_abstract_designator(node)

    next_token = self.__token_stream.next()
    if next_token == '<-':
      self.__parse_assignment_operator(node)
    else:
      raise LanguageSyntaxError(
          '%d: <- operator was expected but "%s" was found' % (
          self.__token_stream.linenum(), next_token))

    self.__parse_abstract_expression(node)

  def __parse_call(self, parent):
    node = Node('keyword', 'call', parent)

    self.__parse_abstract_ident(node)

    if self.__token_stream.look_ahead() != '(':
      return

    self.__token_stream.next()

    try:
      if self.__token_stream.look_ahead() == ')':
        self.__token_stream.next()
        self.__parse_rightparen(node)

      self.__parse_abstract_expression(node)
      while self.__token_stream.look_ahead() == ',':
        self.__token_stream.next()
        self.__parse_abstract_expression(node)

      if self.__token_stream.look_ahead() == ')':
        self.__token_stream.next()
        self.__parse_rightparen(node)
    except RightParenthesisFoundException:
      return

  def __parse_if(self, parent):
    node = Node('keyword', 'if', parent)

    self.__parse_abstract_relation(node)

    next_token = self.__token_stream.next()
    if next_token != 'then':
      raise LanguageSyntaxError('%d: Expected "then" but "%s" was found.' % (
          self.__token_stream.linenum(), next_token))

    then_node = Node('keyword', 'then', node)

    try:
      self.__parse_abstract_stat_sequence(then_node)
      if self.__token_stream.look_ahead() == 'else':
        self.__token_stream.next()
        self.__parse_else(node)
      elif self.__token_stream.look_ahead() == 'fi':
        self.__token_stream.next()
        self.__parse_fi(node)
    except ElseFoundException:
      try:
        else_node = Node('keyword', 'else', node)
        self.__parse_abstract_stat_sequence(else_node)
        if self.__token_stream.look_ahead() == 'fi':
          self.__token_stream.next()
          self.__parse_fi(node)
      except FiFoundException:
        return
    except FiFoundException:
      return

  def __parse_else(self, parent):
    raise ElseFoundException()

  def __parse_fi(self, parent):
    raise FiFoundException()

  def __parse_while(self, parent):
    node = Node('keyword', 'while', parent)

    self.__parse_abstract_relation(node)

    next_token = self.__token_stream.next()
    if next_token != 'do':
      raise LanguageSyntaxError('%d: Expected "do" but "%s" was found.' % (
          self.__token_stream.linenum(), next_token))

    do_node = Node('keyword', 'do', node)

    try:
      self.__parse_abstract_stat_sequence(do_node)
      if self.__token_stream.look_ahead() == 'od':
        self.__token_stream.next()
        self.__parse_od(node)
    except OdFoundException:
      return

  def __parse_od(self, parent):
    raise OdFoundException()

  def __parse_return(self, parent):
    node = Node('keyword', 'return', parent)

    look_ahead_token = self.__token_stream.look_ahead()
    if self.is_keyword(look_ahead_token) and look_ahead_token != 'call':
      return

    self.__parse_abstract_expression(node)

  def __parse_var(self, parent):
    node = Node('keyword', 'var', parent)

  def __parse_array(self, parent):
    node = Node('keyword', 'array', parent)

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token != '[':
      raise LanguageSyntaxError('%d: "[" missing from array declaration.' % (
          self.__token_stream.linenum()))

    dimensions = []
    while self.__token_stream.look_ahead() == '[':
      next_token = self.__token_stream.next()
      number = self.__parse_abstract_number(node)
      dimensions.append(int(number))
      look_ahead_token = self.__token_stream.look_ahead()
      if look_ahead_token == ']':
        next_token = self.__token_stream.next()
        continue
      else:
        raise LanguageSyntaxError('%d: Expected "]" but "%s" was found."' % (
            self.__token_stream.linenum(), look_ahead_token))

    return dimensions

  def __parse_function(self, parent):
    node = Node('keyword', 'function', parent)

    self.__parse_abstract_function_procedure(node)

  def __parse_procedure(self, parent):
    node = Node('keyword', 'procedure', parent)

    self.__parse_abstract_function_procedure(node)

  def __parse_abstract_function_procedure(self, parent):
    func_name = self.__parse_abstract_ident(parent)

    # This function's name is still in the previous scope so that it can be
    # called from the function outside this own function. Once we are done
    # with it update the scope for this function
    self.__update_scope(func_name)

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token == '(':
      next_token = self.__token_stream.next()
      try:
        self.__parse_abstract_formal_param(parent)
      except RightParenthesisFoundException:
        look_ahead_token = self.__token_stream.look_ahead()

    if look_ahead_token != ';':
      raise LanguageSyntaxError('%d: Expected ";" but "%s" was found.' % (
          self.__token_stream.linenum(), look_ahead_token))
    next_token = self.__token_stream.next()

    self.__parse_abstract_func_body(parent)

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token != ';':
      raise LanguageSyntaxError('%d: Expected ";" but "%s" was found.' % (
          self.__token_stream.linenum(), look_ahead_token))
    next_token = self.__token_stream.next()

  def __parse_abstract_statement(self, parent):
    next_token = self.__token_stream.next()
    if not self.is_keyword(next_token):
      raise LanguageSyntaxError('%d: Expected a keyword but "%s" was found.' %
          (self.__token_stream.linenum(), next_token))

    try:
      parse_method = getattr(self, '_Parser__parse_%s' % (next_token))

      node = Node('abstract', 'statement', parent)
      parse_method(node)
    except AttributeError:
      raise ParserBaseException('Handler for %s is not implemented.' %
          (next_token))

  def __parse_abstract_stat_sequence(self, parent):
    node = Node('abstract', 'statSeq', parent)

    self.__parse_abstract_statement(node)

    while self.__token_stream.look_ahead() == ';':
      self.__token_stream.next()

      self.__parse_abstract_statement(node)

  def __parse_abstract_type_decl(self, parent):
    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token == 'var':
      next_token = self.__token_stream.next()
      self.__parse_var(parent)
      return
    elif look_ahead_token == 'array':
      next_token = self.__token_stream.next()
      dimensions = self.__parse_array(parent)
      return dimensions

    raise LanguageSyntaxError('%d: Expected "var" or "array" but "%s" was '
        'found.' % (self.__token_stream.linenum(), look_ahead_token))

  def __parse_abstract_var_decl(self, parent):
    node = Node('abstract', 'varDecl', parent)

    dimensions = self.__parse_abstract_type_decl(node)

    ident = self.__parse_abstract_ident(node)
    self.symbol_table[self.__current_scope][ident] = dimensions

    while self.__token_stream.look_ahead() == ',':
      self.__token_stream.next()

      self.__parse_abstract_ident(node)

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token != ';':
      raise LanguageSyntaxError('%d: Expected ";" but "%s" was found' % (
          self.__token_stream.linenum(), look_ahead_token))

    next_token = self.__token_stream.next()

    raise EndOfStatementFoundException()

  def __parse_abstract_func_decl(self, parent):
    look_ahead_token = self.__token_stream.look_ahead()
    if not (look_ahead_token == 'function' or look_ahead_token == 'procedure'):
      raise LanguageSyntaxError(
          '%d: Expected "function" or "procedure" but "%s" was found.' % (
              self.__token_stream.linenum(), look_ahead_token))

    next_token = self.__token_stream.next()
    parser_method = getattr(self, '_Parser__parse_%s' % (next_token))
    parser_method(parent)

  def __parse_abstract_formal_param(self, parent):
    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token == ')':
      self.__parse_rightparen(parent)

    node = Node('abstract', 'formalParam', parent)
    self.__parse_abstract_ident(node)

    while self.__token_stream.look_ahead() == ',':
      self.__token_stream.next()
      self.__parse_abstract_ident(node)

    if self.__token_stream.look_ahead() == ')':
      self.__token_stream.next()
      self.__parse_rightparen(node)

  def __parse_abstract_func_body(self, parent):
    node = Node('abstract', 'funcBody', parent)

    while True:
      try:
        self.__parse_abstract_var_decl(node)
      except EndOfStatementFoundException:
        continue
      except LanguageSyntaxError, e:
        if str(e).startswith('SyntaxError:%d: Expected "var" or "array" but "'
            % (self.__token_stream.linenum())):
          break
        else:
          raise

    look_ahead_token = self.__token_stream.look_ahead()
    if look_ahead_token != '{':
      raise LanguageSyntaxError('%d: Expected "{" but "%s" was found' % (
          self.__token_stream.linenum(), look_ahead_token))

    self.__token_stream.next()

    try:
      look_ahead_token = self.__token_stream.look_ahead()
      if look_ahead_token == '}':
        self.__token_stream.next()
        self.__parse_rightbrace(node)

      self.__parse_abstract_stat_sequence(node)
      look_ahead_token = self.__token_stream.look_ahead()
      if look_ahead_token == '}':
        self.__token_stream.next()
        self.__parse_rightbrace(node)
    except RightBraceFoundException:
      return

  def __parse_leftbracket(self, parent):
    pass

  def __parse_rightbracket(self, parent):
    raise RightBracketFoundException()

  def __parse_leftbrace(self, parent):
    pass

  def __parse_rightbrace(self, parent):
    raise RightBraceFoundException()

  def __parse_leftparen(self, parent):
    pass

  def __parse_rightparen(self, parent):
    raise RightParenthesisFoundException()

  def __parse_semicolon(self, parent):
    pass

  def __parse_comma(self, parent):
    pass

  def __parse_assignment_operator(self, parent):
    node = Node('operator', '<-', parent)

  def __parse_period_operator(self, parent):
    pass

  def __parse_generic_operator(self, parent, token):
    node = Node('operator', token, parent)

  def is_keyword(self, token):
    return token in self.KEYWORDS

  def is_control_character(self, token):
    return token in self.CONTROL_CHARACTERS_MAP

  def is_relational_operator(self, token):
    return token in self.RELATIONAL_OPERATORS

  def is_term_operator(self, token):
    return token in self.TERM_OPERATORS

  def is_expression_operator(self, token):
    return token in self.EXPRESSION_OPERATORS


def bootstrap():
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=str, nargs='+',
                      help='name of the input files.')
  parser.add_argument('-d', '--debug', action='store_true',
                      help='Enable debug logging to the console.')
  parser.add_argument('-g', '--vcg', metavar="VCG", type=str,
                      nargs='?', const=True,
                      help='Generate the Visualization Compiler Graph output.')
  args = parser.parse_args()

  if args.debug:
    LOGGER.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    LOGGER.addHandler(ch)

  try:
    p = Parser(args.file_names[0])

    if args.vcg:
      vcg_file = open(args.vcg, 'w') if isinstance(args.vcg, str) else \
          sys.stdout
      vcg_file.write(p.root.generate_vcg())
      vcg_file.close()

    return p.root

  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  root = bootstrap()

