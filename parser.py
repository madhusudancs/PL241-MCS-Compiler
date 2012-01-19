import re

from argparse import ArgumentParser


class Parser(object):
  """Abstracts the entire grammar parser along with building the parse tree.

  This parser is implemented as a home-brewn recursive descent parser with
  some help from regular expression library only for tokenizing.
  """

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

  # The alternative solution to this parse map is to find if a parse function
  # exists for every token parsed where each parse function's name is constructed
  # based on the construct it parses and call if exists based on that. I prefer
  # this approach for readability.
  PARSE_MAP = {
    'main': parse_main,
    'let': parse_let,
    'var': parse_var,
    'array': parse_array,
    'if': parse_if,
    'while': parse_while,
    'function': parse_function,
    'procedure': parse_procedure,
    'return': parse_return,
    'call': parse_call
  }


if __name__ == '__main__':
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=file, nargs='+',
                      help='name of the input files.')
  args = parser.parse_args()
  parse(args.file_names[0])

