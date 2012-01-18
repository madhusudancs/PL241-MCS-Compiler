from argparse import ArgumentParser

def parse(program):
  src = program.read()
  src_tokens = src.split()
  print src_tokens

if __name__ == '__main__':
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=file, nargs='+',
                      help='name of the input files.')
  args = parser.parse_args()
  parse(args.file_names[0])

