from argparse import ArgumentParser


START_CONTROL_CHARACTERS = ['(', '{', '[']
END_CONTROL_CHARACTERS = [')', '}', ']']


def split_control_characters(token):
  start_tokens_list = []
  while True:
    if not token or (token[0] not in START_CONTROL_CHARACTERS):
      break

    start_tokens_list = start_tokens_list + [token[0]]
    token = token[1:]

  end_tokens_list = []
  while True:
    if not token or (token[-1] not in END_CONTROL_CHARACTERS):
      break

    end_tokens_list = [token[-1]] + end_tokens_list
    token = token[:-1]

  return start_tokens_list + [token] + end_tokens_list


def parse(program):
  src = program.read()
  src_tokens = src.split()
  final_tokens = []
  for i, token in enumerate(src_tokens):
    split_tokens = split_control_characters(token)
    final_tokens.extend(split_tokens)

  print final_tokens:


if __name__ == '__main__':
  parser = ArgumentParser(description='Compiler arguments.')
  parser.add_argument('file_names', metavar='File Names', type=file, nargs='+',
                      help='name of the input files.')
  args = parser.parse_args()
  parse(args.file_names[0])

