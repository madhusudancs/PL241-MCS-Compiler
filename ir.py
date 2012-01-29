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

"""Generates the Intermediate Representation for PL241.

It is on this Intermediate Representation that we perform all the machine
independent optimizations.
"""

import logging

from argparse import ArgumentParser

from parser import Parser

# Module level logger object
LOGGER = logging.getLogger(__name__)




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
    return p
  except LanguageSyntaxError, e:
    print e
    sys.exit(1)

if __name__ == '__main__':
  parsed = bootstrap()
  from pprint import pprint
  pprint(parsed.symbol_table)


