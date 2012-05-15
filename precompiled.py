# Copyright 2012 Madhusudan C.S.
#
# This file precompiled.py is part of PL241-MCS compiler.
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
# 

"""This file contains the precompiled code.
"""


from x86_64 import CALL


def entry():
  """Returns the precompiled code for starting the code.
  """
  call = CALL()
  return call


def input_num():
  "48 BF 00 00 00 00 00 00 00 00"

  "48 8B F4"

  "48 BA 0A 00 00 00 00 00 00 00"

  "48 B8 00 00 00 00 00 00 00 00"
  "0F 05"
  "48 33 C0"
  "48 BB 00 00 00 00 00 00 00 00"

  "48 8B F4"
  "48 8B 0E"

  "48 81 F9 0A 00 00 00"
  "0F 84 37 00 00 00"

  "48 81 FB 0A 00 00 00"
  "0F 8F 2A 00 00 00"
  "48 81 E9 30 00 00 00"
  "48 81 E1 FF 00 00 00"
  "48 8B D0"
  "48 C1 E0 03"
  "48 D1 E2"
  "48 03 C2"
  "48 03 C1"

  "48 FF C6"
  "48 FF C3"
  "E9 B9 FF FF FF"
  "C3"



def output_num():

  "48 33 D2"

  "48 B9 00 00 00 00 00 00 00 00"
  "48 8B C7"
  "48 BB 0A 00 00 00 00 00 00 00"
  "48 F7 F3"
  "48 81 C2 30 00 00 00"
  "52"

  "48 FF C1"
  "48 81 F8 00 00 00 00"
  "0F 85 E2 FF FF FF"

  "48 81 F9 00 00 00 00"
  "0F 8E 2C 00 00 00"

  "48 BA 01 00 00 00 00 00 00 00"

  "48 8B F4"
  "48 BF 01 00 00 00 00 00 00 00"

  "48 B8 01 00 00 00 00 00 00 00"
  "0F 05"
  "48 FF C9"

  "5B"
  "E9 C7 FF FF FF"
  "C3"


def output_newline():

  "48 C7 04 24 0A 00 00 00"
  "48 BF 01 00 00 00 00 00 00 00"
  "48 8B F4"

  "48 BA 01 00 00 00 00 00 00 00"

  "48 B8 01 00 00 00 00 00 00 00"
  "0F 05"
  "c3"
