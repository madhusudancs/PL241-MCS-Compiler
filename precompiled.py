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
  mov_fd = "\x48\xBF\x00\x00\x00\x00\x00\x00\x00\x00"            # mov 0x0, %rdi

  mov_readaddr = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  mov_numbytes = "\x48\xBA\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov 0xa, %rdx

  mov_syscallnum = "\x48\xB8\x00\x00\x00\x00\x00\x00\x00\x00"    # mov 0x0, %rax

  syscall = "\x0F\x05"                                           # syscall



  xor = "\x48\x33\xC0"                                           # xor %rax, %rax

  mov_rbx = "\x48\xBB\x00\x00\x00\x00\x00\x00\x00\x00"           # mov 0x0, %rbx

  mov_readaddr = "\x48\x8B\xF4"                                  # mov %rsp, %rsi



  # rnext:

  mov_byte = "\x48\x8B\x0E"                                      # mov (%rsi), %rcx

  cmp_tenbytes = "\x48\x81\xF9\x0A\x00\x00\x00"                  # cmp 0xa, %rcx

  je_end = "\x0F\x84\x37\x00\x00\x00"                            # je  return

  cmp_countten = "\x48\x81\xFB\x0A\x00\x00\x00"                  # cmp 0xa, %rbx

  jg_end = "\x0F\x8F\x2A\x00\x00\x00"                            # jg  return

  sub_ascii = "\x48\x81\xE9\x30\x00\x00\x00"                     # sub 0x30, %rcx

  and_clear = "\x48\x81\xE1\xFF\x00\x00\x00"                     # and 0xff, %rcx

  mov_copy = "\x48\x8B\xD0"                                      # mov %rax, %rdx

  shl_three = "\x48\xC1\xE0\x03"                                 # shl 0x3, %rax

  shl_one = "\x48\xD1\xE2"                                       # shl 0x1, %rdx

  add_copy = "\x48\x03\xC2"                                      # add %rdx, %rax

  add_digit = "\x48\x03\xC1"                                     # add %rcx, %rax

  inc_tonextbyte = "\x48\xFF\xC6"                                # inc %rsi

  increment = "\x48\xFF\xC3"                                     # inc %rbx

  jmp_loop = "\xE9\xB9\xFF\xFF\xFF"                              # jmp rnext

  ret = "\xC3"                                                   # return: retq

  # The return value is in %rax according to the Linux AMD64 ABI
  return ''.join([mov_fd, mov_readaddr, mov_numbytes, mov_syscallnum, syscall,
                  xor, mov_rbx, mov_readaddr, mov_byte, cmp_tenbytes, je_end,
                  cmp_countten, jg_end, sub_ascii, and_clear, mov_copy,
                  shl_three, shl_one, add_copy, add_digit, inc_tonextbyte,
                  increment, jmp_loop, ret])


def output_num():
  xor_rdx = "\x48\x33\xD2"                                      # xor  %rdx, %rdx

  mov_countinit = "\x48\xB9\x00\x00\x00\x00\x00\x00\x00\x00"    # mov  0x0, %rcx



  mov_funcarg = "\x48\x8B\xC7"                                  # mov  %rdi, %rax

  mov_baseten = "\x48\xBB\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov  0xa, %rbx


  # wnext:

  div_baseten = "\x48\xF7\xF3"                                  # div  %rbx

  add_ascii = "\x48\x81\xC2\x30\x00\x00\x00"                    # add  0x30, %rdx

  push_ascii = "\x52"                                           # push %rdx

  # xor_rdx                                                     # xor %rdx, %rdx

  inc_count = "\x48\xFF\xC1"                                    # inc  %rcx

  cmp_nomore = "\x48\x81\xF8\x00\x00\x00\x00"                   # cmp  0x0, %rax

  jne_loop = "\x0F\x85\xE2\xFF\xFF\xFF"                         # jne  wnext



  # popnext:

  cmp_countend = "\x48\x81\xF9\x00\x00\x00\x00"                 # cmp  0x0, %rcx

  jle_ret = "\x0F\x8E\x2C\x00\x00\x00"                          # jle  return

  mov_numbytes = "\x48\xBA\x01\x00\x00\x00\x00\x00\x00\x00"     # mov  0x1, %rdx

  mov_writeaddr = "\x48\x8B\xF4"                                # mov  %rsp, %rsi

  mov_fd = "\x48\xBF\x01\x00\x00\x00\x00\x00\x00\x00"           # mov  0x1, %rdi

  mov_syscallnum = "\x48\xB8\x01\x00\x00\x00\x00\x00\x00\x00"   # mov  0x1, %rax

  syscall = "\x0F\x05"                                          # syscall

  dec_count = "\x48\xFF\xC9"                                    # dec  %rcx

  pop_nextbyte = "\x5B"                                         # pop  %rbx

  jmp_loop = "\xE9\xC7\xFF\xFF\xFF"                             # jmp  popnext

  ret = "\xC3"                                                  # return: retq

  return ''.join([xor_rdx, mov_countinit, mov_funcarg, mov_baseten, div_baseten,
                  add_ascii, push_ascii, xor_rdx, inc_count, cmp_nomore,
                  jne_loop, cmp_countend, jle_ret, mov_numbytes, mov_writeaddr,
                  mov_fd, mov_syscallnum, syscall, dec_count, pop_nextbyte,
                  jmp_loop, ret])


def output_newline():

  mov_newline = "\x48\xC7\x04\x24\x0A\x00\x00\x00"              # mov $0xa, 0(%rsp)

  mov_fd = "\x48\xBF\x01\x00\x00\x00\x00\x00\x00\x00"           # mov $0x1, %rdi

  mov_address = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  mov_numbytes = "\x48\xBA\x01\x00\x00\x00\x00\x00\x00\x00"     # mov $0x1, %rdx

  mov_syscallnum = "\x48\xB8\x01\x00\x00\x00\x00\x00\x00\x00"   # mov $0x1, %rax

  syscall = "\x0F\x05"                                          # syscall

  ret = "\xc3"                                                  # return

  return ''.join([mov_newline, mov_fd, mov_address, mov_numbytes,
                  mov_syscallnum, syscall, ret])
