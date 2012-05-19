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


from ir import Immediate
from regalloc import Register

# Architecture specific imports
from x86_64 import CALL
from x86_64 import MOV
from x86_64 import SYSCALL


def entry():
  """Returns the precompiled code for starting the code.
  """
  call = CALL()

  # Move exit status to %rdi
  rdi = Register()
  rdi.color = 5
  mov_status = MOV(rdi, Immediate(0))

  # Move exit SYSCALL descriptor number to %rax
  rax = Register()
  rax.color = 0
  mov_sycall = MOV(rax, Immediate(60))

  # Make syscall
  syscall = SYSCALL()

  return (len(call) + len(mov_status) + len(mov_sycall)
          + len(syscall), [call, mov_status, mov_sycall, syscall])


def input_num():
  mov_fd = "\x48\xBF\x00\x00\x00\x00\x00\x00\x00\x00"            # mov 0x0, %rdi

  mov_readaddr = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  sub_address = "\x48\x81\xEE\x08\x00\x00\x00"                   # sub $0x8, %rsi

  mov_numbytes = "\x48\xBA\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov 0xa, %rdx

  mov_syscallnum = "\x48\xB8\x00\x00\x00\x00\x00\x00\x00\x00"    # mov 0x0, %rax

  syscall = "\x0F\x05"                                           # syscall



  xor = "\x48\x33\xC0"                                           # xor %rax, %rax

  mov_rbx = "\x48\xBB\x00\x00\x00\x00\x00\x00\x00\x00"           # mov 0x0, %rbx

  mov_readaddr = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  # sub_address again for the same reason as before, we want to start after
  # the stack address byte.


  # rnext:

  mov_byte = "\x48\x8B\x0E"                                      # mov (%rsi), %rcx

  and_clear = "\x48\x81\xE1\xFF\x00\x00\x00"                     # and 0xff, %rcx

  cmp_tenbytes = "\x48\x81\xF9\x0A\x00\x00\x00"                  # cmp 0xa, %rcx

  je_end = "\x0F\x84\x2F\x00\x00\x00"                            # je  return

  cmp_countten = "\x48\x81\xFB\x0A\x00\x00\x00"                  # cmp 0xa, %rbx

  jg_end = "\x0F\x8F\x22\x00\x00\x00"                            # jg  return

  sub_ascii = "\x48\x81\xE9\x30\x00\x00\x00"                     # sub 0x30, %rcx

  mov_copy = "\x48\x8B\xD0"                                      # mov %rax, %rdx

  shl_three = "\x48\xC1\xE0\x03"                                 # shl 0x3, %rax

  shl_one = "\x48\xD1\xE2"                                       # shl 0x1, %rdx

  add_copy = "\x48\x03\xC2"                                      # add %rdx, %rax

  add_digit = "\x48\x03\xC1"                                     # add %rcx, %rax

  inc_tonextbyte = "\x48\xFF\xC6"                                # inc %rsi

  increment = "\x48\xFF\xC3"                                     # inc %rbx

  jmp_loop = "\xE9\xBA\xFF\xFF\xFF"                              # jmp rnext

  ret = "\xC3"                                                   # return: retq

  # The return value is in %rax according to the Linux AMD64 ABI
  return ''.join([mov_fd, mov_readaddr, sub_address, mov_numbytes,
                  mov_syscallnum, syscall, xor, mov_rbx, mov_readaddr,
                  sub_address, mov_byte, and_clear, cmp_tenbytes, je_end,
                  cmp_countten, jg_end, sub_ascii, mov_copy, shl_three, shl_one,
                  add_copy, add_digit, inc_tonextbyte, increment, jmp_loop, ret])


def output_num():
  xor_rdx = "\x48\x33\xD2"                                      # xor  %rdx, %rdx

  mov_countinit = "\x49\xB8\x00\x00\x00\x00\x00\x00\x00\x00"    # mov  0x0, %r8



  mov_funcarg = "\x48\x8B\xC7"                                  # mov  %rdi, %rax

  mov_baseten = "\x48\xBB\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov  0xa, %rbx


  # wnext:

  div_baseten = "\x48\xF7\xF3"                                  # div  %rbx

  add_ascii = "\x48\x81\xC2\x30\x00\x00\x00"                    # add  0x30, %rdx

  push_ascii = "\x52"                                           # push %rdx

  # xor_rdx                                                     # xor %rdx, %rdx

  inc_count = "\x49\xFF\xC0"                                    # inc  %r8

  cmp_nomore = "\x48\x81\xF8\x00\x00\x00\x00"                   # cmp  0x0, %rax

  jne_loop = "\x0F\x85\xE2\xFF\xFF\xFF"                         # jne  wnext



  # popnext:

  cmp_countend = "\x49\x81\xF8\x00\x00\x00\x00"                 # cmp  0x0, %r8

  jle_ret = "\x0F\x8E\x2C\x00\x00\x00"                          # jle  return

  mov_numbytes = "\x48\xBA\x01\x00\x00\x00\x00\x00\x00\x00"     # mov  0x1, %rdx

  mov_writeaddr = "\x48\x8B\xF4"                                # mov  %rsp, %rsi

  mov_fd = "\x48\xBF\x01\x00\x00\x00\x00\x00\x00\x00"           # mov  0x1, %rdi

  mov_syscallnum = "\x48\xB8\x01\x00\x00\x00\x00\x00\x00\x00"   # mov  0x1, %rax

  syscall = "\x0F\x05"                                          # syscall

  dec_count = "\x49\xFF\xC8"                                    # dec  %r8

  pop_nextbyte = "\x5B"                                         # pop  %rbx

  jmp_loop = "\xE9\xC7\xFF\xFF\xFF"                             # jmp  popnext

  ret = "\xC3"                                                  # return: retq

  return ''.join([xor_rdx, mov_countinit, mov_funcarg, mov_baseten, div_baseten,
                  add_ascii, push_ascii, xor_rdx, inc_count, cmp_nomore,
                  jne_loop, cmp_countend, jle_ret, mov_numbytes, mov_writeaddr,
                  mov_fd, mov_syscallnum, syscall, dec_count, pop_nextbyte,
                  jmp_loop, ret])


def output_newline():

  mov_newline = "\x48\xC7\x84\x24\xF8\xFF\xFF\xFF\x0A\x00\x00\x00"  # mov $0xa, -8(%rsp)

  mov_fd = "\x48\xBF\x01\x00\x00\x00\x00\x00\x00\x00"           # mov $0x1, %rdi

  mov_address = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  sub_address = "\x48\x81\xEE\x08\x00\x00\x00"                  # sub $0x8, %rsi

  mov_numbytes = "\x48\xBA\x01\x00\x00\x00\x00\x00\x00\x00"     # mov $0x1, %rdx

  mov_syscallnum = "\x48\xB8\x01\x00\x00\x00\x00\x00\x00\x00"   # mov $0x1, %rax

  syscall = "\x0F\x05"                                          # syscall

  ret = "\xc3"                                                  # return

  return ''.join([mov_newline, mov_fd, mov_address, sub_address, mov_numbytes,
                  mov_syscallnum, syscall, ret])
