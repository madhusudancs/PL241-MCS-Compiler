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
from regalloc import PhysicalRegister

# Architecture specific imports
from x86_64 import CALL
from x86_64 import MOV
from x86_64 import SYSCALL


def entry():
  """Returns the precompiled code for starting the code.
  """
  call = CALL()

  # Move exit status to %rdi
  rdi = PhysicalRegister()
  rdi.color = 5
  mov_status = MOV(rdi, Immediate(0))

  # Move exit SYSCALL descriptor number to %rax
  rax = PhysicalRegister()
  rax.color = 0
  mov_sycall = MOV(rax, Immediate(60))

  # Make syscall
  syscall = SYSCALL()

  return (len(call) + len(mov_status) + len(mov_sycall)
          + len(syscall), [call, mov_status, mov_sycall, syscall])


def input_num():
  mov_fd = "\x48\xBF\x00\x00\x00\x00\x00\x00\x00\x00"            # mov 0x0, %rdi

  mov_readaddr = "\x48\x8B\xF4"                                  # mov %rsp, %rsi

  # We make space for 0x15 = 21 bytes because largest 64-bit number consists
  # of 20 digits, so we make space for these 20 digits and one more byte to
  # store the newline character.
  sub_address = "\x48\x81\xEE\x15\x00\x00\x00"                   # sub $0x15, %rsi

  mov_numbytes = "\x48\xBA\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov 0xa, %rdx

  mov_syscallnum = "\x48\xB8\x00\x00\x00\x00\x00\x00\x00\x00"    # mov 0x0, %rax

  syscall = "\x0F\x05"                                           # syscall



  xor = "\x48\x33\xC0"                                           # xor %rax, %rax

  mov_rbx = "\x48\xBB\x00\x00\x00\x00\x00\x00\x00\x00"           # mov 0x0, %rbx

  # Move the stack pointer to start with again
  # mov_readaddr                                                 # mov %rsp, %rsi

  # sub_address again for the same reason as before, we want to start after
  # the stack address byte.
  # sub_address                                                  # sub $0x15, %rsi

  xor_r9 = "\x4D\x33\xC9"                                        # xor  %r9, %r9

  mov_byte = "\x48\x8B\x0E"                                      # mov (%rsi), %rcx

  and_clear = "\x48\x81\xE1\xFF\x00\x00\x00"                     # and 0xff, %rcx

  # 0x2D is the minus sign in ASCII
  cmp_neg = "\x48\x81\xF9\x2D\x00\x00\x00"                       # cmp 0x2D, %rcx

  jne_neg = "\x0F\x85\x0D\x00\x00\x00"                           # jne rnext

  set_sf = "\x49\xB9\x01\x00\x00\x00\x00\x00\x00\x00"            # mov 0x1, %r9

  inc_tonextbyte = "\x48\xFF\xC6"                                # inc %rsi

  # rnext:

  # Repeat move next bytes and clearing it
  # mov_byte                                                     # mov (%rsi), %rcx

  # and_clear                                                    # and 0xff, %rcx

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

  cmp_sf = "\x49\x81\xF9\x01\x00\x00\x00"                        # cmp 0x1, %r9

  jne_nosign = "\x0F\x85\x03\x00\x00\x00"                        # jne return

  neg_rax = "\x48\xF7\xD8"                                       # neg %rax

  # return:

  ret = "\xC3"                                                   # return: retq

  # The return value is in %rax according to the Linux AMD64 ABI
  return ''.join([mov_fd, mov_readaddr, sub_address, mov_numbytes,
                  mov_syscallnum, syscall, xor, mov_rbx, mov_readaddr,
                  sub_address, xor_r9, mov_byte, and_clear, cmp_neg,
                  jne_neg, set_sf, inc_tonextbyte, mov_byte, and_clear, cmp_tenbytes,
                  je_end, cmp_countten, jg_end, sub_ascii, mov_copy, shl_three,
                  shl_one, add_copy, add_digit, inc_tonextbyte, increment,
                  jmp_loop, cmp_sf, jne_nosign, neg_rax, ret])


def output_num():
  xor_rdx = "\x48\x33\xD2"                                      # xor  %rdx, %rdx

  xor_r9 = "\x4D\x33\xC9"                                      # xor  %r9, %r9

  mov_countinit = "\x49\xB8\x00\x00\x00\x00\x00\x00\x00\x00"    # mov  0x0, %r8



  mov_funcarg = "\x48\x8B\xC7"                                  # mov  %rdi, %rax

  mov_baseten = "\x48\xBB\x0A\x00\x00\x00\x00\x00\x00\x00"      # mov  0xa, %rbx

  cmp_neg = "\x48\x81\xF8\x00\x00\x00\x00" 	                 # cmp  0x0, %rax

  # Jump if not sign
  jns = "\x0F\x89\x0D\x00\x00\x00"                              # jns  wnext

  set_sf = "\x49\xB9\x01\x00\x00\x00\x00\x00\x00\x00"           # mov  0x1, %r9

  neg_rax = "\x48\xF7\xD8"                                      # neg %rax


  # wnext:

  div_baseten = "\x48\xF7\xF3"                                  # div  %rbx

  add_ascii = "\x48\x81\xC2\x30\x00\x00\x00"                    # add  0x30, %rdx

  push_ascii = "\x52"                                           # push %rdx

  # xor_rdx                                                     # xor %rdx, %rdx

  inc_count = "\x49\xFF\xC0"                                    # inc  %r8

  # cmp_neg                                                     # cmp  0x0, %rax

  jne_loop = "\x0F\x85\xE2\xFF\xFF\xFF"                         # jne  wnext



  mov_numbytes = "\x48\xBA\x01\x00\x00\x00\x00\x00\x00\x00"     # mov  0x1, %rdx

  mov_fd = "\x48\xBF\x01\x00\x00\x00\x00\x00\x00\x00"           # mov  0x1, %rdi

  mov_syscallnum = "\x48\xB8\x01\x00\x00\x00\x00\x00\x00\x00"   # mov  0x1, %rax

  cmp_sf = "\x49\x81\xF9\x01\x00\x00\x00"                       # cmp  0x1, %r9

  jne_nosign = "\x0F\x85\x0B\x00\x00\x00"                       # jne  popnext

  # print negative sign, the ASCII value of negative sign is 0x2D
  push_neg = "\x68\x2D\x00\x00\x00"                             # push 0x2D

  mov_writeaddr = "\x48\x8B\xF4"                                # mov  %rsp, %rsi

  syscall = "\x0F\x05"                                          # syscall

  # We are popping to some random register, let the victim be %rbx :P
  pop_nextbyte = "\x5B"                                         # pop %rbx

  # popnext:

  cmp_countend = "\x49\x81\xF8\x00\x00\x00\x00"                 # cmp  0x0, %r8

  jle_ret = "\x0F\x8E\x0E\x00\x00\x00"                          # jle  return

  # Move the address to rsi as argument
  # mov_writeaddr                                               # mov  %rsp, %rsi

  # Make a syscall to write out the byte
  # syscall                                                     # syscall

  dec_count = "\x49\xFF\xC8"                                    # dec  %r8

  # Pop the next byte to %rbx
  # pop_nextbyte                                                # pop  %rbx

  jmp_loop = "\xE9\xE5\xFF\xFF\xFF"                             # jmp  popnext

  ret = "\xC3"                                                  # return: retq

  return ''.join([xor_rdx, xor_r9, mov_countinit, mov_funcarg, mov_baseten,
                  cmp_neg, jns, set_sf, neg_rax, div_baseten, add_ascii,
                  push_ascii, xor_rdx, inc_count, cmp_neg, jne_loop,
                  mov_numbytes, mov_fd, mov_syscallnum, cmp_sf, jne_nosign,
                  push_neg, mov_writeaddr, syscall, pop_nextbyte, cmp_countend,
                  jle_ret, mov_writeaddr, syscall, dec_count, pop_nextbyte,
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
