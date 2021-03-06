General Implementation
----------------------

  * Only integers are implemented.

Datastructures
--------------

  * Loop detection algorithm in datastructures.py assumes that the CFG is reducible.

Optimization
------------

  * A lot of optimizations happen after/during liveness analysis too.

    * If the result of the instruction is dead on arrival, the whole instruction is removed

    * If the operand of a phi-instruction is not defined anywhere, no live interval is added for it. If we think about
      the program flow path, it doesn't matter what its value is going to be in that control flow path since it is
      anyway supposed to contain a junk value. So why even bother creating a live range for it and getting a spill?
      So while resolving the phi instructions we don't even attempt to generate move for such operands.

    * If the result of the phi function is dead on arrival, that phi function is completely removed.

  * No Common Subexpression Elimination on CMP instructions since this doesn't work in x86 because x86 CMP instructions
    work by setting condition flags in the flags register and there is no guarantee that these flags are retained
    elsewhere in the program.

Precompiled
-----------

  * InputNum doesn't accept negative numbers as input.

  * For InputNum please enter only 20 digits or less. The current implementation starts storing the bytes at %rsp-21
    and upwards. So the 21st byte must be the newline character, if that is not the case, the return address on the
    stack for the InputNum function is rewritten and we are lost! That is lose the caller information of the InputNum
    function to return back.

Code Generator
--------------

  * Immediate values cannot be moved to memory, they have to be moved to a register and then moved to memory.

  * We cannot add anything to memory directly. The result should be stored in a register and if required then
    moved to memory.

  * Only NEAR jumps are implemented (jumps within signed 32-bits can be the max jump value).

  * Only NEAR CALL and NEAR RET are implemented

  * We do not explicitly setup a StackSegment.

  * Linux AMD64 ABI specification is followed for function calling convention.

  * Register %r15 is used a temporary register for those instructions where both
    the operands are memory operands and such places where we want an intermediate
    temporary register.

  * Function prologue doesn't use ENTER instruction to setup the stack frame,
    but the epilogue uses LEAVE instruction to destroy the stack frame. Stack
    frame for the function is manaully setup using the subtraction of %rsp
    register.

  * CPU is forced to run in 64-bit mode. 64-bit registers are used everywhere.

  * At every place in the program, whenever there is a choice between operands
    of different widths, the operands with the largest widths are used. This is
    done to keep the binary generator simple. However encoding for operands with
    smaller sizes should be trivial with the existing infrastructure.

Register Allocator
------------------

  * The number of physical registers for assignment in the Register Allocator is
    configurable. However currently since the Linux AMD64 ABI is hard coded, this
    may affect the code generator and hence make the binaries not executable.

  * The speed reduction in case of MaxSAT based allocator as compared to general
    SAT solver based allocator, at least until benchmarking with larger source
    files, should be attributed to the fact that akmaxsat doesn't take the input
    from UNIX Pipe but requires us to write the input to the file on disk and then
    read from that file. When I can write a C++ wrapper for the Python code, I
    will have a better way of comparing the two allocators.

  * Additional it should be considered while considering the speed reduction for
    MaxSAT based allocator that this allocator has substantially more number of
    clauses to resolve since we implement coalescing clauses for phi functions.

RegisterAllocator/SAT solver notes
----------------------------------

The details of the strategy for register allocation is as follows. This
strategy makes some simplifying assumptions. This is to get the basic simple
allocator working. As we progress we improve the allocator. But for now the
simplifying assumptions are:

  #. One of the major simplifying assumption is that a variable that is spilled
     is always kept in memory. It is neither reloaded as we do in the case of
     Live Range splitting nor is it re-materialized.

  #. No coalescing is implemented. Coalescing in SSA form graph is nothing but
     assigning all the phi-operands and registers to the same register. We do
     do not do it yet. This is the next first step.

  #. The Live Range Analysis Algorithm has an inherent problem that I had
     realized a while back. While merging live ranges, conceptually I was
     thinking about unions but used to merge live ranges using the lower of the two
     live ranges as the start of new live range and the larger of the two values as
     the end of the new live range. I realized after a while that this was clearly
     wrong because when a live range ends in if-else blocks i.e. different branches
     of the Control Flow Graph, we just take the instruction that gets executed
     the last among the two blocks. This is just so wrong. But I did not know any
     better way to fix this. Now I know what would be better! Take the sets of
     instruction labels and compute their set unions! Stupid me! Why didn't I
     think about this before! This will be my second step.

  #. Every time an allocation fails, we completely re-build the cost function
     priority queue and in turn re-build the whole graph. This is such a waste
     of time. We can do much better than this. This is one of the next steps too.

  #. There is no implementation of re-materialization. This should be implemented
     too.

  #. There is no reason to build the interference graph at all. But it is built as
     a support, to make debugging easier for now. We can build the SAT instances
     directly once we have the sufficient confidence that the allocator works.

  #. No Live Range Splitting is implemented. This is one of the things that must
     be seriously considered.

Here is how the allocator works for now.

  #. Initially we build the interference graph just using the Live Range
     information we have. Reduce it to SAT and try to solve the SAT. If the SAT
     is solved we have our allocation, go to Step 4.

  #. If the allocation fails, we will compute the spill costs. We use a cost
     function to determine the spill costs of each variable. The cost function
     used is the Chaitin's allocator style cost function that computes the
     execution frequency and uses a frequency multiplier of 10 to increase the
     frequency of execution of instructions for each level deep in the loop.
     We build a priority queue of spill costs and pick the one with the least
     spill cost. More about spill costs in the next section.

  #. Then we mark this virtual register as spilled, eliminate it from Live
     Range intervals completely, and we repeat from Step 1.

     Note, when we remove the virtual register from the Live Range intervals,
     when we rebuild everything again, it doesn't appear in the Interference
     graph or the Spill cost priority queue anymore.

  #. Generate the register assignments from the SAT solution.

  #. Deconstruct the SSA.


More about Spill Cost function
------------------------------

  #. The cost function used is the Chaitin' allocator style cost function.

  #. The execution frequency of a particular phi-operand is the execution frequency of the node it comes from. To
     arrive at this, we need to look at how SSA deconstruction happens. When SSA is deconstructed, phi-operands
     are moved to the phi-result at the end of the basic block from where they come. So it all makes sense to use
     that node's execution frequency. This is the sole reason why we store the execution frequency of the node in
     node itself. It is not required to store it in the node otherwise.

  #. The execution frequency of the phi-result is the sum of the execution frequencies of the phi-operands because
     of the same reason as above plus their remaining usage frequencies.

  #. Currently only spilling, no live range splitting

  #. Spilling is implemented as no load/store requirements to optimize for x86_64 architecture.
     "Generalization kills optimization."

Feedback
--------

  * Allow people to optionally use DOT language instead of VCG.

  * It will be nice if it is made more explicit at the beginning of the project (say in the project documentation)
    that people who are going to generate binaries for x86 or similar CISC architectures have a bit more freedom
    to tweak the Intermediate Representation.

  * It will be nice to warn stupid people like me at the start that they are probably going to kill themselves
    if they want to generate binaries for x86 all by hand.

  * Encouraging to read existing compilers code, esp. LLVM's would have been useful, esp the way awesome infrastructure
    they have for optimizations, esp in the register allocator and instruction selection/scheduling.

  * Compare and Branch instructions can perhaps be merged into a single instruction in the intermediate representation?
    It is really painful to handle two separate instructions which are really related and always work together.

