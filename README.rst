PL241 Compiler
==============

** **Work In Progress** **

**Transparency about my course work:** When you are reading the document
below if you by any chance get into the feeling that I am trying to take
advantage of the open source community for my course work, please come back
and read this again. Initially my thought was to keep all the course work
specific code in a separate private branch, which I would not push to this
repository. The reason for that was, that branch had some materials that
are totally irrelevant in public like my progress report, etc. But with
this private branch approach, it looks like it is hard to convince
open source contributors that I am not trying to get the grades from their
work. So I have decided to make the branch public here and the branch is
called **course-specific**. I will be merging **course-specific** branch to
**master** whenever there is something relevant to **master** committed to
that branch but in the other direction I will only cherry-pick my changes
from **master** to **course-specific**.


What is this all about?
-----------------------

This is a very humble attempt to implement a compiler in Python. Prof. Michael
Franz (http://www.ics.uci.edu/~franz/) offers a graduate level course at
University of California, Irvine, on Advanced Compiler Construction
(http://www.ics.uci.edu/~franz/Site/w12cs241.html). This compiler is being
written as part of the course work. Although I have a very limited knowledge
about compiler construction, I believe that this compiler implements some
of the most basic components that are expected of any reasonable compiler
today.

While implementing this compiler I stumbled upon certain interesting concepts
in compiler construction. I could not find simple and quickly understandable
materials for many of them. In most cases it boiled down to reading the
original papers that introduced the idea, toiling for days to understand
how the concepts introduced in the paper worked before actually sitting
down to implement that concept. I hope that I can be of some help to others
who are trying to do the same. So while implementing these algorithms I
have spent good amount of time to make the code as readable as possible
And what more, it is written in Python! What is more readable than Python
anyway!?

I would really like to make this a repository of algorithms that are used
in writing compilers today. So if you would like to use these algorithms
or think you can improve them further, fork it away! And if you want to
push those changes back to this repository, I will be happy to merge the
pull requests!


What is the status? Is this compiler complete?
----------------------------------------------

The compiler compiles the code to x86_64 binary. But the compiler in place is
only a very basic version. There is so much more to do. If you are interested
in contributing, please contact one of the authors.


Which programming languages is this compiler written for?
---------------------------------------------------------

As it is required for the course, at the moment this compiler is planned to
compile only PL241. I am not sure about the copyrights of the Professor's
material to put the language spec here. So the document that contains
the spec for the language is available from the Professor's website at
http://www.ics.uci.edu/~franz/Site/2012WinterCS241/2012CS241Project.pdf

In the future I would like to make this a compiler collection for major
languages.


What are the target architectures to which this compiler generates binaries?
----------------------------------------------------------------------------

Only x86_64 is planned at the moment. Most ideally I would like to see how
this runs on RISC architectures like ARM.


What additional libraries are required to run this compiler?
------------------------------------------------------------

Since the register allocator is implemeted using an off-the-shelf SAT solver
(which is hard coded to be **glucose** at the moment, you will have to
download the source code from **glucose**'s website:
http://www.lri.fr/~simon/?page=glucose and compile the source. The binary
should be available on your system path and with the exact name
*glucose_static*.

Other than this and the standard Python 2.7 installation, you will not want
any additional libraries. Deliberate care has been taken not to use any
other third-party libraries. However, since argparse is used for command line
parsing, Python 2.6 or earlier doesn't work. But argparse component can be
very easily replaced with the older optparse library code which is at the
end of each file in the bootstrap() function.

This code has not been tested on Python 3.x.


How to compile the code?
------------------------

The source program can be compiled by running the following command::

$ pl241c.py [options] <source-file-name>

The options available at the moment are::

  usage: pl241c.py [-h] [-a] [--cfg] [--interferencevcg] [--optimized]
                   [--optimizedvcg] [-p] [-r] [--regassigned] [-s] [--ssavcg]
                   [-t] [--virtualreg] [--virtualregvcg]
                   File Names [File Names ...]

  Compiler arguments.

  positional arguments:
    File Names          name of the input files.

  optional arguments:
    -h, --help          show this help message and exit
    -a, --dumpall       Dump all the intermediate files and graphsgenerated.
    --cfg               Generate the Visualization Compiler Graph output of the
                        Control Flow Graph.
    --interferencevcg   Generate the Visualization Compiler Graph of the
                        interference graph.
    --optimized         Generates the ouput of the optimizer.
    --optimizedvcg      Generate the Visualization Compiler Graph output of the
                        optimized SSA.
    -p, --parsetreevcg  Generate the Visualization Compiler Graph output of the
                        parse tree.
    -r, --ir            Generate the Intermediate Representation.
    --regassigned       Generate the instructions with registers and phi
                        functions resolved.
    -s, --ssa           Generate the Static Single Assignment.
    --ssavcg            Generate the Visualization Compiler Graph output of the
                        SSA form.
    -t, --dom           Generate the Dominator Tree VCG output.
    --virtualreg        Allocate registers in the virtual space of infinite
                        registers.
    --virtualregvcg     Generate the Visualization Compiler Graph for the
                        virtual registers allocated and liveness computed for
                        the subgraphs.


For convenience, 3 test programs are supplied along with the source of which
two of them intentionally include syntax errors. firsttest.pl241 is expected
to generate binary.

What are the things that this compiler does?
--------------------------------------------

The the source program is parsed and translated to an Intermediate Representation
(IR) and then transformed again to Static Single Assignment (SSA) form. A few
very simple yet powerful optimizations: Common Sub-expression Elimination,
Copy Propagation and Constant Folding are implemented. A register allocator
is implemented using an off-the-shelf SAT solver called GLUCOSE. Then the program
is deconstructed out of SSA form and x86_64 machine code is generated from those
instructions.


What algorithms are implemented?
--------------------------------

  * The parser is a home-brewn solution. It doesn't use lex and yacc, however it does make use of regular expressions.
  * "A Fast Algorithm for Finding Dominators in a Flowgraph" by T. Lengauer and R. E. Tarjan
  * "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" by R. Cytron, J. Ferrante, B. K. Rosen, M. N. Wegman and F. K. Zadeck


Contributing
------------

As mentioned in the previous sections it will be nice to see at least the
following implemented

  * Parsing major programming languages
  * Generating binaries for ARM
  * Various optimization algorithms
  * Instruction scheduling
  * Various types of register allocation algorithms

If you are interested in contributing, send the pull requests!
