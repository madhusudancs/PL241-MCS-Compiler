The most important TODO item
----------------------------

  * Save myself before my brain explodes!



TODO
----

  * Other than that first item, this document is henceforth declared dead. It is
    becoming harder to maintain a TODO file like this, and making commits for
    every update of this file. Let us do it the proper way. Let us use github's
    issue tracker, that too when it provides one for free!

  * Nested loops where the loop 1 level deep starts just inside the loop
    outer loop at level 0 is totally broken when resolving phis during SSA
    deconstruction. This may be broken for nested if-else blocks, if-else
    blocks nested in loops and all sorts of combinations. Look into it!
  * Instruction branch numbering while deconstructing SSA is getting messed
    up because of the different functions being compiled at different stages.

Items that I don't remember myself
----------------------------------

  * Take care of statSeq which are empty (optional) especially in case of if-then-else-fi, function body and while statements.
  * Handle function branch and return
  * Every call to a function should check if the function is already defined.
  * Implement the parser and stuff for procedures.
  * Function/Procedures cannot take arrays as arguments.
  * For function calls do calling vs formal parameters checking if the numbers match.
  * Take care of variables used but undefined in parser itself.
  * Take care of "return" with no expression case.
  * Array dimension checking in parser.

Bookmarks
---------

  * http://www.dwheeler.com/essays/high-assurance-floss.html for SAT solver.


Done
----

Binary Generation
~~~~~~~~~~~~~~~~~

  * Function calling mecahnism should change. Whenever a function is entered, in
    the prologue subtract the %rsp register to make enough space for the local
    variables effectively creating a stack frame. This was done earlier, but was
    removed because I did not understand well enough why functions were not
    returning. Now I do. To remedy this problem see next TODO item.

  * Once the %rsp register is subtracted, the stack is pointing to a different
    location than %rbp register, Intel x86 gives LEAVE instruction which brings
    back the %rsp register to %rbp and pops %rbp there by setting the state of
    the stack to the state it was when the function entered. So encode LEAVE
    instruction in the function epilogue.