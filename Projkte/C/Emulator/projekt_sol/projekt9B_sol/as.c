#include "techmic.h"
#include "techmic_fileio.h"
#include "techmic_ops.h"

#include <stdio.h>



int main(int argn, char * argv[])
{
  if (argn != 3) {
    fprintf(stderr, "USAGE %s infile outfile\n\n"
      "  infile must be assembly with extension '.asm'\n"
      "  outfile may have extension '.bin' or '.hex'\n\n", argv[0]);
    return -1;
  }

  instr_t prog[PROGRAM_SIZE];

  prog_size_t len;
  len = techmic_assemble(prog, PROGRAM_SIZE, argv[1]);
  techmic_writeProgram(prog, len, argv[2]);


  return 0;
}
