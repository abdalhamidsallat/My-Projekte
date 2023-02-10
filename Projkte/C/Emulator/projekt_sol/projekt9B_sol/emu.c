#include "techmic.h"
#include "techmic_fileio.h"
#include "techmic_ops.h"

int main(int argn, char * argv[])
{
  if (argn != 2) {
    return -1;
  }

  techmic_uc_t uc;
  techmic_initUc(&uc);

  prog_size_t len;
  len = techmic_readProgram(uc.program, argv[1]);

  if (techmic_execUc(&uc, len) == 0) {
    techmic_dumpUc(&uc, argv[1]);
    return 0;
  }  else  {
    return -1;
  }
}
