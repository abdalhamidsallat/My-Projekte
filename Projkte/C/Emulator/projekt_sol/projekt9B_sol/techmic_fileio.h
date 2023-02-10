#ifndef TECHMIC_FILEIO_H
#define TECHMIC_FILEIO_H

#include "techmic.h"

prog_size_t techmic_readProgram(instr_t * prog, const char * fn);

prog_size_t techmic_writeProgram(const instr_t * prog, prog_size_t size, const char * fn);

void techmic_dumpUc(const techmic_uc_t * uc, const char * fn);

#endif
