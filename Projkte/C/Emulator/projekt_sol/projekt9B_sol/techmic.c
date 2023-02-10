#include "techmic.h"
#include "techmic_ops.h"

// for getline()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/************************************************************************
 * types and defs
 ***********************************************************************/

#define ASM_CMD_MAXLEN  100  // buffer size for ASM code lines


/************************************************************************
 * private interface
 ***********************************************************************/

int techmic_getline(char * buf, size_t size, FILE * f);


/************************************************************************
 * locals
 ***********************************************************************/

//const instr_func_t INSTR[] = {
const techmic_func_t INSTR[] = {
  { instr_mov_toReg, asm_mov_toReg },
  { instr_mov_toMem, asm_mov_toMem },
  { instr_mov_toRegIndir, asm_mov_toRegIndir },
  { instr_mov_toMemIndir, asm_mov_toMemIndir },
  { instr_mov_immediate, asm_mov_immediate },
  { instr_mov_reg, asm_mov_reg },
  { instr_add, asm_add },
  { instr_sub, asm_sub },
  { instr_mul, asm_mul },
  { instr_div, asm_div },
  { instr_and, asm_and },
  { instr_or, asm_or },
  { instr_jz, asm_jz },
  { instr_cmp, asm_cmp },
  { instr_less, asm_less },
  { instr_nop, asm_nop }
};
#define INSTR_NUM  (sizeof(INSTR) / sizeof(INSTR[0]))


/************************************************************************
 * public interface implementation
 ***********************************************************************/

void techmic_initUc(techmic_uc_t * uc)
{
  memset(uc, 0, sizeof(*uc));
}


int8_t techmic_execUc(techmic_uc_t * uc, prog_size_t stop)
{
  for (; uc->pc < stop; uc->pc++) {
    instr_t i = uc->program[uc->pc];
    //fprintf(stderr, "%04x: opcode %u\n", i.raw, i.opcode);
    INSTR[i.opcode].instrFn(uc, i.param);
  }
  return 0;
}


prog_size_t techmic_assemble(instr_t * prog, prog_size_t maxlen, const char * fn)
{
  FILE * ifs;
  prog_size_t len = 0;

  if ((ifs = fopen(fn, "rb")) == NULL) {
    fprintf(stderr, "ERROR: file '%s' does not exist!\n", fn);
    exit(1);
  }

  char line[ASM_CMD_MAXLEN];
  //while (len < maxlen && (read = getline(&line, &linelen, ifs)) != -1) {
  while (len < maxlen && techmic_getline(line, ASM_CMD_MAXLEN, ifs) != -1) {
    bool found = false;
    for (unsigned i = 0; ! found && i < INSTR_NUM; i++) {
      operand_t param;
      if (INSTR[i].asmFn(line, &param)) {
        fprintf(stderr, "FOUND opcode '%04b' with param %3x\n", i, param);
        instr_t tmp;
        tmp.opcode = i;
        tmp.param = param;
        prog[len] = tmp;
        len++;
        found = true;
      }
    }

    if (! found) {
      fprintf(stderr, "ERROR unknwon assembly instruction in line %u!\n", len+1);
      exit(1);
    }
  }

  return len;
}


/************************************************************************
 * private interface implementation
 ***********************************************************************/

/* TODO move to file I/O ? */
int techmic_getline(char * buf, size_t size, FILE * f)
{
  if (fgets(buf, size, f) == NULL) {
    return -1;
  }

  size_t len = strlen(buf);
  if (buf[len - 1] != '\n') {
    int c;
    do {
      // eat up until '\n' or EOF
      c = getc(f);
    } while (c != '\n' && c != EOF);
  }

  return len;
}
