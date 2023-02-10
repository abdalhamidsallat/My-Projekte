#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "techmic_ops.h"


void instr_mov_toReg(techmic_uc_t * uc, operand_t param)
{
  param_1r_t * p = (param_1r_t *)&param;
  uc->reg[p->rn] = uc->memory[p->data];
  fprintf(stderr, "MOV R%u, %u\n", p->rn, p->data);
}


void instr_mov_toMem(techmic_uc_t * uc, operand_t param)
{
  param_1r_t * p = (param_1r_t *)&param;
  uc->memory[p->data] = uc->reg[p->rn];
  fprintf(stderr, "MOV %u, R%u\n", p->data, p->rn);
}


void instr_mov_toRegIndir(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->memory[p->rm];
  fprintf(stderr, "MOV R%u, @R%u\n", p->rn, p->rm);
}


void instr_mov_toMemIndir(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->memory[p->rn] = uc->reg[p->rm];
  fprintf(stderr, "MOV @R%u, R%u\n", p->rn, p->rm);
}


void instr_mov_immediate(techmic_uc_t * uc, operand_t param)
{
  param_1r_t * p = (param_1r_t *)&param;
  uc->reg[p->rn] = p->data;
  fprintf(stderr, "MOV R%u, #%u\n", p->rn, p->data);
}


void instr_mov_reg(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rm];
  fprintf(stderr, "MOV R%u, R%u\n", p->rn, p->rm);
}


void instr_add(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] + uc->reg[p->rm];
  fprintf(stderr, "ADD R%u, R%u\n", p->rn, p->rm);
}


void instr_sub(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] - uc->reg[p->rm];
}


void instr_mul(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] * uc->reg[p->rm];
}


void instr_div(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] / uc->reg[p->rm];
}


void instr_and(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] && uc->reg[p->rm];
}


void instr_or(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] || uc->reg[p->rm];
}


void instr_jz(techmic_uc_t * uc, operand_t param)
{
  param_1r_t * p = (param_1r_t *)&param;
  if (uc->reg[p->rn] == 0) {
    uc->pc += p->data;
  }
  fprintf(stderr, "JZ R%u, %u\n", p->rn, p->data);
}


void instr_cmp(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] == uc->reg[p->rm];
}


void instr_less(techmic_uc_t * uc, operand_t param)
{
  param_2r_t * p = (param_2r_t *)&param;
  uc->reg[p->rn] = uc->reg[p->rn] < uc->reg[p->rm] ? 1 : 0;
  fprintf(stderr, "LESS R%u, R%u\n", p->rn, p->rm);
}


void instr_nop(techmic_uc_t * uc, operand_t param)
{
  // nothing
}



static inline bool asm_convert_1r(const char * fmt, const char * line, operand_t * param)
{
  unsigned r, v;
  int res = sscanf(line, fmt, &r, &v);
  if (res != 2) {
    return false;
  }
  *param = 0;
  param_1r_t * p = (param_1r_t *)param;
  p->rn = r;
  p->data = v;
  return true;
}

static inline bool asm_convert_1ri(const char * fmt, const char * line, operand_t * param)
{
  unsigned r, v;
  int res = sscanf(line, fmt, &v, &r);
  if (res != 2) {
    return false;
  }
  *param = 0;
  param_1r_t * p = (param_1r_t *)param;
  p->rn = r;
  p->data = v;
  return true;
}

static inline bool asm_convert_2r(const char * fmt, const char * line, operand_t * param)
{
  unsigned rn, rm;
  int res = sscanf(line, fmt, &rn, &rm);
  if (res != 2) {
    return false;
  }
  *param = 0;
  param_2r_t * p = (param_2r_t *)param;
  p->rn = rn;
  p->rm = rm;
  return true;
}


bool asm_mov_toReg(const char * line, operand_t * param)
{
  return asm_convert_1r("MOV R%u, %u", line, param);
}


bool asm_mov_toMem(const char * line, operand_t * param)
{
  return asm_convert_1ri("MOV %u, R%u", line, param);
}


bool asm_mov_toRegIndir(const char * line, operand_t * param)
{
  return asm_convert_2r("MOV R%u, @R%u", line, param);
}


bool asm_mov_toMemIndir(const char * line, operand_t * param)
{
  return asm_convert_2r("MOV @R%u, R%u", line, param);
}


bool asm_mov_immediate(const char * line, operand_t * param)
{
  return asm_convert_1r("MOV R%u, #%u", line, param);
}


bool asm_mov_reg(const char * line, operand_t * param)
{
  return asm_convert_2r("MOV R%u, R%u", line, param);
}


bool asm_add(const char * line, operand_t * param)
{
  return asm_convert_2r("ADD R%u, R%u", line, param);
}


bool asm_sub(const char * line, operand_t * param)
{
  return asm_convert_2r("SUB R%u, R%u", line, param);
}


bool asm_mul(const char * line, operand_t * param)
{
  return asm_convert_2r("MUL R%u, R%u", line, param);
}


bool asm_div(const char * line, operand_t * param)
{
  return asm_convert_2r("DIV R%u, R%u", line, param);
}


bool asm_and(const char * line, operand_t * param)
{
  return asm_convert_2r("AND R%u, R%u", line, param);
}


bool asm_or(const char * line, operand_t * param)
{
  return asm_convert_2r("OR R%u, R%u", line, param);
}


bool asm_jz(const char * line, operand_t * param)
{
  return asm_convert_1r("JZ R%u, %u", line, param);
}


bool asm_cmp(const char * line, operand_t * param)
{
  return asm_convert_2r("CMP R%u, R%u", line, param);
}


bool asm_less(const char * line, operand_t * param)
{
  return asm_convert_2r("LESS R%u, R%u", line, param);
}


bool asm_nop(const char * line, operand_t * param)
{
  if (strncmp(line, "NOP", 3) != 0) {
    return false;
  }
  *param = 0;
  return true;
}
