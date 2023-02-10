#ifndef TECHMIC_OPS_H
#define TECHMIC_OPS_H

#include "techmic.h"

#include <stdbool.h>


typedef void (* instr_func_t)(techmic_uc_t * uc, operand_t param);
typedef bool (* asm_func_t)(const char * line, operand_t * param);

typedef struct {
  instr_func_t instrFn;
  asm_func_t asmFn;
} techmic_func_t;

void instr_mov_toReg(techmic_uc_t * uc, operand_t param);
void instr_mov_toMem(techmic_uc_t * uc, operand_t param);
void instr_mov_toRegIndir(techmic_uc_t * uc, operand_t param);
void instr_mov_toMemIndir(techmic_uc_t * uc, operand_t param);
void instr_mov_immediate(techmic_uc_t * uc, operand_t param);
void instr_mov_reg(techmic_uc_t * uc, operand_t param);
void instr_add(techmic_uc_t * uc, operand_t param);
void instr_sub(techmic_uc_t * uc, operand_t param);
void instr_mul(techmic_uc_t * uc, operand_t param);
void instr_div(techmic_uc_t * uc, operand_t param);
void instr_and(techmic_uc_t * uc, operand_t param);
void instr_or(techmic_uc_t * uc, operand_t param);
void instr_jz(techmic_uc_t * uc, operand_t param);
void instr_cmp(techmic_uc_t * uc, operand_t param);
void instr_less(techmic_uc_t * uc, operand_t param);
void instr_nop(techmic_uc_t * uc, operand_t param);

bool asm_mov_toReg(const char * line, operand_t * param);
bool asm_mov_toMem(const char * line, operand_t * param);
bool asm_mov_toRegIndir(const char * line, operand_t * param);
bool asm_mov_toMemIndir(const char * line, operand_t * param);
bool asm_mov_immediate(const char * line, operand_t * param);
bool asm_mov_reg(const char * line, operand_t * param);
bool asm_add(const char * line, operand_t * param);
bool asm_sub(const char * line, operand_t * param);
bool asm_mul(const char * line, operand_t * param);
bool asm_div(const char * line, operand_t * param);
bool asm_and(const char * line, operand_t * param);
bool asm_or(const char * line, operand_t * param);
bool asm_jz(const char * line, operand_t * param);
bool asm_cmp(const char * line, operand_t * param);
bool asm_less(const char * line, operand_t * param);
bool asm_nop(const char * line, operand_t * param);


#endif
