#ifndef TECHMIC_H
#define TECHMIC_H

#include <stdint.h>

#define PROGRAM_SIZE   256
#define MEMORY_SIZE    256
#define NUM_REGISTERS   16

typedef uint16_t prog_size_t;

typedef uint16_t instr_raw_t;
typedef uint16_t operand_t;


typedef union {
  instr_raw_t  raw;
  struct {
    instr_raw_t param : 12;
    instr_raw_t opcode : 4;
  };
} instr_t;

typedef struct {
  instr_raw_t junk : 4;
  instr_raw_t rm : 4;
  instr_raw_t rn : 4;
} param_2r_t;

typedef struct {
  instr_raw_t data : 8;
  instr_raw_t rn : 4;
} param_1r_t;


typedef struct {
  uint8_t pc;
  instr_t program[PROGRAM_SIZE];
  uint8_t memory[MEMORY_SIZE];
  uint8_t reg[NUM_REGISTERS];
} techmic_uc_t;


void techmic_initUc(techmic_uc_t * uc);

int8_t techmic_execUc(techmic_uc_t * uc, prog_size_t stop);

void techmic_dumpUc(const techmic_uc_t * uc, const char * fn);

prog_size_t techmic_assemble(instr_t * prog, prog_size_t maxlen, const char * fn);


#endif
