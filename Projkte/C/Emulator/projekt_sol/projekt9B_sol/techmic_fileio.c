#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>

#include "techmic_fileio.h"


/************************************************************************
 * types and defs
 ***********************************************************************/

typedef struct {
  const char * const file_ext;
  prog_size_t (* readFn)(instr_t *, const char *);
  prog_size_t (* writeFn)(const instr_t *, prog_size_t, const char *);
} techmic_imgtype_t;


#define OUTFILE_NAME_MAXLEN    50
#define OUTFILE_REGISTER_EXT   "reg"
#define OUTFILE_REGISTER_DFLT  "techmic.reg"
#define OUTFILE_MEMORY_EXT     "mem"
#define OUTFILE_MEMORY_DFLT    "techmic.mem"


/************************************************************************
 * private interface
 ***********************************************************************/

static prog_size_t techmic_readProgramBin(instr_t * program, const char * fn);
static prog_size_t techmic_readProgramHex(instr_t * program, const char * fn);

static prog_size_t techmic_writeProgramBin(const instr_t * program, prog_size_t size, const char * fn);
static prog_size_t techmic_writeProgramHex(const instr_t * program, prog_size_t size, const char * fn);

static void techmic_genFilename(char * ofn, const char * fn, const char * ext, size_t olen);


/************************************************************************
 * locals
 ***********************************************************************/

static const techmic_imgtype_t imgTypes[] = {
  {".hex", techmic_readProgramHex, techmic_writeProgramHex},
  {".bin", techmic_readProgramBin, techmic_writeProgramBin}
};

#define EMU_IMGTYPE_NUM  (sizeof(imgTypes) / sizeof(imgTypes[0]))



/************************************************************************
 * public interface implementation
 ***********************************************************************/

prog_size_t techmic_readProgram(instr_t * program, const char * fn)
{
  unsigned len = strlen(fn);
  for (unsigned i = 0; i < EMU_IMGTYPE_NUM; i++) {
    if (! strcmp(fn + len - strlen(imgTypes[i].file_ext), imgTypes[i].file_ext)) {
      return imgTypes[i].readFn(program, fn);
    }
  }

  fprintf(stderr, "ERROR: file format unknown!\n");
  exit(1);
}



prog_size_t techmic_writeProgram(const instr_t * program, prog_size_t size, const char * fn)
{
  unsigned len = strlen(fn);
  for (unsigned i = 0; i < EMU_IMGTYPE_NUM; i++) {
    if (! strcmp(fn + len - strlen(imgTypes[i].file_ext), imgTypes[i].file_ext)) {
      return imgTypes[i].writeFn(program, size, fn);
    }
  }

  fprintf(stderr, "ERROR: file format unknown!\n");
  exit(1);
}



void techmic_dumpUc(const techmic_uc_t * uc, const char * fn)
{
  char ofn[OUTFILE_NAME_MAXLEN + 1];
  FILE * fp;
  uint16_t i;

  if (uc == NULL) {
    fprintf(stderr, "ERROR: NULL pointer in %s", __FUNCTION__);
    exit(1);
  }

  // prepare register dump file
  if (fn == NULL) {
    strcpy(ofn, OUTFILE_REGISTER_DFLT);
  } else {
    techmic_genFilename(ofn, fn, OUTFILE_REGISTER_EXT, OUTFILE_NAME_MAXLEN);
  }

  fp = fopen(ofn, "w");
  if (fp == NULL) {
    fprintf(stderr, "ERROR: cannot create file '%s' in %s", ofn, __FUNCTION__);
    exit(1);
  }

  // dump registers
  printf("registers\n\n");
  for (i = 0; i < NUM_REGISTERS; i++) {
      printf("  R%-2d : %02x\n", i, uc->reg[i]);
      fprintf(fp, "%s%02x", i > 0 ? " " : "", uc->reg[i]);
  }

  fclose(fp);

  // prepare memory dump file
  if (fn == NULL) {
    strcpy(ofn, OUTFILE_MEMORY_DFLT);
  } else {
    techmic_genFilename(ofn, fn, OUTFILE_MEMORY_EXT, OUTFILE_NAME_MAXLEN);
  }

  fp = fopen(ofn, "w");
  if (fp == NULL) {
    fprintf(stderr, "ERROR: cannot create file '%s' in %s", ofn, __FUNCTION__);
    exit(1);
  }

  // dump memory
  printf("\n\nmemory\n\n        ");
  for (i = 0; i < 16; i++) {
    printf("  %1x", i);
  }
  for (i = 0; i < MEMORY_SIZE; i++) {
    if (i % 16 == 0) {
      printf("\n  0x%03x. ", i / 16);
      if (i > 0) {
        fprintf(fp, "\n");
      }
    } else {
      fprintf(fp, " ");
    }
    printf("%02x ", uc->memory[i]);
    fprintf(fp, "%02x", uc->memory[i]);
  }
  printf("\n\n");

  fclose(fp);
}



/************************************************************************
 * private interface implementation
 ***********************************************************************/

prog_size_t techmic_readProgramBin(instr_t * program, const char * fn)
{
  FILE * ifs;
  if ((ifs = fopen(fn, "rb")) == NULL) {
    fprintf(stderr, "ERROR: file '%s' does not exist!\n", fn);
    exit(1);
  }

  // read binary file and correct byte order
  prog_size_t len = fread(program, sizeof(instr_t), PROGRAM_SIZE, ifs);
  fclose(ifs);
  fprintf(stderr, "read %u instructions from bin image '%s'.\n", len, fn);

  for (prog_size_t i = 0; i < len; i++) {
    program[i].raw = ntohs(program[i].raw);
  }

  return len;
}



prog_size_t techmic_readProgramHex(instr_t * program, const char * fn)
{
  FILE * ifs;
  if ((ifs = fopen(fn, "rt")) == NULL) {
    fprintf(stderr, "ERROR: file '%s' does not exist!\n", fn);
    exit(1);
  }

  prog_size_t len = 0;
  while (! feof(ifs)) {
    unsigned tmp;
    if (fscanf(ifs, "%4x", &tmp) > 0) {
      program[len].raw = tmp;
      len++;
    }
  }

  fclose(ifs);
  fprintf(stderr, "read %u instructions from hex image '%s'.\n", len, fn);

  return len;
}



prog_size_t techmic_writeProgramBin(const instr_t * program, prog_size_t size, const char * fn)

{
  FILE * ifs;
  if ((ifs = fopen(fn, "wb")) == NULL) {
    fprintf(stderr, "ERROR: cannot create file '%s'!\n", fn);
    exit(1);
  }

  for (prog_size_t i = 0; i < size; i++) {
    instr_raw_t tmp = htons(program[i].raw);
    fwrite(&tmp, sizeof(tmp), 1, ifs);
    // TODO check return value
  }

  fclose(ifs);
  return size;
}



prog_size_t techmic_writeProgramHex(const instr_t * program, prog_size_t size, const char * fn)
{
  FILE * ifs;
  if ((ifs = fopen(fn, "wt")) == NULL) {
    fprintf(stderr, "ERROR: cannot create file '%s'!\n", fn);
    exit(1);
  }

  for (prog_size_t i = 0; i < size; i++) {
    fprintf(ifs, "%4x", program[i].raw);
    // TODO check return value
  }

  fclose(ifs);
  return size;
}




void techmic_genFilename(char * ofn, const char * fn, const char * ext, size_t olen)
{
    strncpy(ofn, fn, olen);

    char * r = strchr(ofn, '.');
    if (r == NULL) {  // sanity check (no '.' in fn)
      r = ofn + strlen(ofn);
    }

    if (r - ofn + 1 + strlen(ext) > OUTFILE_NAME_MAXLEN) {
      fprintf(stderr, "ERROR: cannot create register dump filename (too long) in %s", __FUNCTION__);
      exit(1);
    }
    strcpy(r + 1, ext);
}
