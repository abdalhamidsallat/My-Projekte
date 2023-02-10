CC = gcc
CFLAGS = -Wall -Wextra -std=c99
LDFLAGS = -lm
OBJFILES = techmic.o techmic_ops.o techmic_fileio.o

all: emu as

emu: $(OBJFILES) emu.o
	$(CC) $(CFLAGS) -o $@ $@.o $(OBJFILES) $(LDFLAGS)

as: $(OBJFILES) as.o
	$(CC) $(CFLAGS) -o $@ $@.o $(OBJFILES) $(LDFLAGS)

clean:
	rm -f $(OBJFILES) $(TARGET) *~
